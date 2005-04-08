/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  GenAxes.cc: Choose one input field to be passed downstream
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   November 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/GeomGrid.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomPoint.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomQuads.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <map>
namespace SCIRun {

class GenAxes : public Module {
private:
#ifdef HAVE_FTGL
  GeomFTGLFontRendererHandle    label_font_;
  GeomFTGLFontRendererHandle    value_font_;
#endif
  // The bounding box of all the input fields
  BBox				bbox_;
  
  // Some basic colors, mainly for debugging
  MaterialHandle		white_;
  MaterialHandle		red_;
  MaterialHandle		green_;

    
  GeomHandle			generateAxisLines(int,int,int);
  GeomHandle			generateLines(const Point  start,
					      const Vector delta,
					      const Vector dir,
					      const double num);

  
  typedef map<string, GuiVar*>	var_map_t;
  typedef var_map_t::iterator	var_iter_t;
  var_map_t			vars_;
  void				add_GuiString(const string &str);
  void				add_GuiDouble(const string &str);
  void				add_GuiInt(const string &str);
  void				add_GuiBool(const string &str);


  const string			get_GuiString(const string &str);
  const double			get_GuiDouble(const string &str);
  const int		        get_GuiInt(const string &str);
  const bool		        get_GuiBool(const string &str);

  void				check_for_changed_fonts();
  
public:
  GenAxes(GuiContext* ctx);
  virtual ~GenAxes();
  virtual void execute();
  virtual void tcl_command(GuiArgs& args, void* userdata);
};

DECLARE_MAKER(GenAxes)
  


GenAxes::GenAxes(GuiContext* ctx)
  : Module("GenAxes", ctx, Filter, "Visualization", "SCIRun"),
#ifdef HAVE_FTGL
    label_font_(0),
    value_font_(0),
#endif
    bbox_(Point(-1.,-1.,-1.),Point(1.,1.,1.)),
    white_(scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20)),
    red_(scinew Material(Color(0,0,0), Color(1,0.4,0.4), Color(1,0.4,0.4), 20)),
    green_(scinew Material(Color(0,0,0), Color(0.4,1.,0.4), Color(0.4,1.,0.4), 20)),
    vars_()
{
  add_GuiString("title");
  add_GuiString("font");
  add_GuiInt("precision");
  add_GuiDouble("squash");
  add_GuiString("valuefont");
  add_GuiString("labelfont");
  add_GuiInt("valuerez");
  add_GuiInt("labelrez");

  for (int i = 0; i < 3; i++) {
    for (int dir = 0; dir < 2; dir++) {
      int axis = i/3*3+(i%3+1)/2+1;
      if (!dir) axis=i/3*3+i%3/2;
      
    ostringstream str;
    str << "Plane-" << i/3*3+i%3/2 << i/3*3+(i%3+1)/2+1 
	<< "-" << axis << "-Axis-";
    const string base = str.str();

    add_GuiString(base+"text");

    add_GuiDouble(base+"absolute");
    add_GuiDouble(base+"divisions");
    add_GuiBool(base+"offset");
    
    add_GuiDouble(base+"range-first");
    add_GuiDouble(base+"range-second");
    add_GuiDouble(base+"min-absolute");
    add_GuiDouble(base+"max-absolute");

    add_GuiInt(base+"minplane");
    add_GuiInt(base+"maxplane");
    add_GuiInt(base+"lines");
    add_GuiInt(base+"minticks");
    add_GuiInt(base+"maxticks");
    add_GuiInt(base+"minlabel");
    add_GuiInt(base+"maxlabel");
    add_GuiInt(base+"minvalue");
    add_GuiInt(base+"maxvalue");

    add_GuiDouble(base+"width");
    add_GuiDouble(base+"double");
    add_GuiDouble(base+"tickangle");
    add_GuiDouble(base+"ticktilt");
    add_GuiDouble(base+"ticksize");
    add_GuiDouble(base+"labelangle");
    add_GuiDouble(base+"labelheight");
    add_GuiDouble(base+"valuesize");
    add_GuiDouble(base+"valuesquash");

    }
  }
}

GenAxes::~GenAxes()
{
}

//#define DEBUGPRINT  std::cerr << "[set " << str << "] { " << ret << " } " << std::endl; std::cerr.flush();
#define DEBUGPRINT

const string
GenAxes::get_GuiString(const string &str) {
  string ret("");
  var_iter_t pos = vars_.find(str);
  if (pos == vars_.end()) return ret;  
  if (GuiString *var = dynamic_cast<GuiString *>((*pos).second)) {
    ret = var->get();
  }
  DEBUGPRINT;
  return ret;
}


const double
GenAxes::get_GuiDouble(const string &str) {
  double ret = 0.0;
  var_iter_t pos = vars_.find(str);
  if (pos == vars_.end()) return ret;  
  if (GuiDouble *var = dynamic_cast<GuiDouble *>((*pos).second)) {
    ret = var->get();
  }
  DEBUGPRINT;
  return ret;
}


const int
GenAxes::get_GuiInt(const string &str) {
  int ret = 0;
  var_iter_t pos = vars_.find(str);
  if (pos == vars_.end()) return ret;  
  if (GuiInt *var = dynamic_cast<GuiInt *>((*pos).second)) {
    ret = var->get();
  }
  DEBUGPRINT;
  return ret;
}

const bool
GenAxes::get_GuiBool(const string &str) {
  bool ret = false;
  var_iter_t pos = vars_.find(str);
  if (pos == vars_.end()) return ret;  
  if (GuiInt *var = dynamic_cast<GuiInt *>((*pos).second)) {
    ret = (var->get()?true:false);
  }
  DEBUGPRINT;
  return ret;
}

//#define DEBUGPRINT std::cerr << "adding GuiVar: " << str << std::endl; std::cerr.flush();
//#define DEBUGPRINT
void
GenAxes::add_GuiString(const string &str) {
  DEBUGPRINT;
  vars_.insert(make_pair(str,scinew GuiString(ctx->subVar(str))));
}


void
GenAxes::add_GuiDouble(const string &str) {
  DEBUGPRINT;
  vars_.insert(make_pair(str,scinew GuiDouble(ctx->subVar(str))));
}


void
GenAxes::add_GuiInt(const string &str) {
  DEBUGPRINT;
  vars_.insert(make_pair(str,scinew GuiInt(ctx->subVar(str))));
}

void
GenAxes::add_GuiBool(const string &str) {
  DEBUGPRINT;
  add_GuiInt(str);
}

GeomHandle
GenAxes::generateLines(const Point  start,
		       const Vector delta,
		       const Vector dir,
		       const double num)
{
  GeomLines *lines = scinew GeomLines();
  for (int i = 0; i <= Round(num); i++) {
    const Point pos = start + delta*i;    
    lines->add(pos, pos + dir);
  }
  
  return scinew GeomMaterial(lines, white_);
}

void 
GenAxes::tcl_command(GuiArgs& args, void* userdata) {
  check_for_changed_fonts();
  if(args.count() < 2){
    args.error("ShowField needs a minor command");
    return;
  } else {
    Module::tcl_command(args, userdata);
  }
}




GeomHandle
GenAxes::generateAxisLines(int prim, int sec, int ter)
{
  GeomGroup *group = scinew GeomGroup();

  const double scale = bbox_.diagonal().maxComponent();
  const double radians = 2.0*M_PI/360.0;
    
  for (int dir = 0; dir < 2; dir++) {
    int axisleft   = (dir?prim:sec);
    int axisup     = (dir?sec:prim);
    int axisnormal = ter;

    ostringstream str;   
    str << "Plane-" << prim << sec << "-" << axisleft << "-Axis-";
    const string base = str.str();

    // ********** THIS DOESNT BELONG HERE ***************
    var_iter_t pos = vars_.find(base+"range-first");
    GuiDouble *var = NULL;
    if (pos != vars_.end() && (var = dynamic_cast<GuiDouble *>((*pos).second))) {
      var->set(bbox_.min().asVector()[axisleft]);
    }

    pos = vars_.find(base+"range-second");
    if (pos != vars_.end() && (var = dynamic_cast<GuiDouble *>((*pos).second))) {
      var->set(bbox_.max().asVector()[axisleft]);
    }
    // **************************************************

   
    Point origin(bbox_.min());
    Vector originv(bbox_.min().asVector());    
    originv[axisleft] = get_GuiDouble(base+"range-first");
    origin = originv.asPoint();

    Vector primary(0.,0.,0.);
    primary[axisleft] = get_GuiDouble(base+"range-second")-get_GuiDouble(base+"range-first");
    primary[axisup] = 0.0;
    primary[axisnormal] = 0.0;
    if (primary.length() < 10e-6) return group;

    const Vector left = primary * (1.0/primary.length());

    Vector secondary(bbox_.diagonal());
    secondary[axisleft] = 0.0;
    secondary[axisnormal] = 0.0;

    if (secondary.length() < 10e-6) return group;

    const Vector up = secondary * (1.0/secondary.length());
    
    Vector tertiary(bbox_.diagonal());
    tertiary[axisleft] = 0.0;
    tertiary[axisup] = 0.0;
        
    Vector normal = tertiary;
    if (normal.length() < 10e-6)
      normal = Cross(left,up);
    normal.normalize();
    Vector nnormal = normal;
    nnormal *= -1.0;

    const int num = Round(get_GuiDouble(base+"divisions"));

    Vector delta = primary;
    delta.normalize();
    GuiDouble *absolute_var = dynamic_cast<GuiDouble *>(vars_[base+"absolute"]);
    if (absolute_var->valid()) 
      delta *= absolute_var->get();
    else 
      delta = primary/num;

    Vector nsecondary = -secondary;
    
    GeomHandle obj;
    
    GeomLines *lines = scinew GeomLines();
    double line_width = get_GuiDouble(base+"width");
    if (line_width < 0.05) line_width = 0.05;
    lines->setLineWidth(line_width);


    for (int i = 0; i <= num; i++) {
      const Point pos = origin + delta*i;    
      lines->add(pos, pos + secondary);
    }
    
    obj = scinew GeomMaterial(lines, white_);


    int show = get_GuiInt(base+"lines");
    GeomGroup *grid = scinew GeomGroup();
    grid->add(scinew GeomSwitch(scinew GeomCull(obj,(show<2)?0:&normal),show>0));

    // Create Tickmarks
    Vector tick_vec = secondary;
    tick_vec[axisleft] = 0.0;
    tick_vec[axisnormal] = 0.0;
    tick_vec[axisup] = secondary[axisup]<0.0?-1.:1.;

    Point Zero(0.,0.,0.), One = tick_vec.asPoint();

    GeomLines *tickline = scinew GeomLines();
    tickline->add(Zero, white_,One,white_);
    tickline->setLineWidth(line_width);

    double tilt = radians*get_GuiDouble(base+"ticktilt");
    Transform trans;
    Vector max_offset = tick_vec;
    trans.pre_scale(Vector(1.,1.,1.)*scale*get_GuiDouble(base+"ticksize")/100.);
    trans.project_inplace(max_offset);
    trans.pre_rotate(tilt,normal);
    trans.project_inplace(tick_vec);


    GeomTransform *tilted_tick = scinew GeomTransform(tickline, trans);

    Vector textup = secondary;
    textup.normalize();
    textup *= scale*get_GuiDouble(base+"valuesize")/100.0;

    max_offset.normalize();
    //    max_offset *= textup.length()*1.1;
    max_offset *= 1.05*scale*(get_GuiDouble(base+"ticksize")+
			     get_GuiDouble(base+"valuesize")*1.1)/100.0;


    GeomGroup *ticks = scinew GeomGroup();
    GeomGroup *values = scinew GeomGroup();
#ifdef HAVE_FTGL
    const int precision = get_GuiInt("precision");
#endif
    Vector squash(1.,1.,1.);
    squash[axisleft] = 
      get_GuiDouble(base+"valuesquash")*get_GuiDouble("squash");
    
    for (int i = 0; i <= num; i++) {
      GeomTransform *ttick = scinew GeomTransform(tilted_tick);
      ttick->translate(delta*i);
      ticks->add(ttick);
      
#ifdef HAVE_FTGL
      if (value_font_.get_rep()) {
	ostringstream ostr;
	ostr.precision(precision);
	ostr << (origin+delta*i).asVector()[axisleft];
	//      value_font_->set_resolution(textup.length()*5,72);
	GeomTextTexture *value_text = scinew GeomTextTexture
	  (value_font_, ostr.str(), Zero, textup, primary);
	value_text->set_anchor(GeomTextTexture::c);
	if (axisleft == 1 && axisup == 0) value_text->up_hack_ = true;
	GeomTransform *value = scinew GeomTransform(value_text);
	value->scale(squash);
	value->translate(delta*i+tick_vec+textup*0.55);
	values->add(value);
      }
#endif
    }
    
    //ticks->add(values);
    
    double angle = radians*get_GuiDouble(base+"tickangle");
    trans.load_identity();
    trans.pre_rotate(angle,left);
    trans.pre_translate((origin+secondary).asVector());

    obj = scinew GeomTransform(ticks, trans);
    show = get_GuiInt(base+"maxticks");
    obj = scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&secondary),show>0);
    grid->add(scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&normal),show>0));
#ifdef HAVE_FTGL
    if (value_font_.get_rep()) {
      obj = scinew GeomTransform(values,trans);
      show = get_GuiInt(base+"maxvalue");
      obj = scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&secondary),show>0);
      grid->add(scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&normal),show>0));
    }
#endif


    
    // Min Ticks
    trans.load_identity();
    tick_vec[axisleft] = 0.0;
    tick_vec[axisnormal] = 0.0;
    tick_vec[axisup] = (secondary[axisup]>0.0)?-1.:1.;

    One = tick_vec.asPoint();

    tickline = scinew GeomLines();
    tickline->add(Zero, white_,One,white_);
    tickline->setLineWidth(line_width);

    Vector min_offset = tick_vec;
    trans.pre_scale(Vector(1.,1.,1.)*scale*get_GuiDouble(base+"ticksize")/100.);
    trans.project_inplace(min_offset);
    trans.pre_rotate(-tilt,normal);
    trans.project_inplace(tick_vec);

    min_offset.normalize();
    min_offset *= scale*(get_GuiDouble(base+"ticksize")+
			 get_GuiDouble(base+"valuesize")*1.1)/100.0;

//    min_offset *= textup.length()*1.1;

    tilted_tick = scinew GeomTransform(tickline, trans);
    ticks = scinew GeomGroup();    
    values = scinew GeomGroup();

    for (int i = 0; i <= num; i++) {
      GeomTransform *ttick = scinew GeomTransform(tilted_tick);
      ttick->translate(delta*i);
      ticks->add(ttick);
      
#ifdef HAVE_FTGL
      if (value_font_.get_rep()) {
	ostringstream ostr;
	ostr.precision(3);
	ostr << (origin+delta*i).asVector()[axisleft];
	GeomTextTexture *value_text = scinew GeomTextTexture
	  (value_font_, ostr.str(), Zero, textup, primary);
	value_text->set_anchor(GeomTextTexture::c);
	if (axisleft == 1 && axisup == 0) value_text->up_hack_ = true;
	GeomTransform *value = scinew GeomTransform(value_text);
	value->scale(squash);
	value->translate(delta*i+tick_vec-textup*0.55);
	values->add(value);
      }
#endif
    }

    angle = radians*get_GuiDouble(base+"tickangle");
    trans.load_identity();
    trans.pre_rotate(-angle,left);
    trans.pre_translate(origin.asVector());

    obj = scinew GeomTransform(ticks, trans);
    show = get_GuiInt(base+"minticks");
    obj = scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&nsecondary),show>0);
    grid->add(scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&normal),show>0));
#ifdef HAVE_FTGL
    if (value_font_.get_rep()) {
      obj = scinew GeomTransform(values, trans);
      show = get_GuiInt(base+"minvalue");
      obj = scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&nsecondary),show>0);
      grid->add(scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&normal),show>0));
    }
#endif


#if 0
    ticks = scinew GeomGroup();
    for (int i = 0; i <= Round(divisions); i++) {
      GeomTransform *ttick = scinew GeomTransform(tick);
      tick->rotate(M_PI,primary);
      ttick->translate((origin+delta*i).asVector());
      ticks->add(ttick);
    }

    show = get_GuiInt(base+"minticks");
    obj = scinew GeomSwitch(scinew GeomCull(ticks, show<2?0:&nsecondary),show>0);
    grid->add(scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&normal),show>0));
#endif

#ifdef HAVE_FTGL
    if (label_font_.get_rep()) {
      // Create Label
      trans.load_identity();
      trans.pre_rotate(radians*get_GuiDouble(base+"labelangle"),left);
      trans.project(up,textup);
      textup.normalize();
      textup *= scale*get_GuiDouble(base+"labelheight") / 100.00;
      
      GeomTextTexture *label;
      
      //    const string text = get_GuiString(base+"text"); 
      const string text(get_GuiString(base+"text"));//J^A^3^A^L   <<");
      //    label_font_->set_resolution(textup.length()*10,72);
      label = scinew GeomTextTexture
	(label_font_, text, origin+secondary+textup/2.+primary/2+max_offset, 
	 textup, primary);
      if (axisleft == 1 && axisup == 0) label->up_hack_ = true;
      label->set_anchor(GeomTextTexture::c);
      
      show = get_GuiInt(base+"maxlabel");
      obj = scinew GeomSwitch(scinew GeomCull(label,(show<2?0:&secondary)),show>0);    
      grid->add(scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&normal),show>0));
      label = scinew GeomTextTexture
	(label_font_, text, origin-textup/2+primary/2+min_offset, textup, primary);
      if (axisleft == 1 && axisup == 0) label->up_hack_ = true;
      label->set_anchor(GeomTextTexture::c);
      
      show = get_GuiInt(base+"minlabel");
      obj = scinew GeomSwitch(scinew GeomCull(label,show<2?0:&nsecondary),show>0);
      grid->add(scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&normal),show>0));
    }
#endif

    obj = grid;
    GeomTransform *mirror = scinew GeomTransform(grid);
    Vector center = origin.asVector()+bbox_.diagonal()/2.;
    
    mirror->translate(-center);
    mirror->rotate(M_PI,up);
    mirror->translate(center);
    
    //    show = get_GuiInt(base+"minplane");
    //obj = scinew GeomSwitch(scinew GeomCull(grid, show<2?0:&normal),show>0);
    //group->add(obj);

    //show = get_GuiInt(base+"maxplane");
    //obj = scinew GeomSwitch(scinew GeomCull(mirror, show<2?0:&nnormal),show>0);
    //group->add(obj);
    
    //    grid->add(scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&normal),show>0));
    group->add(grid);
    group->add(mirror);
  }

  GeomPoints *points = scinew GeomPoints();
  points->add(bbox_.min());
  group->add(points);

  return group;
}
		   

void
GenAxes::execute()
{
  
  update_state(NeedData);

  port_range_type range = get_iports("Field");
  if (range.first == range.second) return;

  port_map_type::iterator pi = range.first;
  
  bbox_.reset();
  while (pi != range.second) {
    FieldIPort *iport = (FieldIPort *)get_iport(pi->second);
    FieldHandle fh;
    if (iport->get(fh) && fh.get_rep()) {
      bbox_.extend(fh->mesh()->get_bounding_box());
    }
    ++pi;
  }

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Axes");
  ogeom->delAll();

  //ostringstream str;
  //  int prim = i/3*3+i%3/2;
  //int sec = i/3*3+(i%3+1)/2+1;
  //int ter = i-(i+1)%3+1;
  
  //str << "Plane-" << prim << sec << "-" << ter << "-Axis-";
  //ogeom->addObj(generateAxisLines(prim,sec,ter),str.str());
  //  check_for_changed_fonts();
  ogeom->addObj(generateAxisLines(0,1,2),string("XY Plane"));
  ogeom->addObj(generateAxisLines(1,2,0),string("XZ Plane"));
  ogeom->addObj(generateAxisLines(0,2,1),string("YZ Plane"));

  ogeom->flushViews();
}


void GenAxes::check_for_changed_fonts() {
#ifdef HAVE_FTGL
  string filename = get_GuiString("labelfont");
  if (label_font_.get_rep()) {
    if (label_font_->filename() != filename && 
	gui->eval("validFile "+filename) == "1") {
      label_font_ = 0;
    } else if (label_font_->get_ptRez() != get_GuiInt("labelrez")+1) {
      label_font_ = 0;
    }
  }
    
  if (!label_font_.get_rep() && gui->eval("validFile "+filename) == "1") {
    std::cerr << "loading " << filename;
    label_font_ = scinew GeomFTGLFontRenderer(filename,
					      get_GuiInt("labelrez")+1,72);
  }
  
  filename = get_GuiString("valuefont");
  if (value_font_.get_rep()) {
    if (value_font_->filename() != filename && 
	gui->eval("validFile "+filename) == "1") {
      value_font_ = 0;
    } else if (value_font_->get_ptRez() != get_GuiInt("valuerez")+1) {
      value_font_ = 0;
    }
  }

  
  if (!value_font_.get_rep() && gui->eval("validFile "+filename) == "1") {
    std::cerr << "loading " << filename << "rez of " << get_GuiInt("valuerez") << std::endl;
    value_font_ = scinew GeomFTGLFontRenderer(filename, 
					      get_GuiInt("valuerez")+1,72);
  }
#endif
}

} // End namespace SCIRun

