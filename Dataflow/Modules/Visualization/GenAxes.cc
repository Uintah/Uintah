/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Core/Geometry/BBox.h>

#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomQuads.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomGrid.h>
#include <Core/Geom/GeomText.h>

#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <map>

namespace SCIRun {

class PSECORESHARE GenAxes : public Module {
private:
#ifdef HAVE_FTGL
  GeomFTGLFontRendererHandle    label_font_;
  GeomFTGLFontRendererHandle    value_font_;
#endif
  vector<Vector>		axes_;
  vector<Point>			origins_;
  //vector<Vector&>		axes_permutations_;
  
  BBox				bbox_;
  MaterialHandle		white_;
  MaterialHandle		red_;
  MaterialHandle		green_;

    
  GeomHandle			generateAxisLines(int,int,int);
  GeomHandle			generateLines(const Point  start,
					      const Vector delta,
					      const Vector dir,
					      const double num);


  GeomHandle			generateText(const Point  start,
					     const Vector delta,
					     const Vector up,
					     const pair<double,double> range,
					     const int num);
  
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

public:
  GenAxes(GuiContext* ctx);
  virtual ~GenAxes();
  virtual void execute();
  virtual void tcl_command(GuiArgs& args, void* userdata);
};

DECLARE_MAKER(GenAxes)

GenAxes::GenAxes(GuiContext* ctx)
  : Module("GenAxes", ctx, Filter, "Visualization", "SCIRun"),
    white_(scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20)),
    red_(scinew Material(Color(0,0,0), Color(1,0.4,0.4), Color(1,0.4,0.4), 20)),
    green_(scinew Material(Color(0,0,0), Color(0.4,1.,0.4), Color(0.4,1.,0.4), 20)),
    vars_()
{
#ifdef HAVE_FTGL
  label_font_ = scinew GeomFTGLFontRenderer("/gozer/fonts/scirun.ttf");
  value_font_ = scinew GeomFTGLFontRenderer("/gozer/fonts/scirun.ttf");
# endif
  
  add_GuiString("title");
  add_GuiString("font");
  add_GuiInt("precision");
  add_GuiDouble("squash");
  add_GuiString("value-font");
  add_GuiString("label-font");
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
  DEBUGPRINT
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
  for (int i = 0; i <= int(round(num)); i++) {
    const Point pos = start + delta*i;    
    lines->add(pos, pos + dir);
  }
  
  return scinew GeomMaterial(lines, white_);
}

void 
GenAxes::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2){
    args.error("ShowField needs a minor command");
    return;
  }
  if (args[1] == "setLabelFont") {
#ifdef HAVE_FTGL
    label_font_ = scinew GeomFTGLFontRenderer(args[2]);
#endif
    want_to_execute();
  } else 
  if (args[1] == "setValueFont") {
#ifdef HAVE_FTGL
    value_font_ = scinew GeomFTGLFontRenderer(args[2]);
#endif
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}


GeomHandle
GenAxes::generateText(const Point  start,
		      const Vector delta,
		      const Vector up,
		      const pair<double,double> range,
		      const int num)
{
  GeomGroup *texts = scinew GeomGroup();
  GeomLines *red = scinew GeomLines();
  GeomLines *green = scinew GeomLines();
  const double dr = (range.second - range.first)/double(num);
  
  std::cerr << "Generating Axis: start: " << start  << "  Delta: " << delta 
	    << "   Up: " << up << std::endl;

  Vector scaled_delta = delta;
  scaled_delta.normalize();
  scaled_delta *= up.length();
  for (int i = 0; i <= num; i++) {
    ostringstream str;
    str.clear();
    str.precision(2);
    str << range.first+dr*i;
    const Point pos = start + delta*i;
#ifdef HAVE_FTGL
    GeomTextTexture *text = scinew GeomTextTexture(value_font_,
						   str.str(),
						   pos,
						   up,
						   delta);

    red->add(pos,pos+scaled_delta);
    green->add(pos,pos+up);
    text->set_anchor(GeomTextTexture::n);
    texts->add(text);
#endif
  }
  texts->add(scinew GeomMaterial(red,red_));
  texts->add(scinew GeomMaterial(green,green_));
  return texts;
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
    GuiDouble *var;
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

    const Vector left = primary * (1.0/primary.length());

    Vector secondary(bbox_.diagonal());
    secondary[axisleft] = 0.0;
    secondary[axisnormal] = 0.0;

    const Vector up = secondary * (1.0/secondary.length());
    

    Vector tertiary(bbox_.diagonal());
    tertiary[axisleft] = 0.0;
    tertiary[axisup] = 0.0;
        
    Vector normal = tertiary;
    normal.normalize();
    Vector nnormal = normal;
    nnormal *= -1.0;

    Vector delta = primary;
    delta.normalize();
    delta *= get_GuiDouble(base+"absolute");

    Vector nsecondary = -secondary;
    
    GeomHandle obj;
    GeomLines *lines;
    
    double divisions = get_GuiDouble(base+"divisions");
    double line_width = get_GuiDouble(base+"width");
    obj = generateLines(origin, delta, secondary, divisions);  
    lines = dynamic_cast<GeomLines *>
      (dynamic_cast<GeomMaterial *>(obj.get_rep())->get_child().get_rep());
    ASSERTMSG(lines,"Cannot cast lines to GeomLines");
    lines->setLineWidth(line_width);

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
    const int precision = get_GuiInt("precision");
    Vector squash(1.,1.,1.);
    squash[axisleft] = 
      get_GuiDouble(base+"valuesquash")*get_GuiDouble("squash");
    
    for (int i = 0; i <= int(round(divisions)); i++) {
      GeomTransform *ttick = scinew GeomTransform(tilted_tick);
      ttick->translate(delta*i);
      ticks->add(ttick);
      
#ifdef HAVE_FTGL
      ostringstream ostr;
      ostr.precision(precision);
      ostr << (origin+delta*i).asVector()[axisleft];
      GeomTextTexture *value_text = scinew GeomTextTexture
	(value_font_, ostr.str(), Zero, textup, primary);
      value_text->set_anchor(GeomTextTexture::c);
      GeomTransform *value = scinew GeomTransform(value_text);
      value->scale(squash);
      value->translate(delta*i+tick_vec+textup*0.55);
      values->add(value);      
#endif
    }
    ticks->add(values);
    
    double angle = radians*get_GuiDouble(base+"tickangle");
    trans.load_identity();
    trans.pre_rotate(angle,left);
    trans.pre_translate((origin+secondary).asVector());

    obj = scinew GeomTransform(ticks, trans);


    show = get_GuiInt(base+"maxticks");
    obj = scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&secondary),show>0);
    grid->add(scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&normal),show>0));

    
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

    for (int i = 0; i <= int(round(divisions)); i++) {
      GeomTransform *ttick = scinew GeomTransform(tilted_tick);
      ttick->translate(delta*i);
      ticks->add(ttick);
      
#ifdef HAVE_FTGL
      ostringstream ostr;
      ostr.precision(3);
      ostr << (origin+delta*i).asVector()[axisleft];
      GeomTextTexture *value_text = scinew GeomTextTexture
	(value_font_, ostr.str(), Zero, textup, primary);
      value_text->set_anchor(GeomTextTexture::c);
      GeomTransform *value = scinew GeomTransform(value_text);
      value->scale(squash);
      value->translate(delta*i+tick_vec-textup*0.55);
      values->add(value);      
#endif
    }
    ticks->add(values);
    
    angle = radians*get_GuiDouble(base+"tickangle");
    trans.load_identity();
    trans.pre_rotate(-angle,left);
    trans.pre_translate(origin.asVector());

    obj = scinew GeomTransform(ticks, trans);

    show = get_GuiInt(base+"minticks");
    obj = scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&nsecondary),show>0);
    grid->add(scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&normal),show>0));


#if 0
    ticks = scinew GeomGroup();
    for (int i = 0; i <= int(round(divisions)); i++) {
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
    // Create Label
    trans.load_identity();
    trans.pre_rotate(radians*get_GuiDouble(base+"labelangle"),left);
    trans.project(secondary,textup);
    textup.normalize();
    textup *= scale*get_GuiDouble(base+"labelheight") / 100.00;
    
    GeomTextTexture *label;
    
    //    const string text = get_GuiString(base+"text"); 
    const string text(get_GuiString(base+"text"));//J^A^3^A^L   <<");

    label = scinew GeomTextTexture
      (label_font_, text, origin+secondary+textup/2.+primary/2+max_offset, 
       textup, primary);
    label->set_anchor(GeomTextTexture::c);
    
    show = get_GuiInt(base+"maxlabel");
    obj = scinew GeomSwitch(scinew GeomCull(label,(show<2?0:&secondary)),show>0);    
    grid->add(scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&normal),show>0));
    
    label = scinew GeomTextTexture
      (label_font_, text, origin-textup/2+primary/2+min_offset, textup, primary);
    label->set_anchor(GeomTextTexture::c);

    show = get_GuiInt(base+"minlabel");
    obj = scinew GeomSwitch(scinew GeomCull(label,show<2?0:&nsecondary),show>0);
    grid->add(scinew GeomSwitch(scinew GeomCull(obj,show<2?0:&normal),show>0));
#endif

    obj = grid;
    GeomTransform *mirror = scinew GeomTransform(grid);
    Vector center = origin.asVector()+bbox_.diagonal()/2.;
    
    mirror->translate(-center);
    mirror->rotate(M_PI,up);
    mirror->translate(center);
    
    show = get_GuiInt(base+"minplane");
    obj = scinew GeomSwitch(scinew GeomCull(grid, show<2?0:&normal),show>0);
    group->add(obj);

    show = get_GuiInt(base+"maxplane");
    obj = scinew GeomSwitch(scinew GeomCull(mirror, show<2?0:&nnormal),show>0);
    group->add(obj);
    
    //    grid->add(scinew GeomSwitch(scinew GeomCull(obj, show<2?0:&normal),show>0));

    //group->add(grid);
    //group->add(mirror);
  }

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

  axes_.push_back(Vector(bbox_.diagonal().x(),0,0));
  axes_.push_back(Vector(0,bbox_.diagonal().y(),0));
  axes_.push_back(Vector(0,0,bbox_.diagonal().z()));

  origins_.push_back(bbox_.min());

  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Axes");
  ogeom->delAll();

  //ostringstream str;
  //  int prim = i/3*3+i%3/2;
  //int sec = i/3*3+(i%3+1)/2+1;
  //int ter = i-(i+1)%3+1;
  
  //str << "Plane-" << prim << sec << "-" << ter << "-Axis-";
  //ogeom->addObj(generateAxisLines(prim,sec,ter),str.str());

  ogeom->addObj(generateAxisLines(0,1,2),string("A"));
  ogeom->addObj(generateAxisLines(1,2,0),string("B"));
  ogeom->addObj(generateAxisLines(0,2,1),string("C"));

  ogeom->flushViews();
}

} // End namespace SCIRun

