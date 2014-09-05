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
 *  EditField.cc:
 *
 *  Written by:
 *   moulding
 *   April 22, 2001
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/share/share.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Modules/Fields/EditField.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {

using std::endl;
using std::pair;

class PSECORESHARE EditField : public Module {
public:
  // out is the output field
  // in is the input field

  GuiString numnodes_;   // the in number of nodes 

  GuiString fldname_;    // the out property "name"
  GuiString typename_;   // the out field type
  GuiDouble datamin_;    // the out data min
  GuiDouble datamax_;    // the out data max
  GuiString dataat_;     // the out data at
  GuiDouble cx_;         // the out geometry (center point and size)
  GuiDouble cy_;
  GuiDouble cz_;
  GuiDouble sizex_;
  GuiDouble sizey_;
  GuiDouble sizez_;

  GuiInt cfldname_;      // change name
  GuiInt ctypename_;     // change type
  GuiInt cdataat_;       // change data at
  GuiInt cdataminmax_;   // change data value extents
  GuiInt cgeom_;         // change geometry

  CrowdMonitor     widget_lock_;
  BoxWidget *box_;
  Transform        box_initial_transform_;
  BBox             box_initial_bounds_;
  int              generation_;

  int widgetid_;
  pair<double,double> minmax_;

  EditField(GuiContext* ctx);

  virtual ~EditField();

  void clear_vals();
  void update_input_attributes(FieldHandle);
  bool check_types(FieldHandle);
  void build_widget(FieldHandle);

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  virtual void widget_moved(bool);
};

  DECLARE_MAKER(EditField)

EditField::EditField(GuiContext* ctx)
  : Module("EditField", ctx, Source, "Fields", "SCIRun"),
    numnodes_(ctx->subVar("numnodes2")),
    fldname_(ctx->subVar("fldname2")),
    typename_(ctx->subVar("typename2")),
    datamin_(ctx->subVar("datamin2")),
    datamax_(ctx->subVar("datamax2")),
    dataat_(ctx->subVar("dataat2")),
    cx_(ctx->subVar("cx2")),
    cy_(ctx->subVar("cy2")),
    cz_(ctx->subVar("cz2")),
    sizex_(ctx->subVar("sizex2")),
    sizey_(ctx->subVar("sizey2")),
    sizez_(ctx->subVar("sizez2")),
    cfldname_(ctx->subVar("cfldname")),
    ctypename_(ctx->subVar("ctypename")),
    cdataat_(ctx->subVar("cdataat")),
    cdataminmax_(ctx->subVar("cdataminmax")),
    cgeom_(ctx->subVar("cgeom")),
    widget_lock_("EditField widget lock"),
    generation_(-1),
    minmax_(1,0)
  
{
  box_ = scinew BoxWidget(this, &widget_lock_, 1.0, false, false);
  widgetid_ = 0;
}

EditField::~EditField(){
}



void EditField::clear_vals() 
{
  gui->execute(string("set ")+id+"-fldname \"---\"");
  gui->execute(string("set ")+id+"-typename \"---\"");
  gui->execute(string("set ")+id+"-datamin \"---\"");
  gui->execute(string("set ")+id+"-datamax \"---\"");
  gui->execute(string("set ")+id+"-numnodes \"---\"");
  gui->execute(string("set ")+id+"-numelems \"---\"");
  gui->execute(string("set ")+id+"-dataat \"---\"");
  gui->execute(string("set ")+id+"-cx \"---\"");
  gui->execute(string("set ")+id+"-cy \"---\"");
  gui->execute(string("set ")+id+"-cz \"---\"");
  gui->execute(string("set ")+id+"-sizex \"---\"");
  gui->execute(string("set ")+id+"-sizey \"---\"");
  gui->execute(string("set ")+id+"-sizez \"---\"");
  gui->execute(id+" update_multifields");
}

void EditField::update_input_attributes(FieldHandle f) 
{
  const string &tname = f->get_type_description()->get_name();
  gui->execute(string("set ")+id+"-typename \"" + tname + "\"");

  switch(f->data_at())
  {
  case Field::NODE:
    gui->execute(string("set ")+id+"-dataat Nodes"); break;
  case Field::EDGE: 
    gui->execute(string("set ")+id+"-dataat Edges"); break;
  case Field::FACE: 
    gui->execute(string("set ")+id+"-dataat Faces"); break;
  case Field::CELL: 
    gui->execute(string("set ")+id+"-dataat Cells"); break;
  default: ;
  }

  Point center,right,down,in;
  Vector size;
// if (box_) {
 if (0) {
    // use geom info from widget
    box_->GetPosition(center, right, down, in);
    size = Vector((right-center).x()*2.,
                  (down-center).y()*2.,
                  (in-center).z()*2.);
  } else {
    // use geom info from bbox
    const BBox bbox = f->mesh()->get_bounding_box();
    size = bbox.diagonal();
    center = bbox.center();
  }
  gui->execute(string("set ")+id+"-cx "+to_string(center.x()));
  gui->execute(string("set ")+id+"-cy "+to_string(center.y()));
  gui->execute(string("set ")+id+"-cz "+to_string(center.z()));
  gui->execute(string("set ")+id+"-sizex "+to_string(size.x()));
  gui->execute(string("set ")+id+"-sizey "+to_string(size.y()));
  gui->execute(string("set ")+id+"-sizez "+to_string(size.z()));

  ScalarFieldInterface *sdi = f->query_scalar_interface();
  if (sdi && f->data_at() != Field::NONE) {
    sdi->compute_min_max(minmax_.first,minmax_.second);
    gui->execute(string("set ")+id+"-datamin "+to_string(minmax_.first));
    gui->execute(string("set ")+id+"-datamax "+to_string(minmax_.second));
  } else {
    gui->execute(string("set ")+id+"-datamin \"--- N/A ---\"");
    gui->execute(string("set ")+id+"-datamax \"--- N/A ---\"");
  }

  string fldname;
  if (f->get_property("name",fldname))
    gui->execute(string("set ")+id+"-fldname "+fldname);
  else
    gui->execute(string("set ")+id+"-fldname \"--- Name Not Assigned ---\"");


  gui->execute(id+" update_multifields");

  // Do this last, sometimes takes a while.
  const TypeDescription *meshtd = f->mesh()->get_type_description();
  CompileInfo *ci = EditFieldAlgoCount::get_compile_info(meshtd);
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    error("Could not compile algorithm.");
    return;
  }
  EditFieldAlgoCount *algo =
    dynamic_cast<EditFieldAlgoCount *>(algo_handle.get_rep());
  if (algo == 0)
  {
    error("Could not get algorithm.");
    return;
  }
  int num_nodes;
  int num_elems;
  int dimension;
  algo->execute(f->mesh(), num_nodes, num_elems, dimension);

  gui->execute(string("set ")+id+"-numnodes "+to_string(num_nodes));
  gui->execute(string("set ")+id+"-numelems "+to_string(num_elems));

  gui->execute(id+" update_multifields");

  // copy valid settings to the un-checked output field attributes
  gui->execute(id+" copy_attributes; update idletasks");
}

bool EditField::check_types(FieldHandle f)
{
  const string &oname = f->get_type_description()->get_name();
  const string &nname = typename_.get();
  
  string::size_type oindx = oname.find('<');
  string::size_type nindx = nname.find('<');

  if (oindx == nindx)
  {
    if (oname.substr(0, oindx) == nname.substr(0, nindx))
    {
      return true;
    }
  }
  warning("The selected type and the input field type are incompatable.");
  return false;
}



void
EditField::build_widget(FieldHandle f)
{
  double l2norm;
  Point center;
  Vector size;
  BBox bbox = f->mesh()->get_bounding_box();
  box_initial_bounds_ = bbox;

  if (!cgeom_.get()) {
    // build a widget identical to the BBox
    size = Vector(bbox.max()-bbox.min());
    if (fabs(size.x())<1.e-4) {
      size.x(2.e-4); 
      bbox.extend(bbox.min()-Vector(1.e-4,0,0));
    }
    if (fabs(size.y())<1.e-4) {
      size.y(2.e-4); 
      bbox.extend(bbox.min()-Vector(0,1.e-4,0));
    }
    if (fabs(size.z())<1.e-4) {
      size.z(2.e-4); 
      bbox.extend(bbox.min()-Vector(0,0,1.e-4));
    }
    center = Point(bbox.min() + size/2.);
  } else {
    // build a widget as described by the UI
    // watch for degenerate sizes!
    cx_.reset();
    cy_.reset();
    cz_.reset();
    sizex_.reset();
    sizey_.reset();
    sizez_.reset();
    if (sizex_.get()<=0) sizex_.set(1);
    if (sizey_.get()<=0) sizey_.set(1);
    if (sizez_.get()<=0) sizez_.set(1);    
    center = Point(cx_.get(),cy_.get(),cz_.get());
    size = Vector(sizex_.get(),sizey_.get(),sizez_.get());
  }

  Vector sizex(size.x(),0,0);
  Vector sizey(0,size.y(),0);
  Vector sizez(0,0,size.z());

  Point right(center + sizex/2.);
  Point down(center + sizey/2.);
  Point in(center +sizez/2.);

  l2norm = size.length();

  // Translate * Rotate * Scale.
  Transform r;
  Point unused;
  box_initial_transform_.load_identity();
  box_initial_transform_.pre_scale(Vector((right-center).length(),
			     (down-center).length(),
			     (in-center).length()));
  r.load_frame(unused, (right-center).normal(),
	       (down-center).normal(),
	       (in-center).normal());
  box_initial_transform_.pre_trans(r);
  box_initial_transform_.pre_translate(center.asVector());

  box_->SetScale(l2norm * 0.015);
  box_->SetPosition(center, right, down, in);
  
  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(box_->GetWidget());

  GeometryOPort *ogport = (GeometryOPort*)get_oport("Transformation Widget");
  if (!ogport) {
    error("Unable to initialize oport 'Transformation Widget'.");
    return;
  }
  widgetid_ = ogport->addObj(widget_group,"EditField Transform widget",
			     &widget_lock_);
  ogport->flushViews();
}

void
EditField::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Input Field"); 
  if (!iport) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  
  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || 
      !(fh.get_rep()))
  {
    clear_vals();
    return;
  }

  // The output port is required.
  FieldOPort *oport = (FieldOPort*)get_oport("Output Field");
  if (!oport) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }


  // build the transform widget
  if (generation_ != fh.get_rep()->generation) {
    generation_ = fh.get_rep()->generation;
    // get and display the attributes of the input field
    update_input_attributes(fh);
    build_widget(fh);
  }

  if (!cfldname_.get() &&
      !ctypename_.get() &&
      !cdataminmax_.get() &&
      !cdataat_.get() &&
      !cgeom_.get())
  {
    // no changes, just send the original through (it may be nothing!)
    oport->send(fh);
    remark("Passing field from input port to output port unchanged.");
    return;
  }

  // verify that the requested edits are possible (type check)
  if (!check_types(fh))
  {
    typename_.set(fh->get_type_description()->get_name());
  }

  // Identify the new data location.
  Field::data_location dataat = fh->data_at();
#if 0  // TODO:: Fix this.
  if (cdataat_.get())
  {
    const string &d = dataat_.get();
    if (d == "Nodes")
    {
      dataat = Field::NODE;
    }
    else if (d == "Edges")
    {
      dataat = Field::EDGE;
    }
    else if (d == "Faces")
    {
      dataat = Field::FACE;
    }
    else if (d == "Cells")
    {
      dataat = Field::CELL;
    }
  }
#endif

  // Setup data transform.
  bool transform_p = cdataminmax_.get();
  double scale = 1.0;
  double translate = 0.0;
  if (transform_p)
  {
    if (fh->query_scalar_interface())
    {
      scale = (datamax_.get() - datamin_.get()) /
	(minmax_.second - minmax_.first);
      translate = datamin_.get() - minmax_.first * scale;
    }
    else
    {
      transform_p = false;
    }
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrc_td = fh->get_type_description();
  CompileInfo *ci =
    EditFieldAlgoCreate::get_compile_info(fsrc_td, typename_.get());
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    error("Could not compile algorithm.");
    return;
  }
  EditFieldAlgoCreate *algo =
    dynamic_cast<EditFieldAlgoCreate *>(algo_handle.get_rep());
  if (algo == 0)
  {
    error("Could not get algorithm.");
    return;
  }
  gui->execute(id + " set_state Executing 0");
  bool same_value_type_p = false;
  FieldHandle ef(algo->execute(fh, dataat, same_value_type_p));

  // Do any necessary data transforms here.
  const bool both_scalar_p =
    ef->query_scalar_interface() && fh->query_scalar_interface();
  if (both_scalar_p || same_value_type_p)
  {
    const TypeDescription *fdst_td = ef->get_type_description();
    CompileInfo *ci =
      EditFieldAlgoCopy::get_compile_info(fsrc_td, fdst_td,
					  both_scalar_p && transform_p);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }
    EditFieldAlgoCopy *algo =
      dynamic_cast<EditFieldAlgoCopy *>(algo_handle.get_rep());
    if (algo == 0)
    {
      error("Could not get algorithm.");
      return;
    }
    gui->execute(id + " set_state Executing 0");
    algo->execute(fh, ef, scale, translate);
  }
  

  // Transform the mesh if necessary.
  // Translate * Rotate * Scale.
  Point center, right, down, in;
  box_->GetPosition(center, right, down, in);
  Transform t, r;
  Point unused;
  t.load_identity();
  t.pre_scale(Vector((right-center).length(),
       (down-center).length(),
       (in-center).length()));
  r.load_frame(unused, (right-center).normal(),
	 (down-center).normal(),
	 (in-center).normal());
  t.pre_trans(r);
  t.pre_translate(center.asVector());

  Transform inv(box_initial_transform_);
  inv.invert();
  t.post_trans(inv);
  if (cgeom_.get())
  {
    ef->mesh_detach();
    ef->mesh()->transform(t);
  }

  // Set some field attributes.
  if (cfldname_.get())
  {
    ef->set_property(string("name"), fldname_.get(), false);
  }

  ScalarFieldInterface* sfi = ef->query_scalar_interface();
  if (sfi)
  {
    std::pair<double, double> minmax(1, 0);
    sfi->compute_min_max(minmax.first, minmax.second);
    ef->set_property(string("minmax"), minmax, true);
  }
    
  oport->send(ef);

  // The output port is required.
  MatrixOPort *moport = (MatrixOPort*)get_oport("Transformation Matrix");
  if (!moport) {
    error("Unable to initialize oport 'Transformation Matrix'.");
    return;
  }  

  // convert the transform into a matrix and send it out   
  DenseMatrix *matrix_transform = scinew DenseMatrix(t);
  MatrixHandle mh = matrix_transform;
  moport->send(mh);
}

    
void EditField::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("EditField needs a minor command");
    return;
  }
 
  if (args[1] == "execute" || args[1] == "update_widget") {
    Point center, right, down, in;
    cx_.reset(); cy_.reset(); cz_.reset();
    sizex_.reset(); sizey_.reset(); sizez_.reset();
    if (sizex_.get() <= 0 || sizey_.get() <= 0 || sizez_.get() <= 0) {
      error("Degenerate BBox requested.");
      widget_moved(true);           // force values back to widget settings
      return;                    // degenerate 
    }
    Vector sizex(sizex_.get(),0,0);
    Vector sizey(0,sizey_.get(),0);
    Vector sizez(0,0,sizez_.get());
    center = Point(cx_.get(),cy_.get(),cz_.get());
    right = Point(center + sizex/2.);
    down = Point(center + sizey/2.);
    in = Point(center + sizez/2.);
    box_->SetPosition(center,right,down,in);
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}

void EditField::widget_moved(bool last)
{
  if (last) {
    Point center, right, down, in;
    cx_.reset(); cy_.reset(); cz_.reset();
    sizex_.reset(); sizey_.reset(); sizez_.reset();
    box_->GetPosition(center,right,down,in);
    cx_.set(center.x());
    cy_.set(center.y());
    cz_.set(center.z());
    sizex_.set((right.x()-center.x())*2.);
    sizey_.set((down.y()-center.y())*2.);
    sizez_.set((in.z()-center.z())*2.);
    cgeom_.set(1);
    want_to_execute();
  }
}


CompileInfo *
EditFieldAlgoCount::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("EditFieldAlgoCountT");
  static const string base_class_name("EditFieldAlgoCount");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}


CompileInfo *
EditFieldAlgoCreate::get_compile_info(const TypeDescription *field_td,
				    const string &fdstname)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class("EditFieldAlgoCreateT");
  static const string base_class_name("EditFieldAlgoCreate");

  CompileInfo *rval = 
    scinew CompileInfo(template_class + "." +
		       field_td->get_filename() + "." +
		       to_filename(fdstname) + ".",
		       base_class_name, 
		       template_class,
                       field_td->get_name() + "," + fdstname + " ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}


CompileInfo *
EditFieldAlgoCopy::get_compile_info(const TypeDescription *fsrctd,
				    const TypeDescription *fdsttd,
				    bool transform_p)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class0("EditFieldAlgoCopyT");
  static const string template_class1("EditFieldAlgoCopyTT");
  static const string base_class_name("EditFieldAlgoCopy");

  CompileInfo *rval = 
    scinew CompileInfo((transform_p?template_class1:template_class0) + "." +
		       fsrctd->get_filename() + "." +
		       fdsttd->get_filename() + ".",
                       base_class_name, 
		       (transform_p?template_class1:template_class0),
                       fsrctd->get_name() + "," + fdsttd->get_name() + " ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrctd->fill_compile_info(rval);
  return rval;
}


} // End namespace Moulding


