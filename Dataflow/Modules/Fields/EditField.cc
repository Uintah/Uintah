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

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Modules/Fields/EditField.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <map>

namespace SCIRun {

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
  GuiDouble minx_;       // the out bounding box
  GuiDouble miny_;
  GuiDouble minz_;
  GuiDouble maxx_;
  GuiDouble maxy_;
  GuiDouble maxz_;

  GuiInt cfldname_;      // change name
  GuiInt ctypename_;     // change type
  GuiInt cdataat_;       // change data at
  GuiInt cdataminmax_;   // change data value extents
  GuiInt cbbox_;         // change bbox

  CrowdMonitor widget_lock_;
  ScaledBoxWidget *box_;

  bool firsttime_;
  int widgetid_;
  pair<double,double> minmax_;

  EditField(const string& id);

  virtual ~EditField();

  void clear_vals();
  void update_input_attributes(FieldHandle);
  bool check_types(FieldHandle);
  void build_widget(FieldHandle);

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
  virtual void widget_moved(int);
};

extern "C" PSECORESHARE Module* make_EditField(const string& id) {
  return scinew EditField(id);
}

EditField::EditField(const string& id)
  : Module("EditField", id, Source, "Fields", "SCIRun"),
    numnodes_("numnodes2", id, this),
    fldname_("fldname2", id, this),
    typename_("typename2", id, this),
    datamin_("datamin2", id, this),
    datamax_("datamax2", id, this),
    dataat_("dataat2", id, this),
    minx_("minx2", id, this),
    miny_("miny2", id, this),
    minz_("minz2", id, this),
    maxx_("maxx2", id, this),
    maxy_("maxy2", id, this),
    maxz_("maxz2", id, this),
    cfldname_("cfldname", id, this),
    ctypename_("ctypename", id, this),
    cdataat_("cdataat", id, this),
    cdataminmax_("cdataminmax", id, this),
    cbbox_("cbbox", id, this),
    widget_lock_("EditField widget lock"),
    minmax_(1,0)
  
{
  box_ = 0;
  firsttime_ = 1;
  widgetid_ = 0;
}

EditField::~EditField(){
}



void EditField::clear_vals() 
{
  TCL::execute(string("set ")+id+"-fldname \"---\"");
  TCL::execute(string("set ")+id+"-typename \"---\"");
  TCL::execute(string("set ")+id+"-datamin \"---\"");
  TCL::execute(string("set ")+id+"-datamax \"---\"");
  TCL::execute(string("set ")+id+"-numnodes \"---\"");
  TCL::execute(string("set ")+id+"-numelems \"---\"");
  TCL::execute(string("set ")+id+"-dataat \"---\"");
  TCL::execute(string("set ")+id+"-minx \"---\"");
  TCL::execute(string("set ")+id+"-miny \"---\"");
  TCL::execute(string("set ")+id+"-minz \"---\"");
  TCL::execute(string("set ")+id+"-maxx \"---\"");
  TCL::execute(string("set ")+id+"-maxy \"---\"");
  TCL::execute(string("set ")+id+"-maxz \"---\"");
  TCL::execute(id+" update_multifields");
}

void EditField::update_input_attributes(FieldHandle f) 
{
  const string &tname = f->get_type_description()->get_name();
  TCL::execute(string("set ")+id+"-typename \"" + tname + "\"");

  switch(f->data_at())
  {
  case Field::NODE:
    TCL::execute(string("set ")+id+"-dataat Nodes"); break;
  case Field::EDGE: 
    TCL::execute(string("set ")+id+"-dataat Edges"); break;
  case Field::FACE: 
    TCL::execute(string("set ")+id+"-dataat Faces"); break;
  case Field::CELL: 
    TCL::execute(string("set ")+id+"-dataat Cells"); break;
  default: ;
  }

  const BBox bbox = f->mesh()->get_bounding_box();
  Point min = bbox.min();
  Point max = bbox.max();

  TCL::execute(string("set ")+id+"-minx "+to_string(min.x()));
  TCL::execute(string("set ")+id+"-miny "+to_string(min.y()));
  TCL::execute(string("set ")+id+"-minz "+to_string(min.z()));
  TCL::execute(string("set ")+id+"-maxx "+to_string(max.x()));
  TCL::execute(string("set ")+id+"-maxy "+to_string(max.y()));
  TCL::execute(string("set ")+id+"-maxz "+to_string(max.z()));

  ScalarFieldInterface *sdi = f->query_scalar_interface();
  if (sdi && f->data_at() != Field::NONE) {
    sdi->compute_min_max(minmax_.first,minmax_.second);
    TCL::execute(string("set ")+id+"-datamin "+to_string(minmax_.first));
    TCL::execute(string("set ")+id+"-datamax "+to_string(minmax_.second));
  } else {
    TCL::execute(string("set ")+id+"-datamin \"--- N/A ---\"");
    TCL::execute(string("set ")+id+"-datamax \"--- N/A ---\"");
  }

  string fldname;
  if (f->get("name",fldname))
    TCL::execute(string("set ")+id+"-fldname "+fldname);
  else
    TCL::execute(string("set ")+id+"-fldname \"--- Name Not Assigned ---\"");


  TCL::execute(id+" update_multifields");

  // Do this last, sometimes takes a while.
  const TypeDescription *meshtd = f->mesh()->get_type_description();
  CompileInfo *ci = EditFieldAlgoCount::get_compile_info(meshtd);
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    cout << "Could not compile algorithm." << std::endl;
    return;
  }
  EditFieldAlgoCount *algo =
    dynamic_cast<EditFieldAlgoCount *>(algo_handle.get_rep());
  if (algo == 0)
  {
    cout << "Could not get algorithm." << std::endl;
    return;
  }
  int num_nodes;
  int num_elems;
  int dimension;
  algo->execute(f->mesh(), num_nodes, num_elems, dimension);

  TCL::execute(string("set ")+id+"-numnodes "+to_string(num_nodes));
  TCL::execute(string("set ")+id+"-numelems "+to_string(num_elems));

  TCL::execute(id+" update_multifields");
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



void EditField::build_widget(FieldHandle f)
{
  double l2norm;
  Point center, right, down, in;
  Point min,max;

  if (!cbbox_.get()) {
    // build a widget identical to the BBox
    const BBox bbox = f->mesh()->get_bounding_box();
    min = bbox.min();
    max = bbox.max();
  } else {
    // build a widget as described by the UI
    min = Point(minx_.get(),miny_.get(),minz_.get());
    max = Point(maxx_.get(),maxy_.get(),maxz_.get());
  }
  if ((max-min).length() < 1e-6)
  {
    min -= Vector(min) * 0.01;
    max += Vector(max) * 0.01;
  }
  
  center = Point(min.x()+(max.x()-min.x())/2.,
		 min.y()+(max.y()-min.y())/2.,
		 min.z()+(max.z()-min.z())/2.);
  right = center + Vector((max.x()-min.x())/2.,0,0);
  down = center + Vector(0,(max.y()-min.y())/2.,0);
  in = center + Vector(0,0,(max.z()-min.z())/2.);

  l2norm = (max-min).length();
  
  box_ = scinew ScaledBoxWidget(this,&widget_lock_,1);
  box_->SetScale(l2norm * 0.015);
  box_->SetPosition(center, right, down, in);
  box_->AxisAligned(1);
  
  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(box_->GetWidget());

  GeometryOPort *ogport=0;
  ogport = (GeometryOPort*)get_oport("Transformation Widget");
  if (!ogport) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  widgetid_ = ogport->addObj(widget_group,"EditField Transform widget",
			     &widget_lock_);
  ogport->flushViews();
}


void EditField::execute()
{
  FieldIPort *iport=(FieldIPort*)get_iport("Input Field"); 
  FieldHandle fh;
  //Field *f=0;
  FieldOPort *oport=(FieldOPort*)get_oport("Output Field");

  if (!iport) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  
  // the output port is required
  if (!oport) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  // the input port (with data) is required
  if (!iport->get(fh) || 
      !(fh.get_rep()))
  {
    clear_vals();
    return;
  }

  // get and display the attributes of the input field
  update_input_attributes(fh);

  // build the transform widget
  if (firsttime_)
  {
    firsttime_ = false;
    build_widget(fh);
  } 

  if (!cfldname_.get() &&
      !ctypename_.get() &&
      !cdataminmax_.get() &&
      !cdataat_.get() &&
      !cbbox_.get())
  {
    // no changes, just send the original through (it may be nothing!)
    oport->send(fh);
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
    cout << "Could not compile algorithm." << std::endl;
    return;
  }
  EditFieldAlgoCreate *algo =
    dynamic_cast<EditFieldAlgoCreate *>(algo_handle.get_rep());
  if (algo == 0)
  {
    cout << "Could not get algorithm." << std::endl;
    return;
  }
  TCL::execute(id + " set_state Executing 0");
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
      cout << "Could not compile algorithm." << std::endl;
      return;
    }
    EditFieldAlgoCopy *algo =
      dynamic_cast<EditFieldAlgoCopy *>(algo_handle.get_rep());
    if (algo == 0)
    {
      cout << "Could not get algorithm." << std::endl;
      return;
    }
    TCL::execute(id + " set_state Executing 0");
    algo->execute(fh, ef, scale, translate);
  }
  

  // Transform the mesh if necessary.
  if (cbbox_.get())
  {
    BBox old = fh->mesh()->get_bounding_box();
    Point oldc = old.min() + (old.max() - old.min()) / 2.0;
    Point center, right, down, in;
    box_->GetPosition(center, right, down, in);

    // Rotate * Scale * Translate.
    Transform t, r;
    Point unused;
    t.load_identity();
    r.load_frame(unused, (right-center).normal(),
		 (down-center).normal(),
		 (in-center).normal());
    t.pre_trans(r);
    t.pre_scale(Vector((right-center).length() / (old.max().x() - oldc.x()),
		       (down-center).length() / (old.max().y() - oldc.y()),
		       (in-center).length() / (old.max().z() - oldc.z())));
    t.pre_translate(Vector(center.x(), center.y(), center.z()));

    ef->mesh_detach();
    ef->mesh()->transform(t);
  }

  // Set some field attributes.
  if (cfldname_.get())
  {
    ef->store(string("name"), fldname_.get());
  }

  ScalarFieldInterface* sfi = ef->query_scalar_interface();
  if (sfi)
  {
    std::pair<double, double> minmax(1, 0);
    sfi->compute_min_max(minmax.first, minmax.second);
    ef->store(string("minmax"), minmax);
  }
    
  oport->send(ef);
}

    
void EditField::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("EditField needs a minor command");
    return;
  }
 
  if (args[1] == "execute") {
    want_to_execute();
  } else if (args[1] == "update_widget") {
    Point center, right, down, in;
    minx_.reset(); miny_.reset(); minz_.reset();
    maxx_.reset(); maxy_.reset(); maxz_.reset();
    Point min(minx_.get(),miny_.get(),minz_.get());
    Point max(maxx_.get(),maxy_.get(),maxz_.get());
    if (max.x()<=min.x() ||
	max.y()<=min.y() ||
	max.z()<=min.z()) {
      widget_moved(1);           // force values back to widget settings
      postMessage("EditField: Degenerate BBox requested!");
      return;                    // degenerate 
    }
    center = min+((max-min)/2.);
    right = Point(max.x(),center.y(),center.z());
    down = Point(center.x(),max.y(),center.z());
    in = Point(center.x(),center.y(),max.z());
    box_->SetPosition(center,right,down,in);
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}

void EditField::widget_moved(int i)
{
  if (i==1) {
    Point center, right, down, in;
    minx_.reset(); miny_.reset(); minz_.reset();
    maxx_.reset(); maxy_.reset(); maxz_.reset();
    box_->GetPosition(center,right,down,in);
    minx_.set((center-(right-center)).x());
    miny_.set((center-(down-center)).y());
    minz_.set((center-(in-center)).z());
    maxx_.set(right.x());
    maxy_.set(down.y());
    maxz_.set(in.z());
    want_to_execute();
  } else {
    //Module::widget_moved(i);
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


