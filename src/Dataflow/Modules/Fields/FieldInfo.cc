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

//    File   : FieldInfo.cc
//    Author : McKay Davis
//    Date   : July 2002

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/share/share.h>

#include <Core/Containers/Handle.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/FieldInfo.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Containers/StringUtil.h>
#include <map>
#include <iostream>

namespace SCIRun {

using std::endl;
using std::pair;

class PSECORESHARE FieldInfo : public Module {
public:
  int              generation_;
  pair<double,double> minmax_;
  FieldInfo(GuiContext* ctx);
  virtual ~FieldInfo();
  void clear_vals();
  void update_input_attributes(FieldHandle);
  virtual void execute();
};

  DECLARE_MAKER(FieldInfo)

FieldInfo::FieldInfo(GuiContext* ctx)
  : Module("FieldInfo", ctx, Sink, "Fields", "SCIRun"),
    generation_(-1),
    minmax_(1,0)
  
{
}

FieldInfo::~FieldInfo(){
}



void
FieldInfo::clear_vals() 
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


void
FieldInfo::update_input_attributes(FieldHandle f) 
{
  const string &tname = f->get_type_description()->get_name();
  gui->execute(string("set ")+id+"-typename \"" + tname + "\"");

  switch(f->data_at())
  {
  case Field::NODE:
    gui->execute(string("set ")+id+"-dataat Nodes");
    break;
  case Field::EDGE: 
    gui->execute(string("set ")+id+"-dataat Edges");
    break;
  case Field::FACE: 
    gui->execute(string("set ")+id+"-dataat Faces");
    break;
  case Field::CELL: 
    gui->execute(string("set ")+id+"-dataat Cells");
    break;
  case Field::NONE: 
    gui->execute(string("set ")+id+"-dataat None");
    break;
  }

  Point center;
  Vector size;

  
  const BBox bbox = f->mesh()->get_bounding_box();
  if (bbox.valid()) {
    size = bbox.diagonal();
    center = bbox.center();
    gui->execute(string("set ")+id+"-cx "+to_string(center.x()));
    gui->execute(string("set ")+id+"-cy "+to_string(center.y()));
    gui->execute(string("set ")+id+"-cz "+to_string(center.z()));
    gui->execute(string("set ")+id+"-sizex "+to_string(size.x()));
    gui->execute(string("set ")+id+"-sizey "+to_string(size.y()));
    gui->execute(string("set ")+id+"-sizez "+to_string(size.z()));
  } else {
    warning("Input ield is empty.");
    gui->execute(string("set ")+id+"-cx \"--- N/A ---\"");
    gui->execute(string("set ")+id+"-cy \"--- N/A ---\"");
    gui->execute(string("set ")+id+"-cz \"--- N/A ---\"");
    gui->execute(string("set ")+id+"-sizex \"--- N/A ---\"");
    gui->execute(string("set ")+id+"-sizey \"--- N/A ---\"");
    gui->execute(string("set ")+id+"-sizez \"--- N/A ---\"");
  }

  ScalarFieldInterfaceHandle sdi = f->query_scalar_interface(this);
  if (sdi.get_rep() && f->data_at() != Field::NONE) {
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
  CompileInfoHandle ci = FieldInfoAlgoCount::get_compile_info(meshtd);
  Handle<FieldInfoAlgoCount> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  int num_nodes;
  int num_elems;
  algo->execute(f->mesh(), num_nodes, num_elems);

  gui->execute(string("set ")+id+"-numnodes "+to_string(num_nodes));
  gui->execute(string("set ")+id+"-numelems "+to_string(num_elems));

  gui->execute(id+" update_multifields");
}


void
FieldInfo::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Input Field"); 
  if (!iport) 
  {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  
  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || !fh.get_rep())
  {
    clear_vals();
    return;
  }

  if (generation_ != fh.get_rep()->generation) 
  {
    generation_ = fh.get_rep()->generation;
    update_input_attributes(fh);

  }
}

CompileInfoHandle
FieldInfoAlgoCount::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldInfoAlgoCountT");
  static const string base_class_name("FieldInfoAlgoCount");

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


} // end SCIRun namespace


