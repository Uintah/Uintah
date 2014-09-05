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

//    File   : ScalarMinMax.cc
//    Author : Kurt Zimmerman
//    Date   : March 2004

#include <sci_values.h>

#include <Packages/Uintah/Dataflow/Modules/Operators/ScalarMinMax.h>

#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/FieldInterface.h>

#include <Core/Containers/Handle.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/IntVector.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Containers/StringUtil.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>


namespace Uintah {

using std::endl;
using std::pair;
using std::ostringstream;

using namespace::SCIRun;




DECLARE_MAKER(ScalarMinMax)
ScalarMinMax::ScalarMinMax(GuiContext* ctx)
  : Module("ScalarMinMax", ctx, Sink, "Operators", "Uintah"),
    gui_min_data_(ctx->subVar("min_data", false)),
    gui_max_data_(ctx->subVar("max_data", false)),
    gui_min_index_(ctx->subVar("min_index", false)),
    gui_max_index_(ctx->subVar("max_index", false)),
    gui_min_values_(ctx->subVar("min_values", false)),
    gui_max_values_(ctx->subVar("max_values", false)),
    generation_(-1)
{
  gui_min_data_.set("---");
  gui_max_data_.set("---");
  gui_min_index_.set("---");
  gui_max_index_.set("---");
  gui_min_values_.set("---");
  gui_max_values_.set("---");
}


ScalarMinMax::~ScalarMinMax()
{
}



void
ScalarMinMax::clear_vals()
{
  gui_min_data_.set("---");
  gui_max_data_.set("---");
  gui_min_index_.set("---");
  gui_max_index_.set("---");
  gui_min_values_.set("---");
  gui_max_values_.set("---");
}


void
ScalarMinMax::update_input_attributes(FieldHandle f)
{

  ScalarFieldInterfaceHandle sdi = f->query_scalar_interface(this);
  if ( !sdi.get_rep() ){
    error("Not a Scalar Field.");
    clear_vals();
    return;
  }

  // Do this last, sometimes takes a while.
  
  if( !get_info( &my_reporter_, f )){
    clear_vals();
  }
}


void
ScalarMinMax::execute()
{
  FieldIPort *iport = (FieldIPort*)get_iport("Scalar Field");
  if (!iport)
  {
    error("Unable to initialize iport 'Scalar Field'.");
    return;
  }

  // The input port (with data) is required.
  FieldHandle fh;
  if (!iport->get(fh) || !fh.get_rep())
  {
    clear_vals();
    generation_ = -1;
    return;
  }

  if (generation_ != fh.get_rep()->generation)
  {
    generation_ = fh.get_rep()->generation;
    update_input_attributes(fh);

  }
}

CompileInfoHandle
ScalarMinMaxAlgoCount::get_compile_info(const TypeDescription *td)
{
  string subname;
  string subinc;
  string sname = td->get_name("", "");
  
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScalarMinMaxAlgoCountT");
  static const string base_class_name("ScalarMinMaxAlgoCount");

  if(sname.find("LatVol") != string::npos ){
    subname.append(td->get_name());
    subinc.append(include_path);
  } else {
    cerr<<"Unsupported Geometry, needs to be of Lattice type.\n";
    subname.append("Cannot compile this unupported type");
  }
  
  CompileInfo *rval =
    scinew CompileInfo(template_class_name + "." +
		       td->get_filename() + ".",
                       base_class_name,
                       template_class_name,
                       td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_include(subinc);
  rval->add_namespace("Uintah");
  td->fill_compile_info(rval);
  return rval;
}


} // end Uintah namespace


