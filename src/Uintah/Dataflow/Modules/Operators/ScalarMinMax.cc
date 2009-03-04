/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/IntVector.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using std::endl;
using std::pair;
using std::ostringstream;

using namespace Uintah;
using namespace SCIRun;

DECLARE_MAKER(ScalarMinMax)
ScalarMinMax::ScalarMinMax(GuiContext* ctx)
  : Module("ScalarMinMax", ctx, Sink, "Operators", "Uintah"),
    gui_field_name_(get_ctx()->subVar("field_name", false)),
    gui_min_data_(get_ctx()->subVar("min_data", false)),
    gui_max_data_(get_ctx()->subVar("max_data", false)),
    gui_min_index_(get_ctx()->subVar("min_index", false)),
    gui_max_index_(get_ctx()->subVar("max_index", false)),
    gui_min_values_(get_ctx()->subVar("min_values", false)),
    gui_max_values_(get_ctx()->subVar("max_values", false)),
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
  gui_field_name_.set("---");
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

