/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  ConvertMeshCoordinateSystem.cc: Take in fields and add all of their points into one field
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Modules/Fields/ConvertMeshCoordinateSystem.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Util/DynamicCompilation.h>
#include <iostream>

namespace SCIRun {

class ConvertMeshCoordinateSystem : public Module {
public:
  ConvertMeshCoordinateSystem(GuiContext* ctx);
  virtual ~ConvertMeshCoordinateSystem();
  virtual void execute();
  GuiString gui_oldsystem_;
  GuiString gui_newsystem_;
};

DECLARE_MAKER(ConvertMeshCoordinateSystem)
ConvertMeshCoordinateSystem::ConvertMeshCoordinateSystem(GuiContext* ctx)
  : Module("ConvertMeshCoordinateSystem", ctx, Filter, "ChangeMesh", "SCIRun"),
    gui_oldsystem_(get_ctx()->subVar("oldsystem"), "Cartesian"),
    gui_newsystem_(get_ctx()->subVar("newsystem"), "Spherical")
{
}

ConvertMeshCoordinateSystem::~ConvertMeshCoordinateSystem()
{
}

void
ConvertMeshCoordinateSystem::execute()
{
  // The input port (with data) is required.
  FieldHandle field;
  if (!get_input_handle("Input Field", field)) return;

  string oldsystem = gui_oldsystem_.get();
  string newsystem = gui_newsystem_.get();
  field.detach();
  field->mesh_detach();
  const TypeDescription *meshtd = field->mesh()->get_type_description();
  CompileInfoHandle ci = ConvertMeshCoordinateSystemAlgo::get_compile_info(meshtd);
  Handle<ConvertMeshCoordinateSystemAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;
  algo->execute(this, field->mesh(), oldsystem, newsystem);

  send_output_handle("Output Field", field);
}

CompileInfo *
ConvertMeshCoordinateSystemAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ConvertMeshCoordinateSystemAlgoT");
  static const string base_class_name("ConvertMeshCoordinateSystemAlgo");

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

} // End namespace SCIRun
