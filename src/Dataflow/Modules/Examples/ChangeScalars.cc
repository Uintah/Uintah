//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ChangeScalars.cc
//    Author : Martin Cole
//    Date   : Mon Sep 11 11:22:14 2006


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/SimplePort.h>

#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Modules/Examples/ChangeScalars.h>
#include <Dataflow/GuiInterface/GuiVar.h>

#include <iostream>

namespace SCIRun {
using namespace std;
using namespace SCIRun;

class ChangeScalars : public Module 
{
public:
  ChangeScalars(GuiContext*);
  virtual ~ChangeScalars();

  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
private:
  GuiDouble       newval_;
};


DECLARE_MAKER(ChangeScalars)
ChangeScalars::ChangeScalars(GuiContext* ctx) : 
  Module("ChangeScalars", ctx, Source, "Examples", "SCIRun"),
  newval_(get_ctx()->subVar("newval"), 1.0)
{
}

ChangeScalars::~ChangeScalars()
{
}

void
ChangeScalars::execute()
{

  FieldHandle field_handle;
  if (! get_input_handle("InField", field_handle, true)) {
    error("ChangeScalars must have a SCIRun::Field as input to continue.");
    return;
  }

 

  // Check for scalar field type.

  if (! field_handle->query_scalar_interface().get_rep()) {
    error("This module can only handle scalar data.");
    return;
  }

  // Check for at least linear basis.
  if (field_handle->basis_order() < 1) {
    error("This module must have a linear or better basis.");
    return;
  }  
  
  newval_.reset();

  const TypeDescription *ftd = field_handle->get_type_description();
  CompileInfoHandle ci = ChangeScalarsAlgoBase::get_compile_info(ftd);
  Handle<ChangeScalarsAlgoBase> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  FieldHandle out_field_handle(algo->execute(this, field_handle, newval_.get()));

  // Create the algorithm for module and call it.

  send_output_handle("OutField", out_field_handle);
}


CompileInfoHandle
ChangeScalarsAlgoBase::get_compile_info(const TypeDescription *td) 
{

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ChangeScalarsT");
  static const string base_class_name("ChangeScalarsAlgoBase");


  CompileInfo *rval = scinew CompileInfo(template_class_name + "." +
					 td->get_filename() + ".",
					 base_class_name, 
					 template_class_name, 
					 td->get_name());
  rval->add_include(include_path);
  td->fill_compile_info(rval);
  return rval;
}



void
ChangeScalars::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


