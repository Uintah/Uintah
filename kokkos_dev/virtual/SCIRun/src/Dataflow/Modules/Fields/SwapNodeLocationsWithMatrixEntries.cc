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
 *  SwapNodeLocationsWithMatrixEntries: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/SwapNodeLocationsWithMatrixEntries.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class SwapNodeLocationsWithMatrixEntries : public Module
{
public:
  SwapNodeLocationsWithMatrixEntries(GuiContext* ctx);
  virtual ~SwapNodeLocationsWithMatrixEntries();

  virtual void execute();
};


DECLARE_MAKER(SwapNodeLocationsWithMatrixEntries)
SwapNodeLocationsWithMatrixEntries::SwapNodeLocationsWithMatrixEntries(GuiContext* ctx)
  : Module("SwapNodeLocationsWithMatrixEntries", ctx, Filter, "ChangeMesh", "SCIRun")
{
}



SwapNodeLocationsWithMatrixEntries::~SwapNodeLocationsWithMatrixEntries()
{
}



void
SwapNodeLocationsWithMatrixEntries::execute()
{
  // Get input field.
  FieldHandle ifieldhandle;
  if (!get_input_handle("Input Field", ifieldhandle)) return;
  
  // Extract the output matrix.
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  CompileInfoHandle ci_extract =
    SwapNodeLocationsWithMatrixEntriesAlgoExtract::get_compile_info(mtd);
  Handle<SwapNodeLocationsWithMatrixEntriesAlgoExtract> algo_extract;
  if (!module_dynamic_compile(ci_extract, algo_extract))
  {
    return;
  }
  MatrixHandle mtmp(algo_extract->execute(ifieldhandle->mesh()));
  send_output_handle("Output Matrix", mtmp);

  // Compute output field.
  FieldHandle result_field;
  MatrixHandle imatrixhandle;
  if (!get_input_handle("Input Matrix", imatrixhandle, false))
  {
    remark("No input matrix connected, sending field as is.");
    result_field = ifieldhandle;
  }
  else
  {
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfoHandle ci_insert =
      SwapNodeLocationsWithMatrixEntriesAlgoInsert::get_compile_info(ftd);
    Handle<SwapNodeLocationsWithMatrixEntriesAlgoInsert> algo_insert;
    if (!DynamicCompilation::compile(ci_insert, algo_insert, true, this))
    {
      error("Could not compile insertion algorithm.");
      error("Input field probably not of editable type.");
      return;
    }

    result_field = algo_insert->execute(this, ifieldhandle, imatrixhandle);

    if (!result_field.get_rep())
    {
      return;
    }

    // Copy the properties.
    result_field->copy_properties(ifieldhandle.get_rep());
  }

  send_output_handle("Output Field", result_field);
}



CompileInfoHandle
SwapNodeLocationsWithMatrixEntriesAlgoExtract::get_compile_info(const TypeDescription *msrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SwapNodeLocationsWithMatrixEntriesAlgoExtractT");
  static const string base_class_name("SwapNodeLocationsWithMatrixEntriesAlgoExtract");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       msrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
SwapNodeLocationsWithMatrixEntriesAlgoInsert::get_compile_info(const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SwapNodeLocationsWithMatrixEntriesAlgoInsertT");
  static const string base_class_name("SwapNodeLocationsWithMatrixEntriesAlgoInsert");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}



} // End namespace SCIRun

