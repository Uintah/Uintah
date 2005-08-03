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
 *  ManageFieldMesh: Store/retrieve values from an input matrix to/from 
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
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ManageFieldMesh.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ManageFieldMesh : public Module
{
public:
  ManageFieldMesh(GuiContext* ctx);
  virtual ~ManageFieldMesh();

  virtual void execute();
};


DECLARE_MAKER(ManageFieldMesh)
ManageFieldMesh::ManageFieldMesh(GuiContext* ctx)
  : Module("ManageFieldMesh", ctx, Filter, "FieldsGeometry", "SCIRun")
{
}



ManageFieldMesh::~ManageFieldMesh()
{
}



void
ManageFieldMesh::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!(ifp->get(ifieldhandle) && (ifieldhandle.get_rep())))
  {
    error( "No handle or representation." );
    return;
  }
  
  // Extract the output matrix.
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  CompileInfoHandle ci_extract =
    ManageFieldMeshAlgoExtract::get_compile_info(mtd);
  Handle<ManageFieldMeshAlgoExtract> algo_extract;
  if (!module_dynamic_compile(ci_extract, algo_extract))
  {
    return;
  }
  MatrixOPort *omp = (MatrixOPort *)get_oport("Output Matrix");
  omp->send(algo_extract->execute(ifieldhandle->mesh()));

  // Compute output field.
  FieldHandle result_field;
  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrixhandle;
  if (!(imatrix_port->get(imatrixhandle) && imatrixhandle.get_rep()))
  {
    remark("No input matrix connected, sending field as is.");
    result_field = ifieldhandle;
  }
  else
  {
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfoHandle ci_insert =
      ManageFieldMeshAlgoInsert::get_compile_info(ftd);
    Handle<ManageFieldMeshAlgoInsert> algo_insert;
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

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
  ofp->send(result_field);
}



CompileInfoHandle
ManageFieldMeshAlgoExtract::get_compile_info(const TypeDescription *msrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ManageFieldMeshAlgoExtractT");
  static const string base_class_name("ManageFieldMeshAlgoExtract");

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
ManageFieldMeshAlgoInsert::get_compile_info(const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ManageFieldMeshAlgoInsertT");
  static const string base_class_name("ManageFieldMeshAlgoInsert");

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

