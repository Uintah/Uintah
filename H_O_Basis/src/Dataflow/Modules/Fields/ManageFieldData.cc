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
 *  ManageFieldData: Store/retrieve values from an input matrix to/from 
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
#include <Dataflow/Modules/Fields/ManageFieldData.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ManageFieldData : public Module
{
  GuiInt gui_preserve_scalar_type_;
public:
  ManageFieldData(GuiContext* ctx);
  virtual ~ManageFieldData();

  virtual void execute();
};


DECLARE_MAKER(ManageFieldData)

ManageFieldData::ManageFieldData(GuiContext* ctx)
  : Module("ManageFieldData", ctx, Filter, "FieldsData", "SCIRun"),
    gui_preserve_scalar_type_(ctx->subVar("preserve-scalar-type"))
{
}



ManageFieldData::~ManageFieldData()
{
}



void
ManageFieldData::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!(ifp->get(ifieldhandle) && (ifieldhandle.get_rep())))
  {
    error( "No field available in the 'Input Field' port.");
    return;
  }

  // TODO: Using datasize this way appears to be wrong, as it depends
  // on the input DATA_AT size and not the one picked for output.
  int datasize = 0;
  int svt_flag = 0;
  if (ifieldhandle->query_scalar_interface(this).get_rep())
  {
    svt_flag = 0;
  }
  else if (ifieldhandle->query_vector_interface(this).get_rep())
  {
    svt_flag = 1;
  }
  else if (ifieldhandle->query_tensor_interface(this).get_rep())
  {
    svt_flag = 2;
  }

  // Compute output matrix.
  if (ifieldhandle->basis_order() == -1)
  {
    remark("Input field contains no data, no output matrix created.");
  }
  else
  {
    CompileInfoHandle ci_field =
      ManageFieldDataAlgoField::
      get_compile_info(ifieldhandle->get_type_description(), svt_flag);
    Handle<ManageFieldDataAlgoField> algo_field;
    if (!DynamicCompilation::compile(ci_field, algo_field, true, this))
    {
      warning("Unable to extract data from input field, no output matrix created.");
    }
    else
    {
      MatrixOPort *omp = (MatrixOPort *)get_oport("Output Matrix");
      omp->send(algo_field->execute(ifieldhandle, datasize));
    }
  }

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
    int matrix_svt_flag = svt_flag;
    if (imatrixhandle->nrows() == 6 || imatrixhandle->ncols() == 6)
    {
      matrix_svt_flag = 3;
    }
    else if (imatrixhandle->nrows() == 9 || imatrixhandle->ncols() == 9)
    {
      matrix_svt_flag = 2;
    }
    else if (imatrixhandle->nrows() == 3 || imatrixhandle->ncols() == 3)
    {
      matrix_svt_flag = 1;
    }
    else if (imatrixhandle->nrows() == 1 || imatrixhandle->ncols() == 1)
    {
      matrix_svt_flag = 0;
    }
    else
    {
      error("Input matrix row/column size mismatch.");
      error("Input matrix does not appear to fit in the field.");
      return;
    }
    if (matrix_svt_flag == 3 && datasize == 6)
    {
      if (imatrixhandle->nrows() == 3 || imatrixhandle->ncols() == 3)
      {
	matrix_svt_flag = 1;
      }
      else if (imatrixhandle->nrows() == 1 || imatrixhandle->ncols() == 1)
      {
	matrix_svt_flag = 0;
      }
    }
    if (matrix_svt_flag == 2 && datasize == 9)
    {
      if (imatrixhandle->nrows() == 3 || imatrixhandle->ncols() == 3)
      {
	matrix_svt_flag = 1;
      }
      else if (imatrixhandle->nrows() == 1 || imatrixhandle->ncols() == 1)
      {
	matrix_svt_flag = 0;
      }
    }
    if (matrix_svt_flag == 1 && datasize == 3)
    {
      if (imatrixhandle->nrows() == 1 || imatrixhandle->ncols() == 1)
      {
	matrix_svt_flag = 0;
      }
    }
    if ((imatrixhandle->nrows() == 9 || imatrixhandle->nrows() == 6) &&
	(imatrixhandle->ncols() == 9 || imatrixhandle->ncols() == 6))
    {
      remark("Input matrix is " + to_string(imatrixhandle->nrows()) + "x" +
	     to_string(imatrixhandle->ncols()) +
	     ".  Using rows or columns as tensors is ambiguous.");
    }
    else if (imatrixhandle->nrows() == 3 && imatrixhandle->ncols() == 3)
    {
      remark("Input matrix is 3x3.  Using rows/columns for vectors is ambiguous.");
    }
    CompileInfoHandle ci_mesh =
      ManageFieldDataAlgoMesh::
      get_compile_info(ifieldhandle->get_type_description(),
		       matrix_svt_flag,
		       gui_preserve_scalar_type_.get()?svt_flag:-1);
    Handle<ManageFieldDataAlgoMesh> algo_mesh;
    if (!module_dynamic_compile(ci_mesh, algo_mesh)) return;

    result_field =
      algo_mesh->execute(this, ifieldhandle->mesh(), imatrixhandle);

    if (!result_field.get_rep())
    {
      return;
    }

    // Copy the properties.
    result_field->copy_properties(ifieldhandle.get_rep());

    // Copy units property from the matrix.
    string units;
    if (imatrixhandle->get_property("units", units))
    {
      result_field->set_property("units", units, false);
    }
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
  ofp->send(result_field);
}



CompileInfoHandle
ManageFieldDataAlgoField::get_compile_info(const TypeDescription *fsrc,
					   int svt_flag)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("ManageFieldDataAlgoField");

  string extension;
  switch (svt_flag)
  {
  case 2:
    extension = "Tensor";
    break;

  case 1:
    extension = "Vector";
    break;

  default:
    extension = "Scalar";
    break;
  }

  CompileInfo *rval = 
    scinew CompileInfo(base_class_name + extension + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       base_class_name + extension, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}



CompileInfoHandle
ManageFieldDataAlgoMesh::get_compile_info(const TypeDescription *fsrc,
					  int svt_flag, int svt2)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("ManageFieldDataAlgoMesh");

  string extension;
  string extension2;
  switch (svt_flag)
  {
  case 3:
    extension = "Tensor6";
    extension2 = "Tensor";
    break;

  case 2:
    extension = "Tensor9";
    extension2 = "Tensor";
    break;

  case 1:
    extension = "Vector";
    extension2 = "Vector";
    break;

  default:
    extension = "Scalar";
    extension2 = "double";
    break;
  }

  string::size_type loc = fsrc->get_name().find_first_of("<");
  string fout = fsrc->get_name().substr(0, loc) + "<" + extension2 + "> ";
  if (svt_flag == 0 && svt2 == 0)
  {
    // Preserve file type if is scalar field.
    fout = fsrc->get_name();
  }

  CompileInfo *rval = 
    scinew CompileInfo(base_class_name + extension + "." +
		       to_filename(fout) + ".",
                       base_class_name, 
                       base_class_name + extension, 
                       fout);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}



} // End namespace SCIRun

