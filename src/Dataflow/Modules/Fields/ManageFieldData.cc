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
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ManageFieldData.h>
#include <Dataflow/GuiInterface/GuiVar.h>
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
    gui_preserve_scalar_type_(get_ctx()->subVar("preserve-scalar-type"), 0)
{
}



ManageFieldData::~ManageFieldData()
{
}



void
ManageFieldData::execute()
{
  // Get input field.
  FieldHandle ifieldhandle;
  if (!get_input_handle("Input Field", ifieldhandle)) return;

  // TODO: Using datasize this way appears to be wrong, as it depends
  // on the input DATA_AT size and not the one picked for output.
  unsigned int datasize = ifieldhandle->data_size();
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
      MatrixHandle mh(algo_field->execute(ifieldhandle));
      send_output_handle("Output Matrix", mh);
    }
  }

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
    string outtypestring;
    if (matrix_svt_flag == 3 || matrix_svt_flag == 2)
    {
      outtypestring = "Tensor";
    }
    else if (matrix_svt_flag == 1)
    {
      outtypestring = "Vector";
    }
    else if (gui_preserve_scalar_type_.get())
    {
      TypeDescription::td_vec *tdv = 
        ifieldhandle->get_type_description(Field::FDATA_TD_E)->get_sub_type();
      outtypestring = (*tdv)[0]->get_name();
    }
    else
    {
      outtypestring = "double";
    }
    
    string basisname = "";
    if (imatrixhandle->nrows() == (int)datasize ||
        imatrixhandle->ncols() == (int)datasize)
    {
      basisname = 
        ifieldhandle->get_type_description(Field::BASIS_TD_E)->get_similar_name(outtypestring,
                                                                0,
                                                                "<", " >, ");
    }
    else
    {
      if (ifieldhandle->get_type_description(Field::BASIS_TD_E)->get_name().find("Constant") == string::npos)
      {
        basisname = "ConstantBasis<" + outtypestring + ">, ";
      }
      else
      {
        TypeDescription::td_vec *bdv =
          ifieldhandle->get_type_description(Field::MESH_TD_E)->get_sub_type();
        const string linear = (*bdv)[0]->get_name();
        const string btype = linear.substr(0, linear.find_first_of('<'));
        basisname = btype + "<" + outtypestring + ">, ";
      }
    }

    const string oftn =
      ifieldhandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
      ifieldhandle->get_type_description(Field::MESH_TD_E)->get_name() + "," +
      basisname +
      ifieldhandle->get_type_description(Field::FDATA_TD_E)->get_similar_name(outtypestring, 0,
                                                              "<", " >") +
      " > ";
    CompileInfoHandle ci_mesh =
      ManageFieldDataAlgoMesh::
      get_compile_info(ifieldhandle->get_type_description(), oftn);
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

  send_output_handle("Output Field", result_field);
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
                                          const string &fout)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("ManageFieldDataAlgoMesh");


  CompileInfo *rval = 
    scinew CompileInfo(base_class_name +
		       to_filename(fout) + ".",
                       base_class_name, 
                       base_class_name + "T", 
                       fout);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_data_include("../src/Core/Geometry/Vector.h");
  rval->add_data_include("../src/Core/Geometry/Tensor.h");
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  fsrc->fill_compile_info(rval);
  return rval;
}



} // End namespace SCIRun

