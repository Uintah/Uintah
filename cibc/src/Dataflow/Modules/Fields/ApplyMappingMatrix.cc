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
#include <Core/Datatypes/MatrixOperations.h>
#include <Core/Algorithms/Fields/ApplyMappingMatrix.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ApplyMappingMatrix : public Module
{
private:
  FieldHandle  field_output_handle_;

public:
  ApplyMappingMatrix(GuiContext* ctx);
  virtual ~ApplyMappingMatrix();

  virtual void execute();
};


DECLARE_MAKER(ApplyMappingMatrix)

ApplyMappingMatrix::ApplyMappingMatrix(GuiContext* ctx)
  : Module("ApplyMappingMatrix", ctx, Filter, "FieldsData", "SCIRun")
{
}


ApplyMappingMatrix::~ApplyMappingMatrix()
{
}


void
ApplyMappingMatrix::execute()
{
  // Get source field.
  FieldHandle src_fh;
  if( !get_input_handle( "Source", src_fh, true ) ) return;

  // Get destination field.
  FieldHandle dst_fh;
  if( !get_input_handle( "Destination", dst_fh, true ) ) return;

  // Get the mapping matrix.
  MatrixHandle matrix_input_handle;
  if( !get_input_handle( "Mapping", matrix_input_handle, true ) ) return;

  // Check to see if the source has changed.
  if( inputs_changed_ ||
      !field_output_handle_.get_rep() )
  {
    TypeDescription::td_vec *tdv = 
      src_fh->get_type_description(Field::FDATA_TD_E)->get_sub_type();
    string accumtype = (*tdv)[0]->get_name();
    if (src_fh->query_scalar_interface(this) != NULL) { accumtype = "double"; }

    const TypeDescription* fno = 
      dst_fh->get_type_description(Field::FIELD_NAME_ONLY_E);
    const TypeDescription* fm = dst_fh->get_type_description(Field::MESH_TD_E);
    const TypeDescription* fb = 
      dst_fh->get_type_description(Field::BASIS_TD_E);
    const TypeDescription* fd = 
      dst_fh->get_type_description(Field::FDATA_TD_E);

    const string oftn = 
      fno->get_name() + "<" + fm->get_name() + ", " +
      fb->get_similar_name(accumtype, 0, "<", " >, ") +
      fd->get_similar_name(accumtype, 0, "<", " >") + " >";
    
    CompileInfoHandle ci =
      ApplyMappingMatrixAlgo::get_compile_info(src_fh->get_type_description(),
			       src_fh->order_type_description(),
			       dst_fh->get_type_description(),
			       oftn,
			       dst_fh->order_type_description(),
			       src_fh->get_type_description(Field::FDATA_TD_E),
			       accumtype);
    Handle<ApplyMappingMatrixAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    field_output_handle_ =
      algo->execute(this,
		    src_fh,
		    dst_fh->mesh(),
		    matrix_input_handle);


    // Copy the properties from source field.
    if (field_output_handle_.get_rep())
      field_output_handle_->copy_properties(src_fh.get_rep());
  }

  send_output_handle( "Output",  field_output_handle_, true );
}






} // namespace SCIRun
