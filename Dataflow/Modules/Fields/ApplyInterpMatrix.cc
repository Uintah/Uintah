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

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/MatrixOperations.h>
#include <Dataflow/Modules/Fields/ManageFieldData.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ApplyInterpMatrix : public Module
{
public:
  ApplyInterpMatrix(GuiContext* ctx);

  virtual void execute();
};


DECLARE_MAKER(ApplyInterpMatrix)

ApplyInterpMatrix::ApplyInterpMatrix(GuiContext* ctx)
  : Module("ApplyInterpMatrix", ctx, Filter, "FieldsData", "SCIRun")
{
}


void
ApplyInterpMatrix::execute()
{
  // Get source field.
  FieldIPort *sfp = (FieldIPort *)get_iport("Source");
  FieldHandle sfieldhandle;
  if (!sfp) {
    error("Unable to initialize iport 'Source'.");
    return;
  }
  if (!(sfp->get(sfieldhandle) && (sfieldhandle.get_rep())))
  {
    error( "No handle or representation." );
    return;
  }

  // TODO: Using datasize this way appears to be wrong, as it depends
  // on the input DATA_AT size and not the one picked for output.
  int datasize = 0;
  int svt_flag = 0;
  if (sfieldhandle->query_scalar_interface(this).get_rep())
  {
    svt_flag = 0;
  }
  else if (sfieldhandle->query_vector_interface(this).get_rep())
  {
    svt_flag = 1;
  }
  else if (sfieldhandle->query_tensor_interface(this).get_rep())
  {
    svt_flag = 2;
  }

  MatrixHandle smatrix(0);

  // Extract the source mantrix.
  if (sfieldhandle->data_at() == Field::NONE)
  {
    error("Input field contains no data, nothing to interpolate.");
    return;
  }
  else
  {
    CompileInfoHandle ci_field =
      ManageFieldDataAlgoField::
      get_compile_info(sfieldhandle->get_type_description(), svt_flag);
    Handle<ManageFieldDataAlgoField> algo_field;
    if (!DynamicCompilation::compile(ci_field, algo_field, true, this))
    {
      error("Unable to extract data from input field.");
      return;
    }
    else
    {
      smatrix = algo_field->execute(sfieldhandle, datasize);
    }
  }

  // Get the interpolant matrix.
  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Interpolant");
  MatrixHandle imatrixhandle;
  if (!imatrix_port) {
    error("Unable to initialize iport 'Interpolant'.");
    return;
  }
  if (!(imatrix_port->get(imatrixhandle) && imatrixhandle.get_rep()))
  {
    error("No interpolant matrix connected.");
    return;
  }

  // Do the matrix multiply.
  MatrixHandle dmatrix = imatrixhandle * smatrix;

  // Get destination field.
  FieldIPort *dfp = (FieldIPort *)get_iport("Destination");
  FieldHandle dfieldhandle;
  if (!dfp) {
    error("Unable to initialize iport 'Destination'.");
    return;
  }
  if (!(dfp->get(dfieldhandle) && (dfieldhandle.get_rep())))
  {
    error( "No field available on the 'Destination' port." );
    return;
  }

  int matrix_svt_flag = svt_flag;
  if (dmatrix->nrows() == 9)
  {
    matrix_svt_flag = 2;
  }
  else if (dmatrix->nrows() == 3)
  {
    matrix_svt_flag = 1;
  }
  else if (dmatrix->nrows() == 1)
  {
    matrix_svt_flag = 0;
  }

  CompileInfoHandle ci_mesh =
    ManageFieldDataAlgoMesh::
    get_compile_info(dfieldhandle->get_type_description(),
		     matrix_svt_flag,
		     svt_flag);
  Handle<ManageFieldDataAlgoMesh> algo_mesh;
  if (!module_dynamic_compile(ci_mesh, algo_mesh)) return;

  FieldHandle result_field =
    algo_mesh->execute(this, dfieldhandle->mesh(), dmatrix);

  if (!result_field.get_rep())
  {
    return;
  }

  // Copy the properties.
  *((PropertyManager *)(result_field.get_rep())) =
    *((PropertyManager *)(sfieldhandle.get_rep()));

  FieldOPort *ofp = (FieldOPort *)get_oport("Output");
  if (!ofp) {
    error("Unable to initialize oport 'Output'.");
    return;
  }

  ofp->send(result_field);
}


} // namespace SCIRun
