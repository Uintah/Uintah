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
 *  NrrdToMatrix.cc: Converts Nrrd(s) to a SCIRun Matrix.  It may convert
 *                   it to a ColumnMatrix, DenseMatrix, or SparseMatrix
 *                   depending on which ports are connected.
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Teem/Core/Datatypes/NrrdData.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Ports/MatrixPort.h>

namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE NrrdToMatrix : public Module {
public:
  NrrdIPort*   ndata_;
  NrrdIPort*   nrows_;
  NrrdIPort*   ncols_;
  MatrixOPort* omat_;

  GuiInt nnz_;


  NrrdToMatrix(GuiContext*);

  virtual ~NrrdToMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  MatrixHandle create_matrix_from_nrrds(NrrdDataHandle dataH,
					NrrdDataHandle rowsH,
					NrrdDataHandle colsH,
					int nnz);

  template<class PTYPE> 
  MatrixHandle create_column_matrix(NrrdDataHandle dataH);

  template<class PTYPE>
  MatrixHandle create_dense_matrix(NrrdDataHandle dataH);

  template<class PTYPE>
  MatrixHandle create_sparse_matrix(NrrdDataHandle dataH, NrrdDataHandle rowsH,
				    NrrdDataHandle colsH, int nnz);
};


DECLARE_MAKER(NrrdToMatrix)
NrrdToMatrix::NrrdToMatrix(GuiContext* ctx)
  : Module("NrrdToMatrix", ctx, Source, "DataIO", "Teem"),
    nnz_(ctx->subVar("nnz"))
{
}

NrrdToMatrix::~NrrdToMatrix(){
}

void
 NrrdToMatrix::execute(){
  // Get ports
  ndata_ = (NrrdIPort *)get_iport("Data");
  nrows_ = (NrrdIPort *)get_iport("Rows");
  ncols_ = (NrrdIPort *)get_iport("Columns");
  omat_ = (MatrixOPort *)get_oport("Matrix");

  if (!ndata_) {
    error("Unable to initialize iport 'Data'.");
    return;
  }
  if (!nrows_) {
    error("Unable to initialize iport 'Rows'.");
    return;
  }
  if (!ncols_) {
    error("Unable to initialize iport 'Columns'.");
    return;
  }
  if (!omat_) {
    error("Unable to initialize oport 'Matrix'.");
    return;
  }

  cerr << "FIX ME : NEED TO CHECK GENERATIONS BEFORE EXECUTING\n";

  NrrdDataHandle dataH;
  NrrdDataHandle rowsH;
  NrrdDataHandle colsH;

  // Determine if we have data, points, connections, etc.
  if (!ndata_->get(dataH))
    dataH = 0;
  if (!nrows_->get(rowsH))
    rowsH = 0;
  if (!ncols_->get(colsH))
    colsH = 0;

  MatrixHandle omat_handle = create_matrix_from_nrrds(dataH, rowsH, colsH, nnz_.get());

  omat_->send(omat_handle);  
}

MatrixHandle
NrrdToMatrix::create_matrix_from_nrrds(NrrdDataHandle dataH, NrrdDataHandle rowsH,
				       NrrdDataHandle colsH, int nnz) {


  // Determine if we have data, rows, columns to indicate whether it is
  // a dense or sparse matrix
  bool has_data = false, has_rows = false, has_cols = false;

  if (dataH != 0)
    has_data = true;
  if (rowsH != 0)
    has_rows = true;
  if (colsH != 0)
    has_cols = true;

  MatrixHandle matrix;
  if (has_data && !has_rows && !has_cols) {
    if (dataH->nrrd->dim == 1) {
      // column matrix
      switch(dataH->nrrd->type) {
      case nrrdTypeChar:
	matrix = create_column_matrix<char>(dataH);
	break;
      case nrrdTypeUChar:
	matrix = create_column_matrix<unsigned char>(dataH);
	break;
      case nrrdTypeShort:
	matrix = create_column_matrix<short>(dataH);
	break;
      case nrrdTypeUShort:
	matrix = create_column_matrix<unsigned short>(dataH);
	break;
      case nrrdTypeInt:
	matrix = create_column_matrix<int>(dataH);
	break;
      case nrrdTypeUInt:
	matrix = create_column_matrix<unsigned int>(dataH);
	break;
      case nrrdTypeFloat:
	matrix = create_column_matrix<float>(dataH);
	break;
      case nrrdTypeDouble:
	matrix = create_column_matrix<double>(dataH);
	break;
      default:
	error("Unkown nrrd type.");
	return 0;
      }
    } else if (dataH->nrrd->dim == 2) {
      // dense matrix
      switch(dataH->nrrd->type) {
      case nrrdTypeChar:
	matrix = create_dense_matrix<char>(dataH);
	break;
      case nrrdTypeUChar:
	matrix = create_dense_matrix<unsigned char>(dataH);
	break;
      case nrrdTypeShort:
	matrix = create_dense_matrix<short>(dataH);
	break;
      case nrrdTypeUShort:
	matrix = create_dense_matrix<unsigned short>(dataH);
	break;
      case nrrdTypeInt:
	matrix = create_dense_matrix<int>(dataH);
	break;
      case nrrdTypeUInt:
	matrix = create_dense_matrix<unsigned int>(dataH);
	break;
      case nrrdTypeFloat:
	matrix = create_dense_matrix<float>(dataH);
	break;
      case nrrdTypeDouble:
	matrix = create_dense_matrix<double>(dataH);
	break;
      default:
	error("Unkown nrrd type.");
	return 0;
      }
    } else {
      error("Can only convert data nrrds of 1 or 2D (Column or Dense Matrix).");
      return 0;
    }
  } else if (has_data && has_rows && has_cols) {
    // sparse matrix
      switch(dataH->nrrd->type) {
      case nrrdTypeChar:
	matrix = create_sparse_matrix<char>(dataH, rowsH, colsH, nnz);
	break;
      case nrrdTypeUChar:
	matrix = create_sparse_matrix<unsigned char>(dataH, rowsH, colsH, nnz);
	break;
      case nrrdTypeShort:
	matrix = create_sparse_matrix<short>(dataH, rowsH, colsH, nnz);
	break;
      case nrrdTypeUShort:
	matrix = create_sparse_matrix<unsigned short>(dataH, rowsH, colsH, nnz);
	break;
      case nrrdTypeInt:
	matrix = create_sparse_matrix<int>(dataH, rowsH, colsH, nnz);
	break;
      case nrrdTypeUInt:
	matrix = create_sparse_matrix<unsigned int>(dataH, rowsH, colsH, nnz);
	break;
      case nrrdTypeFloat:
	matrix = create_sparse_matrix<float>(dataH, rowsH, colsH, nnz);
	break;
      case nrrdTypeDouble:
	matrix = create_sparse_matrix<double>(dataH, rowsH, colsH, nnz);
	break;
      default:
	error("Unkown nrrd type.");
	return 0;
      }
  } else {
    error("Must have data.  Must have rows and columns for a sparse matrix.");
    return 0;
  }

  return matrix;
}

template<class PTYPE> 
MatrixHandle 
NrrdToMatrix::create_column_matrix(NrrdDataHandle dataH) {

  int rows = dataH->nrrd->axis[0].size;

  ColumnMatrix* matrix = scinew ColumnMatrix(rows);
  
  PTYPE *val = (PTYPE*)dataH->nrrd->data;
  double *data = matrix->get_data();

  for(int i=0; i<dataH->nrrd->axis[0].size; i++) {
    *data = *val;
    ++data;
    ++val;
  }

  MatrixHandle result(matrix);
  return result;
}

template<class PTYPE>
MatrixHandle 
NrrdToMatrix::create_dense_matrix(NrrdDataHandle dataH) {

  int rows = dataH->nrrd->axis[1].size;
  int cols = dataH->nrrd->axis[0].size;

  DenseMatrix* matrix = scinew DenseMatrix(rows,cols);
  
  PTYPE *val = (PTYPE*)dataH->nrrd->data;
  double *data = matrix->getData();

  for(int r=0; r<rows; r++) {
    for(int c=0; c<cols; c++) {
      *data = *val;
      ++data;
      ++val;
    }
  }

  MatrixHandle result(matrix);
  return result;
}

template<class PTYPE>
MatrixHandle 
NrrdToMatrix::create_sparse_matrix(NrrdDataHandle dataH, NrrdDataHandle rowsH,
				   NrrdDataHandle colsH, int nnz) {
  
  error("Not implemented for Sparse Matrices yet.");
  return 0;
}


void
 NrrdToMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


