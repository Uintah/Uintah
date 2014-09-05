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
  int data_generation_;
  int rows_generation_;
  int cols_generation_;
  bool has_error_;

  MatrixHandle last_matrix_;

  GuiInt cols_;
  int old_cols_;

  NrrdToMatrix(GuiContext*);

  virtual ~NrrdToMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  MatrixHandle create_matrix_from_nrrds(NrrdDataHandle dataH,
					NrrdDataHandle rowsH,
					NrrdDataHandle colsH,
					int cols);

  template<class PTYPE> 
  MatrixHandle create_column_matrix(NrrdDataHandle dataH);

  template<class PTYPE>
  MatrixHandle create_dense_matrix(NrrdDataHandle dataH);

  template<class PTYPE>
  MatrixHandle create_sparse_matrix(NrrdDataHandle dataH, NrrdDataHandle rowsH,
				    NrrdDataHandle colsH, int cols);
};


DECLARE_MAKER(NrrdToMatrix)
NrrdToMatrix::NrrdToMatrix(GuiContext* ctx)
  : Module("NrrdToMatrix", ctx, Source, "DataIO", "Teem"),
    ndata_(0), nrows_(0), ncols_(0), omat_(0),
    data_generation_(-1), rows_generation_(-1), 
    cols_generation_(-1), has_error_(false),
    last_matrix_(0), cols_(ctx->subVar("cols")),
    old_cols_(-1)
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

  NrrdDataHandle dataH;
  NrrdDataHandle rowsH;
  NrrdDataHandle colsH;

  bool do_execute = false;
  // Determine if we have data, points, connections, etc.
  if (!ndata_->get(dataH)) {
    dataH = 0;
    if (data_generation_ != -1) {
      data_generation_ = -1;
      do_execute = true;
    }
  }

  if (!nrows_->get(rowsH)){
    rowsH = 0;
    if (rows_generation_ != -1) {
      rows_generation_ = -1;
      do_execute = true;
    }
  }

  if (!ncols_->get(colsH)) {
    colsH = 0;
    if (cols_generation_ != -1) {
      cols_generation_ = -1;
      do_execute = true;
    }
  }

  // check the generations to see if we need to re-execute
  if (dataH != 0 && data_generation_ != dataH->generation) {
    data_generation_ = dataH->generation;
    do_execute = true;
  }
  if (rowsH != 0 && rows_generation_ != rowsH->generation) {
    rows_generation_ = rowsH->generation;
    do_execute = true;
  }
  if (colsH != 0 && cols_generation_ != colsH->generation) {
    cols_generation_ = colsH->generation;
    do_execute = true;
  }
  if (old_cols_ != cols_.get()) {
    old_cols_ = cols_.get();
    do_execute = true;
  }

  if (has_error_)
    do_execute = true;
  
  if (do_execute) {
    last_matrix_ = create_matrix_from_nrrds(dataH, rowsH, colsH, cols_.get());
  }

  if (last_matrix_ != 0) {
    has_error_ = false;
    omat_->send(last_matrix_);  
  }
}

MatrixHandle
NrrdToMatrix::create_matrix_from_nrrds(NrrdDataHandle dataH, NrrdDataHandle rowsH,
				       NrrdDataHandle colsH, int cols) {


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
  if (has_data && (!has_rows || !has_cols)) {
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
	has_error_ = true;
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
	has_error_ = true;
	return 0;
      }
    } else {
      error("Can only convert data nrrds of 1 or 2D (Column or Dense Matrix).");
      has_error_ = true;
      return 0;
    }
  } else if (has_data && has_rows && has_cols) {
    // sparse matrix

      // rows and cols should be of type nrrdTypeInt
      if (rowsH->nrrd->type != nrrdTypeInt || colsH->nrrd->type != nrrdTypeInt) {
        error("Rows and Columns nrrds must both be of type nrrdTypeInt");
	has_error_ = true;
        return 0;
      }
      
      if (dataH->nrrd->dim != 1 || rowsH->nrrd->dim != 1 || colsH->nrrd->dim != 1) {
	error("All nrrds must be 1 dimension for a SparseRowMatrix.");
	has_error_ = true;
	return 0;
      }
      switch(dataH->nrrd->type) {
      case nrrdTypeChar:
	matrix = create_sparse_matrix<char>(dataH, rowsH, colsH, cols);
	break;
      case nrrdTypeUChar:
	matrix = create_sparse_matrix<unsigned char>(dataH, rowsH, colsH, cols);
	break;
      case nrrdTypeShort:
	matrix = create_sparse_matrix<short>(dataH, rowsH, colsH, cols);
	break;
      case nrrdTypeUShort:
	matrix = create_sparse_matrix<unsigned short>(dataH, rowsH, colsH, cols);
	break;
      case nrrdTypeInt:
	matrix = create_sparse_matrix<int>(dataH, rowsH, colsH, cols);
	break;
      case nrrdTypeUInt:
	matrix = create_sparse_matrix<unsigned int>(dataH, rowsH, colsH, cols);
	break;
      case nrrdTypeFloat:
	matrix = create_sparse_matrix<float>(dataH, rowsH, colsH, cols);
	break;
      case nrrdTypeDouble:
	matrix = create_sparse_matrix<double>(dataH, rowsH, colsH, cols);
	break;
      default:
	error("Unkown nrrd type.");
	has_error_ = true;
	return 0;
      }
  } else {
    error("Must have data to convert to any type of Matrix.  Must have rows and columns for a SparseRowMatrix.");
    has_error_ = true;
    return 0;
  }

  return matrix;
}

template<class PTYPE> 
MatrixHandle 
NrrdToMatrix::create_column_matrix(NrrdDataHandle dataH) {
  remark("Creating column matrix");
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
  remark("Creating dense matrix");
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
				   NrrdDataHandle colsH, int cols) {

  remark("Creating sparse row matrix");
  Nrrd *data_n = dataH->nrrd;
  Nrrd *rows_n = rowsH->nrrd;
  Nrrd *cols_n = colsH->nrrd;

  // pointers to nnrds
  PTYPE *data_d = (PTYPE*)data_n->data;
  int *rows_d = (int*)rows_n->data;
  int *cols_d = (int*)cols_n->data;

  if (cols == -1) {
    // Auto selected...attempt to determine number of columns
    for (int i=0; i<cols_n->axis[0].size; i++) {
      if (cols_d[i] > cols)
	cols = cols_d[i];
    }
    cols += 1; 
  }
  
  int rows = rows_n->axis[0].size-1;
  int offset = 0;
  if (rows_d[0] != 0) {
    warning("First entry of rows nrrd must be a 0. Inserting 0 in first position.");
    offset = 1;
    rows++;
  }

  int nnz = data_n->axis[0].size;

  // error checking...

  // cols_n and dn should be of size nnz
  if (cols_n->axis[0].size != nnz) {
    error("The Data and Columns nrrds should be the same size.");
    has_error_ = true;
    return 0;
  }
  
  // rows values must be in increasing order
  for (int i=0; i<rows_n->axis[0].size-1; i++) {
    if (rows_d[i] > rows_d[i+1] || rows_d[i] < 0) {
      error("Rows nrrd must contain values in increasing order and positive.");
      has_error_ = true;
      return 0;
    }
  }

  // last rows value should be less than nnz
  if (rows_d[rows_n->axis[0].size-1] > nnz) {
    error("The last entry in the rows array must be less than the number of non zeros.");
    has_error_ = true;
    return 0;
  }

  for (int i=0; i<nnz; i++) {
    if (cols_d[i] < 0) {
      error("Columns nrrd must have positive values");
      has_error_ = true;
      return 0;
    }
  }

  // for each rows[N+1] - rows[N] sections of the cols array,
  // those values must be in increasing order
  for (int i=0; i<rows_n->axis[0].size-1; i++) {
    int span = rows_d[i+1] - rows_d[i];
    for(int j=i;j<(i+span-1);j++) {
      if (cols_d[j] > cols_d[j+1]) {
	error("Columns nrrd ordered incorrectly.");
	has_error_ = true;
	return 0;
      }
    }
  }

  // pointers to matrix arrays
  double *d = scinew double[nnz];
  int *rr = scinew int[rows_n->axis[0].size+offset];
  int *cc = scinew int[nnz];
  
  // copy rest of rows
  if (offset == 1) {
    rr[0] = 0;
  }
  for (int i=0; i<rows_n->axis[0].size; i++) {
    rr[i+offset] = rows_d[i];
  }
	
  // copy data and cols
  for(int i=0; i<nnz; i++) {
    cc[i] = cols_d[i];
    d[i] = data_d[i];
  }


  SparseRowMatrix *matrix = scinew SparseRowMatrix(rows, cols, rr, cc, nnz, d);

  MatrixHandle result(matrix);

  return result;
}


void
 NrrdToMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


