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
 *  MatrixToNrrd.cc: Converts a SCIRun Matrix to Nrrd(s).  
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Network/Ports/MatrixPort.h>

namespace SCITeem {

using namespace SCIRun;

class MatrixToNrrd : public Module {
public:

  NrrdOPort*   ndata_;
  NrrdOPort*   nrows_;
  NrrdOPort*   ncols_;
  int          matrix_generation_;

  MatrixToNrrd(GuiContext*);

  virtual ~MatrixToNrrd();

  virtual void execute();

  void create_and_send_column_matrix_nrrd(MatrixHandle mat);
  void create_and_send_dense_matrix_nrrd(MatrixHandle mat);
  void create_and_send_sparse_matrix_nrrd(MatrixHandle mat);

};


DECLARE_MAKER(MatrixToNrrd)
MatrixToNrrd::MatrixToNrrd(GuiContext* ctx)
  : Module("MatrixToNrrd", ctx, Source, "Converters", "Teem"),
    ndata_(0), nrows_(0), ncols_(0),
    matrix_generation_(-1)
{
}


MatrixToNrrd::~MatrixToNrrd()
{
}


void
MatrixToNrrd::execute()
{
  // Get ports
  ndata_ = (NrrdOPort *)get_oport("Data");
  nrows_ = (NrrdOPort *)get_oport("Rows");
  ncols_ = (NrrdOPort *)get_oport("Columns");

  MatrixHandle matH;
  if (!get_input_handle("Matrix", matH)) return;

  if (matrix_generation_ != matH->generation ||
      !ndata_->have_data() &&
      !nrows_->have_data() &&
      !ncols_->have_data())
  {
    matrix_generation_ = matH->generation;
    Matrix* matrix = matH.get_rep();
    
    if (matrix->is_column()) {
      create_and_send_column_matrix_nrrd(matH);
    } else if (matrix->is_dense()) {
      create_and_send_dense_matrix_nrrd(matH);
    } else {
      create_and_send_sparse_matrix_nrrd(matH);
    }
  }
}


void
MatrixToNrrd::create_and_send_column_matrix_nrrd(MatrixHandle matH)
{
  ColumnMatrix* matrix = matH->as_column();

  size_t size[NRRD_DIM_MAX];
  size[0] = matrix->nrows();
  
  NrrdData *nd = scinew NrrdData();
  nrrdAlloc_nva(nd->nrrd_, nrrdTypeDouble, 1, size);
  nrrdAxisInfoSet_nva(nd->nrrd_, nrrdAxisInfoLabel, "column-data");
  nd->nrrd_->axis[0].kind = nrrdKindDomain;

  double *val = (double*)nd->nrrd_->data;
  double *data = matrix->get_data();

  for(unsigned int i=0; i<size[0]; i++) {
    *val = *data;
    ++data;
    ++val;
  }

  // Send the data nrrd.
  NrrdDataHandle dataH(nd);
  ndata_->send_and_dereference(dataH);  
}


void
MatrixToNrrd::create_and_send_dense_matrix_nrrd(MatrixHandle matH)
{
  DenseMatrix* matrix = matH->as_dense();

  int rows = matrix->nrows();
  int cols = matrix->ncols();
  
  NrrdData *nd = scinew NrrdData();
  size_t size[NRRD_DIM_MAX];
  size[0] = cols; size[1] = rows;
  nrrdAlloc_nva(nd->nrrd_, nrrdTypeDouble, 2, size);

  char *labels[NRRD_DIM_MAX];
  labels[0] = airStrdup("dense-columns");
  labels[1] = airStrdup("dense-rows");
  nrrdAxisInfoSet_nva(nd->nrrd_, nrrdAxisInfoLabel, labels);
  nd->nrrd_->axis[0].kind = nrrdKindDomain;
  nd->nrrd_->axis[1].kind = nrrdKindDomain;

  double *val = (double*)nd->nrrd_->data;
  double *data = matrix->get_data_pointer();

  for(int r=0; r<rows; r++) {
    for(int c=0; c<cols; c++) {
      *val = *data;
      ++data;
      ++val;
    }
  }
  // send the data nrrd
  NrrdDataHandle dataH(nd);
  ndata_->send_and_dereference(dataH);  
}


void
MatrixToNrrd::create_and_send_sparse_matrix_nrrd(MatrixHandle matH)
{
  SparseRowMatrix* matrix = matH->as_sparse();

  size_t nnz[NRRD_DIM_MAX];
  nnz[0] = matrix->get_nnz();
  unsigned int rows = matrix->nrows();

  // create 3 nrrds (data, rows, cols)
  NrrdData *data_n = scinew NrrdData();
  nrrdAlloc_nva(data_n->nrrd_, nrrdTypeDouble, 1, nnz);
  nrrdAxisInfoSet_nva(data_n->nrrd_, nrrdAxisInfoLabel, "sparse-data");
  data_n->nrrd_->axis[0].kind = nrrdKindDomain;

  NrrdData *rows_n = scinew NrrdData();
  size_t sparse_size[NRRD_DIM_MAX];
  sparse_size[0] = rows + 1;
  nrrdAlloc_nva(rows_n->nrrd_, nrrdTypeInt, 1, sparse_size);
  nrrdAxisInfoSet_nva(rows_n->nrrd_, nrrdAxisInfoLabel, "sparse-rows");
  rows_n->nrrd_->axis[0].kind = nrrdKindDomain;

  NrrdData *cols_n = scinew NrrdData();
  nrrdAlloc_nva(cols_n->nrrd_, nrrdTypeInt, 1, nnz);
  nrrdAxisInfoSet_nva(cols_n->nrrd_, nrrdAxisInfoLabel, "sparse-columns");
  cols_n->nrrd_->axis[0].kind = nrrdKindDomain;

  // pointers to nrrds
  double *data_p = (double*)data_n->nrrd_->data;
  int *rows_p = (int*)rows_n->nrrd_->data;
  int *cols_p = (int*)cols_n->nrrd_->data;

  // points to matrix arrays
  int *rr = matrix->get_row();
  int *cc = matrix->get_col();
  double *d = matrix->get_val();

  // copy data and cols (size nnz)
  unsigned int i = 0;
  for (i=0; i<nnz[0]; i++) {
    data_p[i] = d[i];
    cols_p[i] = cc[i];
  }

  // copy rows (size rows+1)
  for (i=0; i<rows+1; i++) {
    rows_p[i] = rr[i];
  }
  
  // send nrrds
  NrrdDataHandle dataH(data_n);
  NrrdDataHandle rowsH(rows_n);
  NrrdDataHandle colsH(cols_n);
  
  ndata_->send_and_dereference(dataH);
  nrows_->send_and_dereference(rowsH);
  ncols_->send_and_dereference(colsH);
}


} // End namespace Teem


