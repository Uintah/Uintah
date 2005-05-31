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
 *  MatrixToNrrd.cc: Converts a SCIRun Matrix to Nrrd(s).  
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Ports/MatrixPort.h>

namespace SCITeem {

using namespace SCIRun;

class MatrixToNrrd : public Module {
public:

  MatrixIPort* imat_;
  NrrdOPort*   ndata_;
  NrrdOPort*   nrows_;
  NrrdOPort*   ncols_;
  int          matrix_generation_;

  MatrixToNrrd(GuiContext*);

  virtual ~MatrixToNrrd();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  void create_and_send_column_matrix_nrrd(MatrixHandle mat);
  void create_and_send_dense_matrix_nrrd(MatrixHandle mat);
  void create_and_send_sparse_matrix_nrrd(MatrixHandle mat);

};


DECLARE_MAKER(MatrixToNrrd)
MatrixToNrrd::MatrixToNrrd(GuiContext* ctx)
  : Module("MatrixToNrrd", ctx, Source, "Converters", "Teem"),
    imat_(0), ndata_(0), nrows_(0), ncols_(0),
    matrix_generation_(-1)
{
}

MatrixToNrrd::~MatrixToNrrd(){
}

void
 MatrixToNrrd::execute(){
  // Get ports
  imat_ = (MatrixIPort *)get_iport("Matrix");
  ndata_ = (NrrdOPort *)get_oport("Data");
  nrows_ = (NrrdOPort *)get_oport("Rows");
  ncols_ = (NrrdOPort *)get_oport("Columns");

  // Determine if it is a Column, Dense or Sparse matrix
  MatrixHandle matH;
  if (!imat_->get(matH)) {
    return;
  }

  if (matrix_generation_ != matH->generation) {
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
MatrixToNrrd::create_and_send_column_matrix_nrrd(MatrixHandle matH) {
  ColumnMatrix* matrix = dynamic_cast<ColumnMatrix*>(matH.get_rep());

  int size = matrix->nrows();
  
  NrrdData *nd = scinew NrrdData();
  nrrdAlloc(nd->nrrd, nrrdTypeDouble, 1, size);
  nrrdAxisInfoSet(nd->nrrd, nrrdAxisInfoLabel, "column-data");
  nd->nrrd->axis[0].kind = nrrdKindDomain;

  double *val = (double*)nd->nrrd->data;
  double *data = matrix->get_data();

  for(int i=0; i<size; i++) {
    *val = *data;
    ++data;
    ++val;
  }

  // send the data nrrd
  NrrdDataHandle dataH(nd);
  ndata_->send(dataH);  
}


void
MatrixToNrrd::create_and_send_dense_matrix_nrrd(MatrixHandle matH) {
  DenseMatrix* matrix = dynamic_cast<DenseMatrix*>(matH.get_rep());

  int rows = matrix->nrows();
  int cols = matrix->ncols();
  
  NrrdData *nd = scinew NrrdData();
  nrrdAlloc(nd->nrrd, nrrdTypeDouble, 2, cols, rows);
  nrrdAxisInfoSet(nd->nrrd, nrrdAxisInfoLabel, "dense-columns" , "dense-rows");
  nd->nrrd->axis[0].kind = nrrdKindDomain;
  nd->nrrd->axis[1].kind = nrrdKindDomain;

  double *val = (double*)nd->nrrd->data;
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
  ndata_->send(dataH);  
}

void
MatrixToNrrd::create_and_send_sparse_matrix_nrrd(MatrixHandle matH) {
  SparseRowMatrix* matrix = dynamic_cast<SparseRowMatrix*>(matH.get_rep());

  int nnz = matrix->get_nnz();
  int rows = matrix->nrows();

  // create 3 nrrds (data, rows, cols)
  NrrdData *data_n = scinew NrrdData();
  nrrdAlloc(data_n->nrrd, nrrdTypeDouble, 1, nnz);
  nrrdAxisInfoSet(data_n->nrrd, nrrdAxisInfoLabel, "sparse-data");
  data_n->nrrd->axis[0].kind = nrrdKindDomain;

  NrrdData *rows_n = scinew NrrdData();
  nrrdAlloc(rows_n->nrrd, nrrdTypeInt, 1, rows+1);
  nrrdAxisInfoSet(rows_n->nrrd, nrrdAxisInfoLabel, "sparse-rows");
  rows_n->nrrd->axis[0].kind = nrrdKindDomain;

  NrrdData *cols_n = scinew NrrdData();
  nrrdAlloc(cols_n->nrrd, nrrdTypeInt, 1, nnz);
  nrrdAxisInfoSet(cols_n->nrrd, nrrdAxisInfoLabel, "sparse-columns");
  cols_n->nrrd->axis[0].kind = nrrdKindDomain;

  // pointers to nrrds
  double *data_p = (double*)data_n->nrrd->data;
  int *rows_p = (int*)rows_n->nrrd->data;
  int *cols_p = (int*)cols_n->nrrd->data;

  // points to matrix arrays
  int *rr = matrix->get_row();
  int *cc = matrix->get_col();
  double *d = matrix->get_val();

  // copy data and cols (size nnz)
  int i = 0;
  for (i=0; i<nnz; i++) {
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
  
  ndata_->send(dataH);
  nrows_->send(rowsH);
  ncols_->send(colsH);
}


void
 MatrixToNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


