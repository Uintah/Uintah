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
 *  MatrixToNrrd.cc: Converts a SCIRun Matrix to Nrrd(s).  
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

class PSECORESHARE MatrixToNrrd : public Module {
public:

  MatrixIPort* imat_;
  NrrdOPort*   ndata_;
  NrrdOPort*   nrows_;
  NrrdOPort*   ncols_;

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
  : Module("MatrixToNrrd", ctx, Source, "DataIO", "Teem")
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

  if (!imat_) {
    error("Unable to initialize iport 'Matrix'.");
    return;
  }
  if (!ndata_) {
    error("Unable to initialize oport 'Data'.");
    return;
  }
  if (!nrows_) {
    error("Unable to initialize oport 'Rows'.");
    return;
  }
  if (!ncols_) {
    error("Unable to initialize oport 'Columns'.");
    return;
  }
  
  // Determine if it is a Column, Dense or Sparse matrix
  MatrixHandle matH;
  if (!imat_->get(matH)) {
    return;
  }

  Matrix* matrix = matH.get_rep();
  
  if (matrix->is_column()) {
    create_and_send_column_matrix_nrrd(matH);
  } else if (matrix->is_dense()) {
    create_and_send_dense_matrix_nrrd(matH);
  } else {
    create_and_send_sparse_matrix_nrrd(matH);
  }
}

void
MatrixToNrrd::create_and_send_column_matrix_nrrd(MatrixHandle matH) {
  ColumnMatrix* matrix = dynamic_cast<ColumnMatrix*>(matH.get_rep());

  int size = matrix->nrows();
  
  NrrdData *nd = scinew NrrdData();
  nrrdAlloc(nd->nrrd, nrrdTypeDouble, 1, size);
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
  nd->nrrd->axis[0].kind = nrrdKindDomain;
  nd->nrrd->axis[1].kind = nrrdKindDomain;

  double *val = (double*)nd->nrrd->data;
  double *data = matrix->getData();

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
  
  error("Not implemented for Sparse Matrices yet.");
}


void
 MatrixToNrrd::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


