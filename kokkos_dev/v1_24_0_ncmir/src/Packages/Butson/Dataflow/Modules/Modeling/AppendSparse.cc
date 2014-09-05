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
 *  AppendSparse: Unary matrix operations -- transpose, negate
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Datatypes/SparseRowMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class AppendSparse : public Module {
private:

  GuiString rows_or_columns_;
  
public:
  AppendSparse(GuiContext* ctx);
  virtual ~AppendSparse();
  virtual void execute();
};

DECLARE_MAKER(AppendSparse);

AppendSparse::AppendSparse(GuiContext* ctx)
  : Module("AppendSparse", ctx, Filter,"Modeling", "Butson"),
    rows_or_columns_(ctx->subVar("appendmode"))
{
}

AppendSparse::~AppendSparse()
{
}

void
AppendSparse::execute()
{
  bool column_append = rows_or_columns_.get() == "columns";

  MatrixIPort *imat1 = (MatrixIPort *)get_iport("A");
  MatrixIPort *imat2 = (MatrixIPort *)get_iport("B");
  
  MatrixOPort *omat = (MatrixOPort *)get_oport("Output");

  if (!imat1) {
    error("Unable to initialize iport 'A'.");
    return;
  }
  if (!imat2) {
    error("Unable to initialize iport 'B'.");
    return;
  }
  if (!omat) {
    error("Unable to initialize oport 'Output'.");
    return;
  }
  
  update_state(NeedData);

  MatrixHandle im1H;
  if (!imat1->get(im1H))
    return;

  if (!im1H.get_rep()) {
    warning("Empty input matrix.");
    return;
  }

  MatrixHandle im2H;
  if (!imat2->get(im2H))
    return;

  if (!im2H.get_rep()) {
    warning("Empty input matrix.");
    return;
  }

  SparseRowMatrix *smat1 = im1H->sparse();
  SparseRowMatrix *smat2 = im2H->sparse();
  
  if (smat1 == 0 || smat2 == 0)
  {
    error("Only works on sparse matrices.");
    return;
  }

  if (column_append)
  {
    if (smat1->nrows() != smat2->nrows())
    {
      error("Input matrices must contain same number of rows.");
      return;
    }      
    smat1 = smat1->transpose();
    smat2 = smat2->transpose();
  }

  if (smat1->ncols() != smat2->ncols())
  {
    error("Input matrices must contain same number of columns.");
    return;
  }

  const int nnz1 = smat1->get_nnz();
  const int nnz2 = smat2->get_nnz();
  int i;

  int *rows = scinew int[smat1->nrows() + smat2->nrows() + 1];
  int *cols = scinew int[nnz1 + nnz2];
  double *vals = scinew double[nnz1 + nnz2];

  memcpy(rows, smat1->get_row(), smat1->nrows() * sizeof(int));
  int offset = smat1->get_row()[smat1->nrows()];
  for (i=0; i <= smat2->nrows(); i++)
  {
    rows[smat1->nrows() + i] = smat2->get_row()[i] + offset;
  }

  memcpy(cols, smat1->get_col(), nnz1 * sizeof(int));
  memcpy(cols+nnz1, smat2->get_col(), nnz2 * sizeof(int));

  memcpy(vals, smat1->get_val(), nnz1 * sizeof(double));
  memcpy(vals+nnz1, smat2->get_val(), nnz2 * sizeof(double));

  SparseRowMatrix *om = scinew SparseRowMatrix(smat1->nrows() + smat2->nrows(),
					       smat1->ncols(),
					       rows, cols, nnz1 + nnz2, vals);

  if (column_append)
  {
    SparseRowMatrix *tmp = om->transpose();
    delete om;
    om = tmp;
    delete smat1;
    delete smat2;
  }

  MatrixHandle omH(om);
  omat->send(omH);
}
} // End namespace SCIRun
