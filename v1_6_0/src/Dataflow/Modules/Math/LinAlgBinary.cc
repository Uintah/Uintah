
/*
 *  LinAlgBinary: Binary matrix operations -- add, multiply
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Math/function.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class LinAlgBinary : public Module {
  MatrixIPort* imatA_;
  MatrixIPort* imatB_;
  MatrixOPort* omat_;

  GuiString op_;
  GuiString function_;
public:
  LinAlgBinary(GuiContext* ctx);
  virtual ~LinAlgBinary();
  virtual void execute();
};

DECLARE_MAKER(LinAlgBinary)
LinAlgBinary::LinAlgBinary(GuiContext* ctx)
: Module("LinAlgBinary", ctx, Filter,"Math", "SCIRun"),
  op_(ctx->subVar("op")), function_(ctx->subVar("function"))
{
}

LinAlgBinary::~LinAlgBinary()
{
}

void LinAlgBinary::execute() {
  imatA_ = (MatrixIPort *)get_iport("A");
  imatB_ = (MatrixIPort *)get_iport("B");
  omat_ = (MatrixOPort *)get_oport("Output");

  if (!imatA_) {
    error("Unable to initialize iport 'A'.");
    return;
  }
  if (!imatB_) {
    error("Unable to initialize iport 'B'.");
    return;
  }
  if (!omat_) {
    error("Unable to initialize oport 'Output'.");
    return;
  }
  
  update_state(NeedData);
  MatrixHandle aH, bH;
  if (!imatA_->get(aH)) {
    if (!imatB_->get(bH))
      return;
  } else imatB_->get(bH);
      
  if (!aH.get_rep()) {
    warning("Empty input matrix A.");
  }
  if (!bH.get_rep()) {
    warning("Empty input matrix B.");
  }

  string op = op_.get();
  if (op == "Add") {
//    Matrix *m;
//    omat_->send(MatrixHandle(m));
    error("LinAlgBinary: Add has not been implemented yet.");
    return;
  } else if (op == "Mult") {
    if (!aH.get_rep()) {
      aH = scinew DenseMatrix(Transform());
    } else if (!bH.get_rep()) {
      bH = scinew DenseMatrix(Transform());
    }
    if (aH->nrows() == 4 && aH->ncols() == 4 &&
	bH->nrows() == 4 && bH->ncols() == 4) {
      Transform aT(aH->toTransform());
      Transform bT(bH->toTransform());
      aT.post_trans(bT);
      DenseMatrix *cm = scinew DenseMatrix(aT);
      omat_->send(MatrixHandle(cm));
      return;
    } else {
      error("Mult has only been implemented for 4x4 matrices.");
      return;
    }
  } else if (op == "Function") {
    if (aH->nrows()*aH->ncols() != bH->nrows()*bH->ncols()) {
      error("Function only works if input matrices have the same number of elements.");
      return;
    }
    Function *f = new Function(1);
    fnparsestring(function_.get().c_str(), &f);
    MatrixHandle m = aH->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    double *a = &((*(aH.get_rep()))[0][0]);
    double *b = &((*(bH.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    double *ab = new double[2];
    for (int i=0; i<n; i++) {
      ab[0]=a[i]; ab[1]=b[i];
      x[i]=f->eval(ab);
    }
    omat_->send(MatrixHandle(m));
    delete[] ab;
  } else {
    warning("Don't know operation "+op);
    return;
  }
}
} // End namespace SCIRun
