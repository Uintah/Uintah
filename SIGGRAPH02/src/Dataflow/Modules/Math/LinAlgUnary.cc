/*
 *  LinAlgUnary: Unary matrix operations -- just transpose for now
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Math/function.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class LinAlgUnary : public Module {
  GuiString op_;
  GuiString function_;
  void insertion_sort(double *x, int n);
  void subtract_mean(double *x, int n);
public:
  LinAlgUnary(GuiContext* ctx);
  virtual ~LinAlgUnary();
  virtual void execute();
};

DECLARE_MAKER(LinAlgUnary)
LinAlgUnary::LinAlgUnary(GuiContext* ctx)
: Module("LinAlgUnary", ctx, Filter,"Math", "SCIRun"),
  op_(ctx->subVar("op")), function_(ctx->subVar("function"))
{
}

LinAlgUnary::~LinAlgUnary()
{
}

void LinAlgUnary::insertion_sort(double *x, int n) {
  double tmp;
  for (int i=0; i<n-1; i++)
    for (int j=i+1; j<n; j++)
      if (x[i] > x[j]) {
	tmp = x[i]; x[i]=x[j]; x[j]=tmp;
      }
}

void LinAlgUnary::subtract_mean(double *x, int n) {
  double sum = 0.0;
  for (int i=0; i<n; i++) {
    sum = sum + x[i];
  }
  double avg = sum / (double)n;
  for (int i=0; i<n; i++) {
    x[i] = x[i] - avg;
  }
}

void LinAlgUnary::execute() {
  MatrixIPort* imat_ = (MatrixIPort *)get_iport("Input");
  MatrixOPort* omat_ = (MatrixOPort *)get_oport("Output");

  if (!imat_) {
    error("Unable to initialize iport 'Input'.");
    return;
  }
  if (!omat_) {
    error("Unable to initialize oport 'Output'.");
    return;
  }
  
  update_state(NeedData);
  MatrixHandle mh;
  if (!imat_->get(mh))
    return;
  if (!mh.get_rep()) {
    warning("Empty input matrix.");
    return;
  }

  string op = op_.get();
  if (op == "Transpose") {
    Matrix *m = mh->transpose();
    omat_->send(MatrixHandle(m));
  } else if (op == "Sort") {
    MatrixHandle m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    insertion_sort(x, n);
    omat_->send(MatrixHandle(m));
  } else if (op == "Subtract_Mean") {
    MatrixHandle m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    subtract_mean(x, n);
    omat_->send(MatrixHandle(m));
  } else if (op == "Function") {
    Function *f = new Function(1);
    fnparsestring(function_.get().c_str(), &f);
    MatrixHandle m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    for (int i=0; i<n; i++)
      x[i]=f->eval(&(x[i]));
    omat_->send(MatrixHandle(m));
  } else {
    warning("Don't know operation "+op);
    return;
  }
}
} // End namespace SCIRun
