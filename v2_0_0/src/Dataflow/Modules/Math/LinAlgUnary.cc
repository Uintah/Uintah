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
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Math/function.h>
#include <iostream>
#include <sstream>
#include <math.h>

namespace SCIRun {

class LinAlgUnary : public Module {
  GuiString op_;
  GuiString function_;
  void insertion_sort(double *x, int n);
  void subtract_mean(double *x, int n);
  void normalize(double *x, int n);
  void round(double *x, int n);
  void Floor(double *x, int n);
  void Ceil(double *x, int n);
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

void LinAlgUnary::normalize(double *x, int n) {
  double min =  1.0e36;
  double max = -1.0e36;
  for (int i=0; i<n; i++) {
    if( min > x[i] ) min = x[i];
    else if( max < x[i] ) max = x[i];
  }
  double mult = 1.0 / (max-min);
  for (int i=0; i<n; i++) {
    x[i] = (int) ((x[i]-min) * mult);
  }
}

void LinAlgUnary::round(double *x, int n) {
  for (int i=0; i<n; i++) {
    x[i] = (int) (x[i] + (x[i] < 0 ?  -0.5 : +0.5) );
  }
}

void LinAlgUnary::Floor(double *x, int n) {
  for (int i=0; i<n; i++) {
    x[i] = floor(x[i]);
  }
}

void LinAlgUnary::Ceil(double *x, int n) {
  for (int i=0; i<n; i++) {
    x[i] = ceil(x[i]);
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
  if (!imat_->get(mh) || !mh.get_rep()) {
    error( "No handle or representation" );
    return;
  }

  string op = op_.get();

  MatrixHandle m;

  if (op == "Transpose") {
    Matrix *mat = mh->transpose();
    m = mat;
  } else if (op == "Invert") {
    DenseMatrix *dm = mh->dense();
    if (! dm->invert()) {
      error("Input Matrix not invertible.");
      return;
    }
    m = dm;
  } else if (op == "Sort") {
    m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    insertion_sort(x, n);
  } else if (op == "Subtract_Mean") {
    m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    subtract_mean(x, n);
  } else if (op == "Normalize") {
    m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    normalize(x, n);
  } else if (op == "Round") {
    m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    round(x, n);
  } else if (op == "Floor") {
    m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    Floor(x, n);
  } else if (op == "Ceil") {
    m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    Ceil(x, n);
  } else if (op == "Function") {
    Function *f = new Function(1);
    fnparsestring(function_.get().c_str(), &f);
    m = mh->clone();
    double *x = &((*(m.get_rep()))[0][0]);
    int n = m->nrows()*m->ncols();
    for (int i=0; i<n; i++)
      x[i]=f->eval(&(x[i]));
  } else {
    warning("Don't know operation "+op);
    return;
  }

  omat_->send(MatrixHandle(m));
}
} // End namespace SCIRun
