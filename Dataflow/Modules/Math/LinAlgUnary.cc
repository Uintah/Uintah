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
#include <iostream>
#include <sstream>

namespace SCIRun {

class LinAlgUnary : public Module {
  GuiString op_;
  void insertion_sort(double *x, int n);
public:
  LinAlgUnary(const string& id);
  virtual ~LinAlgUnary();
  virtual void execute();
};

extern "C" Module* make_LinAlgUnary(const string& id)
{
    return new LinAlgUnary(id);
}

LinAlgUnary::LinAlgUnary(const string& id)
: Module("LinAlgUnary", id, Filter,"Math", "SCIRun"),
  op_("op", id, this)
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

void LinAlgUnary::execute() {
  MatrixIPort* imat_ = (MatrixIPort *)get_iport("Input");
  MatrixOPort* omat_ = (MatrixOPort *)get_oport("Output");

  if (!imat_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!omat_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
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
  } else {
    warning("Don't know operation "+op);
    return;
  }
}
} // End namespace SCIRun
