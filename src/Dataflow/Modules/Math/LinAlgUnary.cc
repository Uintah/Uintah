
/*
 *  LinAlgUnary: Unary matrix operations -- transpose, negate
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
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Parts/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class LinAlgUnary : public Module {
  MatrixIPort* imat_;
  MatrixOPort* omat_;

  GuiString op_;
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

void LinAlgUnary::execute() {
  imat_ = (MatrixIPort *)get_iport("Input");
  omat_ = (MatrixOPort *)get_oport("Output");

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
  } else {
    warning("Don't know operation "+op);
    return;
  }
}
} // End namespace SCIRun
