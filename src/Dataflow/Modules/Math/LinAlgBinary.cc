
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
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class LinAlgBinary : public Module {
  MatrixIPort* imatA_;
  MatrixIPort* imatB_;
  MatrixOPort* omat_;

  GuiString op_;
public:
  LinAlgBinary(const string& id);
  virtual ~LinAlgBinary();
  virtual void execute();
};

extern "C" Module* make_LinAlgBinary(const string& id)
{
    return new LinAlgBinary(id);
}

LinAlgBinary::LinAlgBinary(const string& id)
: Module("LinAlgBinary", id, Filter,"Math", "SCIRun"),
  op_("op", id, this)
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
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!imatB_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!omat_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  update_state(NeedData);
  MatrixHandle aH, bH;
  if (!imatA_->get(aH))
    return;
  if (!aH.get_rep()) {
    warning("Empty input matrix.");
    return;
  }
  if (!imatB_->get(bH))
    return;
  if (!bH.get_rep()) {
    warning("Empty input matrix.");
    return;
  }

  string op = op_.get();
  if (op == "Add") {
//    Matrix *m;
//    omat_->send(MatrixHandle(m));
    error("LinAlgBinary: Add has not been implemented yet.");
    return;
  } else if (op == "Mult") {
    if (aH->nrows() == 4 && aH->ncols() == 4 &&
	bH->nrows() == 4 && bH->ncols() == 4) {
      Transform aT(aH->toTransform());
      Transform bT(bH->toTransform());
      aT.post_trans(bT);
      DenseMatrix *cm = scinew DenseMatrix(aT);
      omat_->send(MatrixHandle(cm));
      return;
    } else {
      error("LinAlgBinary: Mult has only been implemented for 4x4 matrices.");
      return;
    }
  } else {
    warning("Don't know operation "+op);
    return;
  }
}
} // End namespace SCIRun
