
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
#include <Core/Parts/GuiVar.h>
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
    ColumnMatrix *bC = dynamic_cast<ColumnMatrix*>(bH.get_rep());
    if (!bC) {
      error("LinAlgBinary: Mult has only been implemented for Mat x ColMat.");
      return;
    }
    
    //    ColumnMatrix *r;
//    omat_->send(MatrixHandle(m));
    return;
  } else {
    warning("Don't know operation "+op);
    return;
  }
}
} // End namespace SCIRun
