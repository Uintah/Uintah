
/*
 *  CastMatrix: Unary matrix operations -- transpose, negate
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

class CastMatrix : public Module {
  MatrixIPort* imat_;
  MatrixOPort* omat_;

  GuiString oldtype_;
  GuiString newtype_;
public:
  CastMatrix(const string& id);
  virtual ~CastMatrix();
  virtual void execute();
};

extern "C" Module* make_CastMatrix(const string& id)
{
    return new CastMatrix(id);
}

CastMatrix::CastMatrix(const string& id)
: Module("CastMatrix", id, Filter,"Math", "SCIRun"),
  oldtype_("oldtype", id, this), newtype_("newtype", id, this)
{
}

CastMatrix::~CastMatrix()
{
}

void CastMatrix::execute() {
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
  MatrixHandle imH;
  if (!imat_->get(imH))
    return;
  if (!imH.get_rep()) {
    warning("Empty input matrix.");
    return;
  }

  string newtype = newtype_.get();
  MatrixHandle omH;

  if (newtype == "DenseMatrix") {
    omH = imH->dense();
  } else if (newtype == "SparseRowMatrix") {
    omH = imH->sparse();
  } else if (newtype == "ColumnMatrix") {
    omH = imH->column();
  } else {
    error("CastMatrix: unknown cast type "+newtype);
    return;
  }
  omat_->send(omH);
}
} // End namespace SCIRun
