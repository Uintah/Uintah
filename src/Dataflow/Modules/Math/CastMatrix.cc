
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

  DenseMatrix *dm;
  ColumnMatrix *cm;
  SparseRowMatrix *sm;

  if (newtype == "Dense") {
    if (dynamic_cast<DenseMatrix *>(imH.get_rep())) {
      oldtype_.set("DenseMatrix");
      omH = imH;
    } else if ((cm = dynamic_cast<ColumnMatrix *>(imH.get_rep()))) {
      oldtype_.set("ColumnMatrix");
      omH = cm->toDense();
    } else if ((sm = dynamic_cast<SparseRowMatrix *>(imH.get_rep()))) {
      oldtype_.set("SparseRowMatrix");
      omH = sm->toDense();
    } else {
      error("CastMatrix: failed to determine type of input matrix.\n");
      return;
    }
  } else if (newtype == "Sparse") {
    if (dynamic_cast<SparseRowMatrix *>(imH.get_rep())) {
      oldtype_.set("SparseRowMatrix");
      omH = imH;
    } else if ((dm = dynamic_cast<DenseMatrix *>(imH.get_rep()))) {
      oldtype_.set("DenseMatrix");
      omH = dm->toSparse();
    } else if ((cm = dynamic_cast<ColumnMatrix *>(imH.get_rep()))) {
      oldtype_.set("ColumnMatrix");
      omH = cm->toSparse();
    } else {
      error("CastMatrix: failed to determine type of input matrix.\n");
      return;
    }
  } else if (newtype == "Column") {
    if (dynamic_cast<ColumnMatrix *>(imH.get_rep())) {
      oldtype_.set("ColumnMatrix");
      omH = imH;
    } else if ((dm = dynamic_cast<DenseMatrix *>(imH.get_rep()))) {
      oldtype_.set("DenseMatrix");
      omH = dm->toColumn();
    } else if ((sm = dynamic_cast<SparseRowMatrix *>(imH.get_rep()))) {
      oldtype_.set("SparseRowMatrix");
      omH = sm->toColumn();
    } else {
      error("CastMatrix: failed to determine type of input matrix.\n");
      return;
    }
  } else if (newtype == "Same") {
    if (dynamic_cast<ColumnMatrix *>(imH.get_rep())) {
      oldtype_.set("ColumnMatrix");
    } else if (dynamic_cast<DenseMatrix *>(imH.get_rep())) {
      oldtype_.set("DenseMatrix");
    } else if (dynamic_cast<SparseRowMatrix *>(imH.get_rep())) {
      oldtype_.set("SparseRowMatrix");
    } else {
      error("CastMatrix: failed to determine type of input matrix.\n");
      oldtype_.set("Error");
    }
    omH = imH;
  } else {
    error("CastMatrix: unknown cast type "+newtype);
    return;
  }
  omat_->send(omH);
}
} // End namespace SCIRun
