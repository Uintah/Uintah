/*
 *  AppendMatrix: Matrix operations -- concatenate, replace
 *
 *  Written by:
 *   David Weinstein &
 *   Chris Butson
 *   Department of Computer Science
 *   University of Utah
 *   July 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <stdio.h>

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class AppendMatrix : public Module {
  MatrixIPort* imatA_;
  MatrixIPort* imatB_;
  MatrixOPort* omat_;
  MatrixHandle matrixH_;

  GuiInt append_;   // append or replace
  GuiInt row_;      // row or column
  GuiInt front_;    // append at the beginning or the end
  void concat_cols(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out);
  void concat_rows(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out);
public:
  AppendMatrix(GuiContext* ctx);
  virtual ~AppendMatrix();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void *);
};

DECLARE_MAKER(AppendMatrix)
AppendMatrix::AppendMatrix(GuiContext* ctx)
: Module("AppendMatrix", ctx, Filter,"Math", "SCIRun"),
  append_(ctx->subVar("append")),
  row_(ctx->subVar("row")),
  front_(ctx->subVar("front"))
{
}

AppendMatrix::~AppendMatrix()
{
}

void AppendMatrix::concat_cols(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out) {

    int r, c;
    for (r = 0; r <= m1H->nrows()-1; r++)
    {
      for (c = 0; c <= m1H->ncols()-1; c++)
      {
        out->put(r, c, m1H->get(r,c));
      }
    }

    for (r = 0; r <= m2H->nrows()-1; r++)
    {
      for (c = m1H->ncols(); c <= m1H->ncols()+m2H->ncols()-1; c++)
      {
        out->put(r, c, m2H->get(r,c - m1H->ncols()));
      }
    }

}

void AppendMatrix::concat_rows(MatrixHandle m1H, MatrixHandle m2H, DenseMatrix *out) {

    int r, c;
    for (r = 0; r <= m1H->nrows()-1; r++)
    {
      for (c = 0; c <= m1H->ncols()-1; c++)
      {
        out->put(r, c, m1H->get(r,c));
      }
    }

    for (r = m1H->nrows(); r <= m1H->nrows()+m2H->nrows()-1; r++)
    {
      for (c = 0; c <= m2H->ncols()-1; c++)
      {
        out->put(r, c, m2H->get(r - m1H->nrows(), c));
      }
    }

}

void AppendMatrix::execute() {
  imatA_ = (MatrixIPort *)get_iport("Optional BaseMatrix");
  imatB_ = (MatrixIPort *)get_iport("SubMatrix");
  omat_ = (MatrixOPort *)get_oport("CompositeMatrix");

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
  if (!imatA_->get(aH)) {
    if (!imatB_->get(bH))
      return;
  } else imatB_->get(bH);

  if (!bH.get_rep()) {
    warning("Empty input matrix Submatrix.");
    return;
  }

  DenseMatrix *omatrix = 0;

  bool append = append_.get();
  bool row = row_.get();
  bool front = front_.get();

  if (!append) { matrixH_=bH; omat_->send(matrixH_); } // Replace -- just send B matrix
  else if (!aH.get_rep()) { // No A matrix
    if (!matrixH_.get_rep()) { matrixH_=bH; omat_->send(matrixH_); }// No previous CompositeMatrix, so send B
    else {   // Previous CompositeMatrix exists, concatenate with B
      //DenseMatrix *oldmatrix=matrixH_->dense();
      if (row) {
	if (matrixH_->ncols() != bH->ncols()) {
	  warning("SubMatrix and CompositeMatrix must have same number of columns");
	  return;
	} else {
	  omatrix=scinew DenseMatrix(matrixH_->nrows()+bH->nrows(),matrixH_->ncols());
	  if (front) concat_rows(bH,matrixH_,omatrix);
	  else concat_rows(matrixH_,bH,omatrix);
	}
      } else {
	if (matrixH_->nrows() != bH->nrows()) {
	  warning("SubMatrix and CompositeMatrix must have same number of rows");
	  return;
	} else {
	  omatrix=scinew DenseMatrix(matrixH_->nrows(),matrixH_->ncols()+bH->ncols());
	  if (front) concat_cols(bH,matrixH_,omatrix);
	  else concat_cols(matrixH_,bH,omatrix);
	} // columns
      } // rows - columns
    } // previous matrix exists
  } else { // A exists
    if (row) {
      if (aH->ncols() != bH->ncols()) {
	warning("BaseMatrix and CompositeMatrix must have same number of columns");
	return;
      } else {
	omatrix=scinew DenseMatrix(aH->nrows()+bH->nrows(),aH->ncols());
	if (front) concat_rows(bH,aH,omatrix);
	else concat_rows(aH,bH,omatrix);
      }
    } else {
      if (aH->nrows() != bH->nrows()) {
	warning("BaseMatrix and CompositeMatrix must have same number of rows");
	return;
      } else {
	omatrix=scinew DenseMatrix(aH->nrows(),aH->ncols()+bH->ncols());
	if (front) concat_cols(bH,aH,omatrix);
	else concat_cols(aH,bH,omatrix);
      } // columns
    } // rows - columns
  } // A exists

  if (omatrix)
  {
    matrixH_=omatrix;
    omat_->send(matrixH_);
  }
}

void
AppendMatrix::tcl_command(GuiArgs& args, void* userdata)
{

  if (args[1] == "clear")
  {
    //    DenseMatrix *omatrix = 0;
    matrixH_=0;
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }

}

} // End namespace SCIRun


