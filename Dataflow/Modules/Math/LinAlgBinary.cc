
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
      error( "No handle or representation" );
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
    if (!aH.get_rep()) {
      error("Empty A matrix for Add");
      return;
    }
    if (!bH.get_rep()) {
      error("Empty B matrix for Add");
      return;
    }
    if (aH->ncols() != bH->ncols() || aH->nrows() != bH->nrows())
    {
      error("Addition requires A and B must be the same size.");
      return;
    }
    if (aH->ncols() == 1)
    {
      ColumnMatrix *ac, *bc, *cc;
      ac = aH->column();
      bc = bH->column();
      cc = scinew ColumnMatrix(ac->nrows());
      Add(*cc, *ac, *bc);
      omat_->send(MatrixHandle(cc));
    }
    else if (dynamic_cast<SparseRowMatrix *>(aH.get_rep()) &&
	     dynamic_cast<SparseRowMatrix *>(bH.get_rep()))
    {
      SparseRowMatrix *as, *bs, *cs;
      as = aH->sparse();
      bs = bH->sparse();
      cs = AddSparse(*as, *bs);
      omat_->send(MatrixHandle(cs));
    }
    else
    {
      DenseMatrix *ad, *bd, *cd;
      ad=aH->dense();
      bd=bH->dense();
      cd=scinew DenseMatrix(ad->nrows(), bd->ncols());
      Add(*cd, *ad, *bd);
      omat_->send(MatrixHandle(cd));
    }
    return;
  } else if (op == "Mult") {
    DenseMatrix *ad, *bd, *cd;
    if (!aH.get_rep()) {
      error("Empty A matrix for Add");
      return;
    }
    if (!bH.get_rep()) {
      error("Empty B matrix for Add");
      return;
    }
    if (aH->ncols() != bH->nrows())
    {
      error("Matrix multiply requires the number of columns in A to be the same as the number of rows in B.");
      return;
    }
    ad=aH->dense();
    bd=bH->dense();
    cd=scinew DenseMatrix(ad->nrows(), bd->ncols());
    Mult(*cd, *ad, *bd);
    omat_->send(MatrixHandle(cd));
    return;
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
  } else if (op == "SelectColumns") {
    if (!aH.get_rep() || !bH.get_rep()) {
      cerr << "LinAlgBinary:SelectColumns Error - can't have an empty input matrix for this operation.\n";
      return;
    }
    ColumnMatrix *bc = dynamic_cast<ColumnMatrix *>(bH.get_rep());
    if (!bc) {
      cerr << "LinAlgBinary:SelectColumns Error - second input must be a ColumnMatrix!\n";
      return;
    }
    DenseMatrix *cd = scinew DenseMatrix(aH->nrows(), bc->nrows());
    for (int i=0; i<cd->ncols(); i++) {
      int idx = (int)(*bc)[i];
      if (idx == -1) continue;
      if (idx > aH->ncols()) {
	cerr << "LinAlgBinary:SelectColumns Error - tried to select a column ("<<idx<<") that was out of range ("<<aH->ncols()<<")\n";
	return;
      }
      for (int j=0; j<aH->nrows(); j++) {
	(*cd)[j][i]=aH->get(j,idx);
      }
    }
    if (dynamic_cast<DenseMatrix *>(aH.get_rep()))
      omat_->send(MatrixHandle(cd));
    else if (dynamic_cast<ColumnMatrix *>(aH.get_rep())) {
      omat_->send(MatrixHandle(cd->column()));
      delete cd;
    } else {
      omat_->send(MatrixHandle(cd->sparse()));
      delete cd;
    }
    return;
  } else if (op == "SelectRows") {
    if (!aH.get_rep() || !bH.get_rep()) {
      cerr << "LinAlgBinary:SelectRows Error - can't have an empty input matrix for this operation.\n";
      return;
    }
    ColumnMatrix *bc = dynamic_cast<ColumnMatrix *>(bH.get_rep());
    if (!bc) {
      cerr << "LinAlgBinary:SelectRows Error - second input must be a ColumnMatrix!\n";
      return;
    }
    DenseMatrix *cd = scinew DenseMatrix(bc->nrows(), aH->ncols());
    for (int i=0; i<cd->nrows(); i++) {
      int idx = (int)(*bc)[i];
      if (idx == -1) continue;
      if (idx > aH->nrows()) {
	cerr << "LinAlgBinary:SelectRows Error - tried to select a row ("<<idx<<") that was out of range ("<<aH->nrows()<<")\n";
	return;
      }
      for (int j=0; j<aH->ncols(); j++) {
	(*cd)[i][j]=aH->get(idx,j);
      }
    }
    if (dynamic_cast<DenseMatrix *>(aH.get_rep()))
      omat_->send(MatrixHandle(cd));
    else if (dynamic_cast<ColumnMatrix *>(aH.get_rep())) {
      omat_->send(MatrixHandle(cd->column()));
      delete cd;
    } else {
      omat_->send(MatrixHandle(cd->sparse()));
      delete cd;
    }
    return;
  } else if (op == "NormalizeAtoB") {
    if (!aH.get_rep() || !bH.get_rep()) {
      cerr << "LinAlgBinary:NormalizeAtoB Error - can't have an empty input matrix for this operation.\n";
      return;
    }
    double amin, amax, bmin, bmax;
    MatrixHandle anewH = aH->clone();
    double *a = &((*(aH.get_rep()))[0][0]);
    double *anew = &((*(anewH.get_rep()))[0][0]);
    double *b = &((*(bH.get_rep()))[0][0]);
    int na = aH->nrows()*aH->ncols();
    int nb = bH->nrows()*bH->ncols();
    amin=amax=a[0];
    bmin=bmax=b[0];
    int i;
    for (i=1; i<na; i++) {
      if (a[i]<amin) amin=a[i];
      else if (a[i]>amax) amax=a[i];
    }
    for (i=1; i<nb; i++) {
      if (b[i]<bmin) bmin=b[i];
      else if (b[i]>bmax) bmax=b[i];
    }
    double da=amax-amin;
    double db=bmax-bmin;
    double scale = db/da;
    for (i=0; i<na; i++)
      anew[i] = (a[i]-amin)*scale+bmin;
    omat_->send(anewH);
  } else {
    warning("Don't know operation "+op);
    return;
  }
}
} // End namespace SCIRun
