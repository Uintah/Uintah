
/*
 *  MatrixSelectVector: Select a row or column of a matrix
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
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <sstream>

namespace SCIRun {

class MatrixSelectVector : public Module {
  MatrixIPort* imat_;
  MatrixOPort* ovec_;
  GuiString row_or_col_;
  GuiInt row_;
  GuiInt row_max_;
  GuiInt col_;
  GuiInt col_max_;
  GuiInt animate_;
public:
  MatrixSelectVector(const clString& id);
  virtual ~MatrixSelectVector();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
  int stop_;
};

extern "C" Module* make_MatrixSelectVector(const clString& id)
{
    return new MatrixSelectVector(id);
}

MatrixSelectVector::MatrixSelectVector(const clString& id)
: Module("MatrixSelectVector", id, Filter), animate_("animate_", id, this),
  col_("col_", id, this), col_max_("col_max_", id, this),
  row_("row_", id, this), row_max_("row_max_", id, this),
  row_or_col_("row_or_col_", id, this)
{
  // Create the input port
  imat_=new MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
  add_iport(imat_);
  
  // Create the output port
  ovec_=new MatrixOPort(this,"Vector", MatrixIPort::Atomic);
  add_oport(ovec_);
  stop_=0;
}

MatrixSelectVector::~MatrixSelectVector()
{
}

void MatrixSelectVector::execute() {
  stop_=0;
  update_state(NeedData);
  MatrixHandle mh;
  if (!imat_->get(mh))
    return;
  if (!mh.get_rep()) {
    cerr << "Error: empty matrix\n";
    return;
  }
  
  int changed=0;
  
  update_state(JustStarted);
  
  if (col_max_.get() != mh->ncols()-1) {
    col_max_.set(mh->ncols()-1);
    changed=1;
  }
  if (row_max_.get() != mh->nrows()-1) {
    row_max_.set(mh->nrows()-1);
    changed=1;
  }
  if (changed) {
    std::ostringstream str;
    str << id << " update";
    TCL::execute(str.str().c_str());
  }
  
  reset_vars();
  
  int which;
  int use_row=(row_or_col_.get() == "row");
  if (use_row) which=row_.get();
  else which=col_.get();
  
  if (!animate_.get()) {
    ColumnMatrix *cm;
    if (use_row) {
      cm=new ColumnMatrix(mh->ncols());
      double *data = cm->get_rhs();
      for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
    } else {
      cm=new ColumnMatrix(mh->nrows());
      double *data = cm->get_rhs();
      for (int r=0; r<mh->nrows(); r++) data[r]=mh->get(r, which);
    }	    
    MatrixHandle cmh(cm);
    ovec_->send(cmh);
  } else {
    ColumnMatrix *cm;
    if (use_row) {
      for (; which<mh->nrows()-1; which++, row_.set(which)) {
	if (stop_) { stop_=0; break; }
	cerr << which << "\n";
	cm=new ColumnMatrix(mh->ncols());
	double *data = cm->get_rhs();
	for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
	MatrixHandle cmh(cm);
	ovec_->send_intermediate(cmh);
      }
    } else {
      for (; which<mh->ncols()-1; which++, col_.set(which)) {
	if (stop_) { stop_=0; break; }
	cerr << which << "\n";
	cm=new ColumnMatrix(mh->nrows());
	double *data = cm->get_rhs();
	for (int r=0; r<mh->nrows(); r++) data[r]=mh->get(r, which);
	MatrixHandle cmh(cm);
	ovec_->send_intermediate(cmh);
      }
    }	    
    if (use_row) {
      cm=new ColumnMatrix(mh->ncols());
      double *data = cm->get_rhs();
      for (int c=0; c<mh->ncols(); c++) data[c]=mh->get(which, c);
    } else {
      cm=new ColumnMatrix(mh->nrows());
      double *data = cm->get_rhs();
      for (int r=0; r<mh->nrows(); r++) data[r]=mh->get(r, which);
    }	    
    MatrixHandle cmh(cm);
    ovec_->send(cmh);
  }
}    


void MatrixSelectVector::tcl_command(TCLArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("MatrixSelectVector needs a minor command");
    return;
  }
  if (args[1] == "stop_") stop_=1;
  else Module::tcl_command(args, userdata);
}
} // End namespace SCIRun
