
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
#include <sstream>

namespace SCIRun {

class MatrixSelectVector : public Module {
  MatrixIPort* imat_;
  MatrixIPort* ivec_;
  MatrixOPort* ovec_;
  
  GuiInt row_;
  GuiInt row_max_;
  GuiInt col_;
  GuiInt col_max_;
  GuiString row_or_col_;
  GuiInt animate_;
public:
  MatrixSelectVector(const string& id);
  virtual ~MatrixSelectVector();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
  bool stop_;
};


extern "C" Module* make_MatrixSelectVector(const string& id)
{
  return new MatrixSelectVector(id);
}


MatrixSelectVector::MatrixSelectVector(const string& id)
  : Module("MatrixSelectVector", id, Filter,"Math", "SCIRun"),
    row_("row", id, this),
    row_max_("row_max", id, this),
    col_("col", id, this),
    col_max_("col_max", id, this),
    row_or_col_("row_or_col", id, this),
    animate_("animate", id, this),
    stop_(false)
{
}


MatrixSelectVector::~MatrixSelectVector()
{
}


void
MatrixSelectVector::execute()
{
  imat_ = (MatrixIPort *)get_iport("Matrix");
  ivec_ = (MatrixIPort *)get_iport("Weight Vector");
  ovec_ = (MatrixOPort *)get_oport("Vector");

  if (!imat_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ivec_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ovec_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  stop_ = false;
  update_state(NeedData);
  MatrixHandle mh;
  if (!imat_->get(mh))
    return;
  if (!mh.get_rep()) {
    warning("Empty input matrix.");
    return;
  }
  
  bool changed = false;
  
  update_state(JustStarted);
  
  if (col_max_.get() != mh->ncols()-1)
  {
    col_max_.set(mh->ncols()-1);
    changed = true;
  }
  if (row_max_.get() != mh->nrows()-1)
  {
    row_max_.set(mh->nrows()-1);
    changed = true;
  }
  if (changed)
  {
    std::ostringstream str;
    str << id << " update";
    TCL::execute(str.str().c_str());
  }
  
  reset_vars();
  const bool use_row = (row_or_col_.get() == "row");

  MatrixHandle weightsH;
  if (ivec_->get(weightsH) && weightsH.get_rep())
  {
    ColumnMatrix *w = dynamic_cast<ColumnMatrix*>(weightsH.get_rep());
    ColumnMatrix *cm;
    if (use_row) 
    {
      cm = scinew ColumnMatrix(mh->ncols());
      cm->zero();
      double *data = cm->get_data();
      for (int i = 0; i<w->nrows()/2; i++)
      {
	const int idx = (int)((*w)[i*2]);
	double wt = (*w)[i*2+1];
	for (int j = 0; j<mh->ncols(); j++)
	{
	  data[j]+=mh->get(idx, j)*wt;
	}
      }
    }
    else
    {
      cm = scinew ColumnMatrix(mh->nrows());
      cm->zero();
      double *data = cm->get_data();
      for (int i = 0; i<w->nrows()/2; i++)
      {
	const int idx = (int)((*w)[i*2]);
	double wt = (*w)[i*2+1];
	for (int j = 0; j<mh->nrows(); j++)
	{
	  data[j]+=mh->get(j, idx)*wt;
	}
      }
    }
    ovec_->send(MatrixHandle(cm));
    return;
  }

  int which = use_row?row_.get():col_.get();

  if (!animate_.get())
  {
    ColumnMatrix *cm;
    if (use_row)
    {
      cm = scinew ColumnMatrix(mh->ncols());
      double *data = cm->get_data();
      for (int c = 0; c<mh->ncols(); c++)
      {
	data[c] = mh->get(which, c);
      }
    }
    else
    {
      cm = scinew ColumnMatrix(mh->nrows());
      double *data = cm->get_data();
      for (int r = 0; r<mh->nrows(); r++)
      {
	data[r] = mh->get(r, which);
      }
    }	    
    MatrixHandle cmh(cm);
    ovec_->send(cmh);
  }
  else
  {
    ColumnMatrix *cm;
    if (use_row)
    {
      for (; which<mh->nrows()-1; which++, row_.set(which))
      {
	if (stop_) { stop_ = false; break; }
	cm = scinew ColumnMatrix(mh->ncols());
	double *data = cm->get_data();
	for (int c = 0; c<mh->ncols(); c++)
	{
	  data[c] = mh->get(which, c);
	}
	MatrixHandle cmh(cm);
	ovec_->send_intermediate(cmh);
      }
    }
    else
    {
      for (; which<mh->ncols()-1; which++, col_.set(which))
      {
	if (stop_) { stop_ = false; break; }
	cm = scinew ColumnMatrix(mh->nrows());
	double *data = cm->get_data();
	for (int r = 0; r<mh->nrows(); r++)
	{
	  data[r] = mh->get(r, which);
	}
	MatrixHandle cmh(cm);
	ovec_->send_intermediate(cmh);
      }
    }	    
    if (use_row)
    {
      cm = scinew ColumnMatrix(mh->ncols());
      double *data = cm->get_data();
      for (int c = 0; c<mh->ncols(); c++)
      {
	data[c] = mh->get(which, c);
      }
    }
    else
    {
      cm = scinew ColumnMatrix(mh->nrows());
      double *data = cm->get_data();
      for (int r = 0; r<mh->nrows(); r++)
      {
	data[r] = mh->get(r, which);
      }
    }	    
    MatrixHandle cmh(cm);
    ovec_->send(cmh);
  }
}    


void
MatrixSelectVector::tcl_command(TCLArgs& args, void* userdata)
{
  if (args.count() < 2)
  {
    args.error("MatrixSelectVector needs a minor command");
    return;
  }
  if (args[1] == "stop") stop_ = true;
  else Module::tcl_command(args, userdata);
}


} // End namespace SCIRun
