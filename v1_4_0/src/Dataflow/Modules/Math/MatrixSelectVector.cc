
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
  GuiString row_or_col_;
  GuiDouble selectable_min_;
  GuiDouble selectable_max_;
  GuiInt    selectable_inc_;
  GuiString selectable_units_;
  GuiInt    range_min_;
  GuiInt    range_max_;
  GuiString playmode_;
  GuiInt    current_;
  GuiString execmode_;
  int       inc_;
  bool      stop_;

public:
  MatrixSelectVector(const string& id);
  virtual ~MatrixSelectVector();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};


extern "C" Module* make_MatrixSelectVector(const string& id)
{
  return new MatrixSelectVector(id);
}


MatrixSelectVector::MatrixSelectVector(const string& id)
  : Module("MatrixSelectVector", id, Filter,"Math", "SCIRun"),
    row_or_col_("row_or_col", id, this),
    selectable_min_("selectable_min", id, this),
    selectable_max_("selectable_max", id, this),
    selectable_inc_("selectable_inc", id, this),
    selectable_units_("selectable_units", id, this),
    range_min_("range_min", id, this),
    range_max_("range_max", id, this),
    playmode_("playmode", id, this),
    current_("current", id, this),
    execmode_("execmode", id, this),
    inc_(1),
    stop_(false)
{
}


MatrixSelectVector::~MatrixSelectVector()
{
}


void
MatrixSelectVector::execute()
{
  update_state(NeedData);

  MatrixIPort *imat = (MatrixIPort *)get_iport("Matrix");
  if (!imat) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  MatrixHandle mh;
  if (!(imat->get(mh) && mh.get_rep()))
  {
    remark("Empty input matrix.");
    return;
  }
  
  MatrixOPort *ovec = (MatrixOPort *)get_oport("Vector");
  if (!ovec) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  MatrixOPort *osel = (MatrixOPort *)get_oport("Selected Index");
  if (!osel) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  update_state(JustStarted);
  
  const bool use_row = (row_or_col_.get() == "row");
  bool changed_p = false;
  PropertyManager *mh_prop = mh.get_rep();
  if (use_row)
  {
    string units;
    if (!mh_prop->get_property("row_units", units))
    {
      units = "Units";
    }
    if (units != selectable_units_.get())
    {
      selectable_units_.set(units);
      changed_p = true;
    }

    double minlabel;
    if (!mh_prop->get_property("row_min", minlabel))
    {
      minlabel = 0.0;
    }
    if (minlabel != selectable_min_.get())
    {
      selectable_min_.set(minlabel);
      changed_p = true;
    }

    double maxlabel;
    if (!mh_prop->get_property("row_max", maxlabel))
    {
      maxlabel = mh->nrows() - 1.0;
    }
    if (maxlabel != selectable_max_.get())
    {
      selectable_max_.set(maxlabel);
      changed_p = true;
    }

    int increments = mh->nrows();
    if (increments != selectable_inc_.get())
    {
      selectable_inc_.set(increments);
      changed_p = true;
    }
  }
  else
  {
    string units;
    if (!mh_prop->get_property("col_units", units))
    {
      units = "Units";
    }
    if (units != selectable_units_.get())
    {
      selectable_units_.set(units);
      changed_p = true;
    }

    double minlabel;
    if (!mh_prop->get_property("col_min", minlabel))
    {
      minlabel = 0.0;
    }
    if (minlabel != selectable_min_.get())
    {
      selectable_min_.set(minlabel);
      changed_p = true;
    }

    double maxlabel;
    if (!mh_prop->get_property("col_max", maxlabel))
    {
      maxlabel = mh->ncols() - 1.0;
    }
    if (maxlabel != selectable_max_.get())
    {
      selectable_max_.set(maxlabel);
      changed_p = true;
    }

    int increments = mh->ncols();
    if (increments != selectable_inc_.get())
    {
      selectable_inc_.set(increments);
      changed_p = true;
    }
  }

  if (changed_p)
  {
    std::ostringstream str;
    str << id << " update";
    TCL::execute(str.str().c_str());
  }
  
  reset_vars();

  if (execmode_.get() != "play" || execmode_.get() != "step") { return; }
  stop_ = false;

  const int start = range_min_.get();
  const int end = range_max_.get();
  int which = start;
  if (changed_p)
  {
    inc_ = (start>end)?-1:1;
  }
  if (execmode_.get() == "step")
  {
    int a = start;
    int b = end;
    if (a > b) {int tmp = a; a = b; b = tmp; }
    if (current_.get() >= a && current_.get() <= b)
    {
      which = current_.get();
    }
  }
  for (;!stop_; which += inc_, current_.set(which))
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

    // Attempt to copy no-transient properties.
    // TODO: update min/max to be the current value:  min + (max - min) * inc
    //PropertyManager *cmp = cm;
    //*cmp = *mh_prop;

    ColumnMatrix *selected = scinew ColumnMatrix(1);
    selected->put(0, 0, (double)which);

    if (which == end)
    {
      if (playmode_.get() == "bounce")
      {
	inc_ *= -1;
      }
      else
      {
	which = start - inc_;
      }
      if (playmode_.get() == "once")
      {
	stop_ = true;
      }
    }
    if (execmode_.get() == "step")
    {
      stop_ = true;
    }
    if (stop_)
    {
      ovec->send(MatrixHandle(cm));
      osel->send(MatrixHandle(selected));
    }
    else
    {
      ovec->send_intermediate(MatrixHandle(cm));
      osel->send_intermediate(MatrixHandle(selected));
    }
  }
}


void
MatrixSelectVector::tcl_command(TCLArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("MatrixSelectVector needs a minor command");
    return;
  }
  if (args[1] == "stop")
  {
    stop_ = true;
  }
  else Module::tcl_command(args, userdata);
}


} // End namespace SCIRun
