
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
  GuiInt    delay_;
  int       inc_;
  bool      stop_;

  void send_selection(MatrixHandle mh, int which, bool use_row, bool last_p);
  int increment(int which, int lower, int upper);

public:
  MatrixSelectVector(GuiContext* ctx);
  virtual ~MatrixSelectVector();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(MatrixSelectVector)

MatrixSelectVector::MatrixSelectVector(GuiContext* ctx)
  : Module("MatrixSelectVector", ctx, Filter,"Math", "SCIRun"),
    row_or_col_(ctx->subVar("row_or_col")),
    selectable_min_(ctx->subVar("selectable_min")),
    selectable_max_(ctx->subVar("selectable_max")),
    selectable_inc_(ctx->subVar("selectable_inc")),
    selectable_units_(ctx->subVar("selectable_units")),
    range_min_(ctx->subVar("range_min")),
    range_max_(ctx->subVar("range_max")),
    playmode_(ctx->subVar("playmode")),
    current_(ctx->subVar("current")),
    execmode_(ctx->subVar("execmode")),
    delay_(ctx->subVar("delay")),
    inc_(1),
    stop_(false)
{
}


MatrixSelectVector::~MatrixSelectVector()
{
}


void
MatrixSelectVector::send_selection(MatrixHandle mh, int which,
				   bool use_row, bool last_p)
{
  MatrixOPort *ovec = (MatrixOPort *)get_oport("Vector");
  if (!ovec) {
    error("Unable to initialize oport 'Vector'.");
    return;
  }

  MatrixOPort *osel = (MatrixOPort *)get_oport("Selected Index");
  if (!osel) {
    error("Unable to initialize oport 'Selected Index'.");
    return;
  }

  current_.set(which);

  ColumnMatrix *cm;
  if (use_row)
  {
    if (which < 0 || which >= mh->nrows())
    {
      warning("Row out of range, skipping.");
      return;
    }
    cm = scinew ColumnMatrix(mh->ncols());
    double *data = cm->get_data();
    for (int c = 0; c<mh->ncols(); c++)
    {
      data[c] = mh->get(which, c);
    }
  }
  else
  {
    if (which < 0 || which >= mh->ncols())
    {
      warning("Column out of range, skipping.");
      return;
    }
    cm = scinew ColumnMatrix(mh->nrows());
    double *data = cm->get_data();
    for (int r = 0; r<mh->nrows(); r++)
    {
      data[r] = mh->get(r, which);
    }
  }	    

  ColumnMatrix *selected = scinew ColumnMatrix(1);
  selected->put(0, 0, (double)which);

  if (last_p)
  {
    ovec->send(MatrixHandle(cm));
    osel->send(MatrixHandle(selected));
  }
  else
  {
    osel->send(MatrixHandle(selected));
    ovec->send_intermediate(MatrixHandle(cm));
  }
}


int
MatrixSelectVector::increment(int which, int lower, int upper)
{
  // Do nothing if no range.
  if (upper == lower)
  {
    if (playmode_.get() == "once")
    {
      stop_ = true;
    }
    return upper;
  }
  which += inc_;

  if (which > upper)
  {
    if (playmode_.get() == "bounce1")
    {
      inc_ *= -1;
      return increment(upper, lower, upper);
    }
    else if (playmode_.get() == "bounce2")
    {
      inc_ *= -1;
      return upper;
    }
    else
    {
      if (playmode_.get() == "once")
      {
	stop_ = true;
      }
      return lower;
    }
  }
  if (which < lower)
  {
    if (playmode_.get() == "bounce1")
    {
      inc_ *= -1;
      return increment(lower, lower, upper);
    }
    else if (playmode_.get() == "bounce2")
    {
      inc_ *= -1;
      return lower;
    }
    else
    {
      if (playmode_.get() == "once")
      {
	stop_ = true;
      }
      return upper;
    }
  }
  return which;
}



void
MatrixSelectVector::execute()
{
  update_state(NeedData);

  MatrixIPort *imat = (MatrixIPort *)get_iport("Matrix");
  if (!imat) {
    error("Unable to initialize iport 'Matrix'.");
    return;
  }
  MatrixHandle mh;
  if (!(imat->get(mh) && mh.get_rep()))
  {
    remark("Empty input matrix.");
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
    gui->execute(str.str().c_str());
  }
  
  reset_vars();

#if 1
  // Specialized matrix multiply, with Weight Vector given as a sparse
  // matrix.  It's not clear what this has to do with MatrixSelectVector.
  MatrixIPort *ivec = (MatrixIPort *)get_iport("Weight Vector");
  if (!ivec) {
    error("Unable to initialize iport 'Weight Vector'.");
    return;
  }
  MatrixHandle weightsH;
  if (ivec->get(weightsH) && weightsH.get_rep())
  {
    MatrixOPort *ovec = (MatrixOPort *)get_oport("Vector");
    if (!ovec) {
      error("Unable to initialize oport 'Vector'.");
      return;
    }

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
    ovec->send(MatrixHandle(cm));
    return;
  }
#endif

  // If there is a current index matrix, use it.
  MatrixIPort *icur = (MatrixIPort *)get_iport("Current Index");
  if (!icur) {
    error("Unable to initialize iport 'Current Index'.");
    return;
  }
  MatrixHandle currentH;
  if (icur->get(currentH) && currentH.get_rep())
  {
    send_selection(mh, (int)(currentH->get(0, 0)), use_row, true);
    return;
  }

  // Update the increment.
  const int start = range_min_.get();
  const int end = range_max_.get();
  if (changed_p || playmode_.get() == "once" || playmode_.get() == "loop")
  {
    inc_ = (start>end)?-1:1;
  }

  // If the current value is invalid, reset it to the start.
  int lower = start;
  int upper = end;
  if (lower > upper) {int tmp = lower; lower = upper; upper = tmp; }
  if (current_.get() < lower || current_.get() > upper)
  {
    current_.set(start);
    inc_ = (start>end)?-1:1;
  }

  // Cash execmode and reset it in case we bail out early.
  const string execmode = execmode_.get();
  // If updating, we're done for now.
  if (execmode == "update")
  {
  }
  else if (execmode == "step")
  {
    int which = current_.get();

    // TODO: INCREMENT
    which = increment(which, lower, upper);
    send_selection(mh, which, use_row, true);
  }
  else if (execmode == "play")
  {
    stop_ = false;
    int which = current_.get();
    if (which >= end && playmode_.get() == "once")
    {
      which = start;
    }
    const int delay = delay_.get();
    int stop;
    do {
      int next;
      if (playmode_.get() == "once")
      {
	next = increment(which, lower, upper);
      }
      stop = stop_;
      send_selection(mh, which, use_row, stop);
      if (!stop && delay > 0)
      {
	const unsigned int secs = delay / 1000;
	const unsigned int msecs = delay % 1000;
	if (secs)  { sleep(secs); }
	if (msecs) { usleep(msecs * 1000); }
      }
      if (playmode_.get() == "once")
      {
	which = next;
      }
      else if (!stop)
      {
	which = increment(which, lower, upper);
      }
    } while (!stop);
  }
  else
  {
    send_selection(mh, current_.get(), use_row, true);
  }
  execmode_.set("init");
}


void
MatrixSelectVector::tcl_command(GuiArgs& args, void* userdata)
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
