/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


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
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>
#ifdef _WIN32
#include <Windows.h>
#endif

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
  GuiString dependence_;
  GuiInt    current_;
  GuiString execmode_;
  GuiInt    delay_;
  GuiInt    inc_amount_;
  GuiInt    send_amount_;
  GuiInt    data_series_done_;
  int       inc_;
  bool      loop_;
  int       use_row_;
  int       last_gen_;

  void send_selection(MatrixHandle mh, int which, int ncopy, bool cache);
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
    dependence_(ctx->subVar("dependence")),
    current_(ctx->subVar("current")),
    execmode_(ctx->subVar("execmode")),
    delay_(ctx->subVar("delay")),
    inc_amount_(ctx->subVar("inc-amount")),
    send_amount_(ctx->subVar("send-amount")),
    data_series_done_(ctx->subVar("data_series_done")),
    inc_(1),
    loop_(false),
    use_row_(-1),
    last_gen_(-1)
{
}


MatrixSelectVector::~MatrixSelectVector()
{
}


void
MatrixSelectVector::send_selection(MatrixHandle mh, int which,
				   int ncopy, bool cache)
{
  MatrixOPort *ovec = (MatrixOPort *)get_oport("Vector");
  MatrixOPort *osel = (MatrixOPort *)get_oport("Selected Index");

  MatrixHandle matrix(0);
  if (use_row_) {
    if (ncopy == 1) {
      ColumnMatrix *cm = scinew ColumnMatrix(mh->ncols());
      double *data = cm->get_data();
      for (int c = 0; c<mh->ncols(); c++)
      {
	data[c] = mh->get(which, c);
      }
      matrix = cm;
    } else {
      DenseMatrix *dm = scinew DenseMatrix(ncopy, mh->ncols());
      for (int i = 0; i < ncopy; i++)
	for (int c = 0; c < mh->ncols(); c++)
	  dm->put(i, c, mh->get(which + i, c));

      matrix = dm;
    }
  } else {
    if (ncopy == 1) {
      ColumnMatrix *cm = scinew ColumnMatrix(mh->nrows());
      double *data = cm->get_data();
      for (int r = 0; r<mh->nrows(); r++)
	data[r] = mh->get(r, which);
      matrix = cm;
    } else {
      DenseMatrix *dm = scinew DenseMatrix(mh->nrows(), ncopy);
      for (int r = 0; r < mh->nrows(); r++)
	for (int i = 0; i < ncopy; i++)
	  dm->put(r, i, mh->get(r, which + i));

      matrix = dm;
    }
  }	    

  ovec->set_cache( cache );
  ovec->send(matrix);


  ColumnMatrix *selected = scinew ColumnMatrix(1);
  selected->put(0, 0, (double)which);

  osel->send(MatrixHandle(selected));
}


int
MatrixSelectVector::increment(int which, int lower, int upper)
{
  data_series_done_.reset();
  if (playmode_.get() == "autoplay" && data_series_done_.get()) {
    return which;
  }
  // Do nothing if no range.
  if (upper == lower) {
    if (playmode_.get() == "once")
      execmode_.set( "stop" );
    return upper;
  }
  const int inc_amount = Max(1, Min(upper, inc_amount_.get()));

  which += inc_ * inc_amount;

  if (which > upper) {
    if (playmode_.get() == "bounce1") {
      inc_ *= -1;
      return increment(upper, lower, upper);
    } else if (playmode_.get() == "bounce2") {
      inc_ *= -1;
      return upper;
    } else if (playmode_.get() == "autoplay") {
      data_series_done_.set(1);
      return lower;
    } else {
      if (playmode_.get() == "once")
	execmode_.set( "stop" );
      return lower;
    }
  }
  if (which < lower) {
    if (playmode_.get() == "bounce1") {
      inc_ *= -1;
      return increment(lower, lower, upper);
    } else if (playmode_.get() == "bounce2") {
      inc_ *= -1;
      return lower;
    } else {
      if (playmode_.get() == "once")
	execmode_.set( "stop" );
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
  MatrixHandle mh;
  if (!(imat->get(mh) && mh.get_rep()))
  {
    error("Empty input matrix.");
    return;
  }
  update_state(JustStarted);
  if (playmode_.get() == "autoplay") {
    data_series_done_.reset();
    while (last_gen_ == mh->generation && data_series_done_.get()) {
      //cerr << "waiting" << std::endl;
      //want_to_execute();
      return;
    } 
    last_gen_ = mh->generation;
    data_series_done_.set(0);
  }

  
  bool changed_p = false;

  bool use_row = (row_or_col_.get() == "row");

  if( use_row_ != use_row ) {
    use_row_ = use_row;
    changed_p = true;
  }

  if (use_row_){
    string units("");
    //    if (!mh->get_property("row_units", units))
    //{
    //  units = "Units";
    //}
    if (units != selectable_units_.get()) {
      selectable_units_.set(units);
      changed_p = true;
    }

    double minlabel = 0.0;
    if (!mh->get_property("row_min", minlabel))
      minlabel = 0.0;

    if (minlabel != selectable_min_.get()) {
      selectable_min_.set(minlabel);
      changed_p = true;
    }

    double maxlabel = 0.0;
    if (!mh->get_property("row_max", maxlabel))
      maxlabel = mh->nrows() - 1.0;

    if (maxlabel != selectable_max_.get()) {
      selectable_max_.set(maxlabel);
      changed_p = true;
    }

    int increments = mh->nrows();
    if (increments != selectable_inc_.get()) {
      selectable_inc_.set(increments);
      changed_p = true;
    }
  } else {
    string units("");;
    //    if (!mh->get_property("col_units", units))
    //{
    //  units = "Units";
    // }
    if (units != selectable_units_.get()) {
      selectable_units_.set(units);
      changed_p = true;
    }

    double minlabel = 0.0;
    if (!mh->get_property("col_min", minlabel))
      minlabel = 0.0;

    if (minlabel != selectable_min_.get()) {
      selectable_min_.set(minlabel);
      changed_p = true;
    }

    double maxlabel = 0.0;
    if (!mh->get_property("col_max", maxlabel))
      maxlabel = mh->ncols() - 1.0;

    if (maxlabel != selectable_max_.get()) {
      selectable_max_.set(maxlabel);
      changed_p = true;
    }

    int increments = mh->ncols();
    if (increments != selectable_inc_.get()) {
      selectable_inc_.set(increments);
      changed_p = true;
    }
  }

  if (changed_p)
    gui->execute(id + " update_range");
  
  reset_vars();

  int which;

  // Specialized matrix multiply, with Weight Vector given as a sparse
  // matrix.  It's not clear what this has to do with MatrixSelectVector.
  MatrixIPort *ivec = (MatrixIPort *)get_iport("Weight Vector");
  MatrixHandle weightsH;
  if (ivec->get(weightsH) && weightsH.get_rep()) {
    MatrixOPort *ovec = (MatrixOPort *)get_oport("Vector");

    ColumnMatrix *w = dynamic_cast<ColumnMatrix*>(weightsH.get_rep());
    if (w == 0)  {
      error("Weight Vector must be a column matrix.");
      return;
    }
    ColumnMatrix *cm;
    if (use_row_) {
      cm = scinew ColumnMatrix(mh->ncols());
      cm->zero();
      double *data = cm->get_data();
      for (int i = 0; i<w->nrows()/2; i++) {
	const int idx = (int)((*w)[i*2]);
	double wt = (*w)[i*2+1];
	for (int j = 0; j<mh->ncols(); j++)
	  data[j]+=mh->get(idx, j)*wt;
      }
    }
    else
    {
      cm = scinew ColumnMatrix(mh->nrows());
      cm->zero();
      double *data = cm->get_data();
      for (int i = 0; i<w->nrows()/2; i++) {
	const int idx = (int)((*w)[i*2]);
	double wt = (*w)[i*2+1];
	for (int j = 0; j<mh->nrows(); j++)
	  data[j]+=mh->get(j, idx)*wt;
      }
    }
    ovec->send(MatrixHandle(cm));
    return;
  }

  const int maxsize = (use_row_?mh->nrows():mh->ncols())-1;
  const int send_amount = Max(1, Min(maxsize, send_amount_.get()));

  // If there is a current index matrix, use it.
  MatrixIPort *icur = (MatrixIPort *)get_iport("Current Index");
  MatrixHandle currentH;
  if (icur->get(currentH) && currentH.get_rep()) {
    which = (int)(currentH->get(0, 0));
    send_selection(mh, which, send_amount, true);
  } else {

    // Cache var
    bool cache = (playmode_.get() != "inc_w_exec");

    // Get the current start and end.
    const int start = range_min_.get();
    const int end = range_max_.get();

    int lower = start;
    int upper = end;
    if (lower > upper) {int tmp = lower; lower = upper; upper = tmp; }


    // Update the increment.
    if (changed_p || playmode_.get() == "once" || 
	playmode_.get() == "autoplay" || playmode_.get() == "loop") {
      inc_ = (start>end)?-1:1;
    }

    // If the current value is invalid, reset it to the start.
    if (current_.get() < lower || current_.get() > upper) {
      current_.set(start);
      inc_ = (start>end)?-1:1;
    }

    // Cash execmode and reset it in case we bail out early.
    const string execmode = execmode_.get();

    which = current_.get();

    // If updating, we're done for now.
    if (execmode == "update") {
    
    } else if (execmode == "step") {
      which = increment(current_.get(), lower, upper);
      send_selection(mh, which, send_amount, cache);

    } else if (execmode == "stepb") {
      inc_ *= -1;
      which = increment(current_.get(), lower, upper);
      inc_ *= -1;
      send_selection(mh, which, send_amount, cache);

    } else if (execmode == "play") {
      if( !loop_ ) {
	if (playmode_.get() == "once" && which >= end)
	  which = start;
	if (playmode_.get() == "autoplay" && which >= end)
	{
	  which = start;
	  cerr << "setting to wait" << std::endl;
	  data_series_done_.set(1);
	}
    }

      send_selection(mh, which, send_amount, cache);

      // User may have changed the execmode to stop so recheck.
      execmode_.reset();
      if ( loop_ = (execmode_.get() == "play") ) {
	const int delay = delay_.get();
      
	if( delay > 0) {
#ifndef _WIN32
	  const unsigned int secs = delay / 1000;
	  const unsigned int msecs = delay % 1000;
	  if (secs)  { sleep(secs); }
	  if (msecs) { usleep(msecs * 1000); }
#else 
	  Sleep(delay);
#endif
	}
    
	int next = increment(which, lower, upper);    

	// Incrementing may cause a stop in the execmode so recheck.
	execmode_.reset();
	if(loop_ = (execmode_.get() == "play")) {
	  which = next;
	  want_to_execute();
	}
      }
    } else {
      if( execmode == "rewind" )
	which = start;

      else if( execmode == "fforward" )
	which = end;
    
      send_selection(mh, which, send_amount, cache);
    
      if (playmode_.get() == "inc_w_exec") {
	which = increment(which, lower, upper);
      }
    }
  }
  current_.set(which);
}


void
MatrixSelectVector::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("MatrixSelectVector needs a minor command");
    return;

  }

  if (args[1] == "restart") {

  } else Module::tcl_command(args, userdata);
}


} // End namespace SCIRun
