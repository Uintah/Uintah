/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

//    File   : NrrdSelectTime.cc
//    Author : Martin Cole
//    Date   : Wed Mar 26 15:20:49 2003

#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Thread/Time.h>
#include <iostream>
#include <sstream>

namespace SCITeem {

using namespace SCIRun;

class NrrdSelectTime : public Module {
public:
  NrrdSelectTime(GuiContext* ctx);
  virtual ~NrrdSelectTime();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

private:
  void send_selection(NrrdDataHandle, int, unsigned int, bool);
  int increment(int which, int lower, int upper);

  GuiDouble      selectable_min_;
  GuiDouble      selectable_max_;
  GuiInt         selectable_inc_;
  GuiInt         range_min_;
  GuiInt         range_max_;
  GuiString      playmode_;
  GuiInt         current_;
  GuiString      execmode_;
  GuiInt         delay_;
  GuiInt         inc_amount_;
  int            inc_;
  bool           loop_;
  int            last_input_;
  NrrdDataHandle last_output_;
};


DECLARE_MAKER(NrrdSelectTime)
  
NrrdSelectTime::NrrdSelectTime(GuiContext* ctx) : 
  Module("NrrdSelectTime", ctx, Iterator,"NrrdData", "Teem"),
  selectable_min_(get_ctx()->subVar("selectable_min")),
  selectable_max_(get_ctx()->subVar("selectable_max")),
  selectable_inc_(get_ctx()->subVar("selectable_inc")),
  range_min_(get_ctx()->subVar("range_min")),
  range_max_(get_ctx()->subVar("range_max")),
  playmode_(get_ctx()->subVar("playmode")),
  current_(get_ctx()->subVar("current")),
  execmode_(get_ctx()->subVar("execmode")),
  delay_(get_ctx()->subVar("delay")),
  inc_amount_(get_ctx()->subVar("inc-amount")),
  inc_(1),
  loop_(false),
  last_input_(-1),
  last_output_(0)
{
}


NrrdSelectTime::~NrrdSelectTime()
{
}


void
NrrdSelectTime::send_selection(NrrdDataHandle in, 
			       int which, unsigned int time_axis, bool cache)
{
  NrrdDataHandle onrrd_handle;

  if ((unsigned int)time_axis == in->nrrd_->dim - 1)
  {
    NrrdData *out = scinew NrrdData(in.get_rep());

    // Copy all of the nrrd header from in to out.
    if (nrrdBasicInfoCopy(out->nrrd_, in->nrrd_,
			  NRRD_BASIC_INFO_DATA_BIT
			  | NRRD_BASIC_INFO_CONTENT_BIT
			  | NRRD_BASIC_INFO_COMMENTS_BIT))
    {
      error(biffGetDone(NRRD));
      return;
    }
    out->nrrd_->dim--;

    if (nrrdAxisInfoCopy(out->nrrd_, in->nrrd_, NULL, NRRD_AXIS_INFO_NONE))
    {
      error(biffGetDone(NRRD));
      return;
    }
    if (nrrdContentSet_va(out->nrrd_, "slice", in->nrrd_, "%d,%d",
			  time_axis, which))
    {
      error(biffGetDone(NRRD));
      return;
    }  
    
    size_t offset = which * nrrdTypeSize[in->nrrd_->type];
    for (unsigned int i = 0; i < in->nrrd_->dim - 1; i++)
    {
      offset *= in->nrrd_->axis[i].size;
    }
    out->nrrd_->data = ((unsigned char *)(in->nrrd_->data)) + offset;
    onrrd_handle = out;
  }
  else
  {
    // Do the slice.
    NrrdData *out = scinew NrrdData();
    if (nrrdSlice(out->nrrd_, in->nrrd_, time_axis, which))
    {
      char *err = biffGetDone(NRRD);
      error(string("Trouble slicing: ") + err);
      free(err);
      return;
    }
    onrrd_handle = out;
  }

  // Copy the properties.
  onrrd_handle->copy_properties(in.get_rep());

  NrrdOPort *onrrd = (NrrdOPort *)get_oport("Time Slice");
  onrrd->set_cache( cache );
  onrrd->send_and_dereference(onrrd_handle);

  ColumnMatrix *selected = scinew ColumnMatrix(1);
  selected->put(0, 0, (double)which);
  MatrixHandle stmp(selected);
  send_output_handle("Selected Index", stmp);
}


int
NrrdSelectTime::increment(int which, int lower, int upper)
{
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
NrrdSelectTime::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("Time Axis nrrd", nrrd_handle)) return;

  update_state(JustStarted);

  int which;

  // Must have a time axis.
  int time_axis = -1;
  for (unsigned int i = 0; i < (unsigned int)nrrd_handle->nrrd_->dim; i++)
  {
    if (nrrd_handle->nrrd_->axis[i].label &&
	string(nrrd_handle->nrrd_->axis[i].label) == string("Time"))
    {
      time_axis = i;
      break;
    }
  }
  
  if (time_axis == -1) {
    warning("This nrrd has no time axis (Must be labeled 'Time')");
    warning("Using the last axis as the time axis.");
    time_axis = nrrd_handle->nrrd_->dim - 1;
  }

  if (nrrd_handle->generation != last_input_)
  {
    // set the slider values.
    selectable_min_.set(0.0);
    selectable_max_.set(nrrd_handle->nrrd_->axis[time_axis].size - 1);

    get_gui()->execute(get_id() + " update_range");
    last_input_ = nrrd_handle->generation;
  }
 
  reset_vars();

  // If there is a current index matrix, use it.
  MatrixHandle currentH;
  if (get_input_handle("Current Index", currentH, false))
  {
    which = (int) (currentH->get(0, 0));
    send_selection(nrrd_handle, which, time_axis, true);
  }
  else
  {
    // Cache var
    bool cache = (playmode_.get() != "inc_w_exec");

    // Get the current start and end.
    const int start = range_min_.get();
    const int end   = range_max_.get();

    int lower = start;
    int upper = end;
    if (lower > upper) {int tmp = lower; lower = upper; upper = tmp; }

    // Update the increment.
    if (playmode_.get() == "once" || playmode_.get() == "loop")
      inc_ = (start>end)?-1:1;

    // If the current value is invalid, reset it to the start.
    if (current_.get() < lower || current_.get() > upper) {
      current_.set(start);
      inc_ = (start>end)?-1:1;
    }

    // Cache execmode and reset it in case we bail out early.
    const string execmode = execmode_.get();

    which = current_.get();

    // If updating, we're done for now.
    if (execmode == "update") {

    } else if (execmode == "step") {
      which = increment(current_.get(), lower, upper);
      send_selection(nrrd_handle, which, time_axis, cache);

    } else if (execmode == "stepb") {
      inc_ *= -1;
      which = increment(current_.get(), lower, upper);
      inc_ *= -1;
      send_selection(nrrd_handle, which, time_axis, cache);
      
    } else if (execmode == "play") {

      if( !loop_ ) {
	if (playmode_.get() == "once" && which >= end)
	  which = start;
      }

      send_selection(nrrd_handle, which, time_axis, cache);

      // User may have changed the execmode to stop so recheck.
      execmode_.reset();
      if ( loop_ = (execmode_.get() == "play") ) {
	const int delay = delay_.get();
      
	if( delay > 0) {
	  Time::waitFor(delay/1000.0); // use this for cross platform instead of below
	  //const unsigned int secs = delay / 1000;
	  //const unsigned int msecs = delay % 1000;
	  //if (secs)  { sleep(secs); }
	  //if (msecs) { usleep(msecs * 1000); }
	}
    
	int next = increment(which, lower, upper);    

	// Incrementing may cause a stop in the execmode so recheck.
	execmode_.reset();
	if( loop_ = (execmode_.get() == "play") ) {
	  which = next;

	  want_to_execute();
	}
      }
    } else {
      if( execmode == "rewind" )
	which = start;

      else if( execmode == "fforward" )
	which = end;
    
      send_selection(nrrd_handle, which, time_axis, cache);
    
      if (playmode_.get() == "inc_w_exec") {
	which = increment(which, lower, upper);
      }
    }
  }

  current_.set(which);
}


void
NrrdSelectTime::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("NrrdSelectTime needs a minor command");
    return;
  }

  if (args[1] == "restart") {
  }
  else Module::tcl_command(args, userdata);
}

} // End namespace SCITeem
