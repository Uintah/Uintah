//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : NrrdSelectTime.cc
//    Author : Martin Cole
//    Date   : Wed Mar 26 15:20:49 2003

#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>
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
  bool           stop_;
  int            last_input_;
  NrrdDataHandle last_output_;

};


DECLARE_MAKER(NrrdSelectTime)
  
NrrdSelectTime::NrrdSelectTime(GuiContext* ctx) : 
  Module("NrrdSelectTime", ctx, Iterator,"NrrdData", "Teem"),
  selectable_min_(ctx->subVar("selectable_min")),
  selectable_max_(ctx->subVar("selectable_max")),
  selectable_inc_(ctx->subVar("selectable_inc")),
  range_min_(ctx->subVar("range_min")),
  range_max_(ctx->subVar("range_max")),
  playmode_(ctx->subVar("playmode")),
  current_(ctx->subVar("current")),
  execmode_(ctx->subVar("execmode")),
  delay_(ctx->subVar("delay")),
  inc_amount_(ctx->subVar("inc-amount")),
  inc_(1),
  stop_(false),
  last_input_(-1),
  last_output_(0)
{
}


NrrdSelectTime::~NrrdSelectTime()
{
}


void
NrrdSelectTime::send_selection(NrrdDataHandle nrrd_handle, 
			       int which, unsigned int time_axis, bool last_p)
{
  NrrdOPort *onrrd = (NrrdOPort *)get_oport("Time Slice");
  onrrd->set_dont_cache();
  if (!onrrd) {
    error("Unable to initialize oport 'Time Slice'.");
    return;
  }

  current_.set(which);

  NrrdDataHandle onrrd_handle(0);

  if (which < 0 || which > nrrd_handle->nrrd->axis[time_axis].size - 1)
  {
    warning("Row out of range, skipping.");
    return;
  }

  // do the slice
  NrrdData *out = scinew NrrdData();
  out->nrrd = nrrdNew();
  if (nrrdSlice(out->nrrd, nrrd_handle->nrrd, time_axis, which)) {
    char *err = biffGetDone(NRRD);
    error(string("Trouble slicing: ") + err);
    free(err);
    return;
  }
  onrrd_handle = out;
  onrrd_handle->copy_sci_data(*(nrrd_handle.get_rep()));

  if (last_p) {
    onrrd->send(onrrd_handle);
  } else {
    onrrd->send_intermediate(onrrd_handle);
  }
}


int
NrrdSelectTime::increment(int which, int lower, int upper)
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
  const int inc_amount = Max(1, Min(upper, inc_amount_.get()));
  which += inc_ * inc_amount;

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
NrrdSelectTime::execute()
{
  update_state(NeedData);

  NrrdIPort *inrrd = (NrrdIPort *)get_iport("Time Axis nrrd");
  if (!inrrd) {
    error("Unable to initialize iport 'Time Axis nrrd'.");
    return;
  }
  NrrdDataHandle nrrd_handle;
  if (!(inrrd->get(nrrd_handle) && nrrd_handle.get_rep()))
  {
    error("Empty input NrrdData.");
    return;
  }

  update_state(JustStarted);


  // Must have a time axis.
  // axis 0 is always tuple axis, so time cannot be axis 0.
  unsigned int time_axis = 0;

  for (unsigned int i = 1; i < (unsigned int)nrrd_handle->nrrd->dim; i++) {
    string al(nrrd_handle->nrrd->axis[i].label);
    const string t("Time");
    if (al == t) {
      time_axis = i;
      break;
    }
  }
  
  if (!time_axis) {
    error("This nrrd has no time axis (Must be labeled 'Time')");
    return;
  }

  if (nrrd_handle->generation != last_input_)
  {
    // set the slider values.
    selectable_min_.set(0.0);
    selectable_max_.set(nrrd_handle->nrrd->axis[time_axis].size - 1);

    std::ostringstream str;
    str << id << " update";
    gui->execute(str.str().c_str());
    last_input_ = nrrd_handle->generation;
  }
 
  reset_vars();


  // Update the increment.
  const int start = range_min_.get();
  const int end = range_max_.get();
  if (playmode_.get() == "once" || playmode_.get() == "loop")
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
    send_selection(nrrd_handle, which, time_axis, true);
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
      send_selection(nrrd_handle, which, time_axis, stop);
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
    if (playmode_.get() == "inc_w_exec")
    {
      int which = current_.get();

      send_selection(nrrd_handle, which, time_axis, true);
      which = increment(which, lower, upper);
      current_.set(which);
    } else {
      send_selection(nrrd_handle, current_.get(), time_axis, true);
    }
  }
  execmode_.set("init");
}


void
NrrdSelectTime::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("NrrdSelectTime needs a minor command");
    return;
  }
  if (args[1] == "stop")
  {
    stop_ = true;
  }
  else Module::tcl_command(args, userdata);
}


} // End namespace SCITeem
