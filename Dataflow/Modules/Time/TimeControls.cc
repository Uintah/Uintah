//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : TimeControls.cc
//    Author : Martin Cole
//    Date   : Thu Apr 14 12:43:53 2005

#include <Dataflow/Ports/TimePort.h>
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Runnable.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

namespace SCIRun {
using namespace std;

class RTTime;

class TimeControls : public Module {
public:
  TimeControls(GuiContext*);

  virtual ~TimeControls();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
  void set_started(double t);
  void set_time(double t);
  void inc_pause(double t);
  bool playing() { return playing_; }
  bool rewind() { return rewind_; }
  void done_rewind() { rewind_ = false; }
  bool fforward() { return fforward_; }
  void done_fforward() { fforward_ = false; }
  double time_sf() { return time_sf_.get(); }

private:
  TimeData                 time_;
  GuiString                exec_mode_;
  GuiDouble                time_sf_;
  RTTime                  *trun_;
  Thread                  *trun_thread_;
  bool                     playing_;
  bool                     rewind_;
  bool                     fforward_;
};

class RTTime : public Runnable {
public:
  RTTime(TimeControls* module) : 
    module_(module), 
    throttle_(), 
    dead_(0) 
  {};
  virtual ~RTTime();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
private:
  TimeControls            *module_;
  TimeThrottle	           throttle_;
  bool		           dead_;
};

RTTime::~RTTime()
{
}

void
RTTime::run()
{
  throttle_.start();
  const double inc = 1./33.;
  double t = throttle_.time();
  module_->set_started(t);
  double tlast = t;
  while (!dead_) {

    if (module_->rewind()) {
      bool playing = throttle_.current_state() != Timer::Stopped;
      if (playing) throttle_.stop();
      throttle_.clear();
      if (playing) throttle_.start();
      tlast = 0;
      module_->done_rewind();
    }
    if (module_->fforward()) {
      // fforward increments the time by 10 mins. + 
      bool playing = throttle_.current_state() != Timer::Stopped;
      if (!playing) throttle_.start();
      throttle_.add(600.);
      if (!playing) throttle_.stop();
      tlast += 600.;
      module_->done_fforward();
    }

    t = throttle_.time();
    double newt = (module_->time_sf() * (t - tlast)) + tlast;
    throttle_.add(newt - t);

    t = throttle_.time();
    if (module_->playing()) {
      if (throttle_.current_state() == Timer::Stopped) {
	throttle_.start();
      }
    } else {
      // stop the throttle.
      if (throttle_.current_state() != Timer::Stopped) {
	throttle_.stop();
      }
    }

    module_->set_time(t);
    throttle_.wait_for_time(t + inc);
    tlast = t;
  }
}


DECLARE_MAKER(TimeControls)
TimeControls::TimeControls(GuiContext* ctx) : 
  Module("TimeControls", ctx, Source, "Time", "SCIRun"),
  exec_mode_(ctx->subVar("execmode")),
  time_sf_(ctx->subVar("scale_factor")),
  playing_(false),
  rewind_(false),
  fforward_(false)
{
  trun_ = scinew RTTime(this);
  trun_thread_ = scinew Thread(trun_, 
			       string(id+" time global sync").c_str());

}

TimeControls::~TimeControls()
{
  if (trun_thread_) {
    trun_->set_dead(true);
    trun_thread_->join();
    trun_thread_ = 0;
  }
}

void
TimeControls::execute()
{
  TimeOPort *oport;
  if (!(oport = static_cast<TimeOPort *>(get_oport("time"))))
    {
      error("Could not find time output port");
      return;
    }
  TimeViewerHandle tvh(&time_);
  oport->send(tvh);
}

void 
TimeControls::set_started(double t)
{
  time_.set_started(t);
  ostringstream str;
  str << id << " do_update " << Round(time_.view_started());
  gui->execute(str.str());
}

void 
TimeControls::inc_pause(double t)
{
  time_.set_started(time_.view_started() + t);
}

void 
TimeControls::set_time(double t)
{
  time_.set_now(t);
  ostringstream str;
  str << id << " do_update " << Round(time_.view_elapsed_since_start());
  gui->execute(str.str());
  gui->execute("update idletasks");
}


void
TimeControls::tcl_command(GuiArgs& args, void* userdata)
{
  if (args[1] == "set_time") {
    std::cerr << "set_time" << std::endl;
  } else if (args[1] == "update_scale") {
    time_sf_.reset();
  } else if (args[1] == "fforward") {
    fforward_ = true;
  } else if (args[1] == "rewind") {
    rewind_ = true;
  } else if (args[1] == "play") {
    playing_ = true;
    want_to_execute();
  } else if (args[1] == "pause") {
    playing_ = false;
  } else {
    Module::tcl_command(args, userdata);
  }
}

}


