//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : Signals.cc
//    Author : McKay Davis
//    Date   : Thu Jun 29 18:14:47 2006


#include <Core/Skinner/Signals.h>
#include <Core/Skinner/Variables.h>
#include <Core/Util/Environment.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Skinner/Drawable.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace SCIRun {
  namespace Skinner {
    Persistent *make_Signal() { return new Signal("",0,0); }
    PersistentTypeID Signal::type_id("Signal", "BaseEvent", &make_Signal);

    Signal::Signal(const string &name, 
                   SignalThrower *thrower, 
                   Variables *vars) :
      BaseEvent("", 0),
      signal_name_(name),
      variables_(vars),
      thrower_(thrower),
      result_(BaseTool::STOP_E)
      
    {
    }


    Signal::Signal(const Signal &copy) :
      BaseEvent("", 0),
      signal_name_(copy.signal_name_),
      variables_(copy.variables_),
      thrower_(copy.thrower_),
      result_(copy.result_)
    {
    }

    Signal::~Signal() {}

    void
    Signal::io(Piostream &) {} 


    
    SignalCatcher::SignalCatcher() :
      catcher_targets_()
    {
    }

    SignalCatcher::~SignalCatcher() 
    {  
      for (unsigned i = 0; i < catcher_targets_.size(); i++)
        delete catcher_targets_[i];
    } 

    SignalCatcher::CatcherTargetInfoBase::~CatcherTargetInfoBase() {}
      

    SignalThrower::SignalThrower() :
      all_catchers_()
    {
    }

    SignalThrower::~SignalThrower() {
    }
      
    void
    SignalThrower::register_default_thrower(SignalCatcher::CatcherTargetInfoBase* callback)
    {
      all_catchers_[callback->targetname_].push_back(callback);
    }
    


    event_handle_t
    SignalThrower::throw_signal(const string &signalname, Variables *vars) {
      event_handle_t signal = new Signal(signalname, this, vars);
      return throw_signal(all_catchers_, signal);
    }

    event_handle_t
    SignalThrower::throw_signal(event_handle_t &signal) {
      return throw_signal(all_catchers_, signal);
    }
    
    class ThreadedCallback : public Runnable {
    public:     
      ThreadedCallback(SignalCatcher::CatcherTargetInfoBase *callback,
                       const event_handle_t &signal) : 
        callback_(callback), signal_(0), state_(BaseTool::STOP_E)
      {
        signal_ = signal;
        signal_.detach();
      }

      virtual ~ThreadedCallback() { signal_ = 0; }
      
      virtual void run() {
        ASSERT(callback_);
        state_ = callback_->doCallback(signal_);
      }
      
      BaseTool::propagation_state_e     get_state() { return state_; }
    private:
      SignalCatcher::CatcherTargetInfoBase *    callback_;
      event_handle_t                            signal_;
      BaseTool::propagation_state_e             state_;
    };
    
      
      


    event_handle_t
    SignalThrower::throw_signal(SignalToAllCatchers_t &catchers,
                                event_handle_t &signalh)
    {
      Signal *signal = dynamic_cast<Signal *>(signalh.get_rep());
      ASSERT(signal);
      const string &signalname = signal->get_signal_name();
      signal->set_signal_result(BaseTool::CONTINUE_E);
      if (catchers.find(signalname) == catchers.end()) return signalh;
      BaseTool::propagation_state_e state = BaseTool::CONTINUE_E;
      SignalToAllCatchers_t::iterator iter = catchers.find(signalname);
      if (iter != catchers.end()) {
        AllSignalCatchers_t::iterator riter = iter->second.begin();
        AllSignalCatchers_t::iterator rend = iter->second.end();
        for(;riter != rend; ++riter) {
          SignalCatcher::CatcherTargetInfoBase* callback = *riter;

          //ASSERT(callback->catcher_);
          //ASSERT(callback->function_);
          signal->set_vars(callback->variables_);
          bool threaded = 
            callback->variables_ && callback->variables_->get_bool("threaded");

          if (!threaded) {
            state = callback->doCallback(signal);
          } else {
            ThreadedCallback *thread = new ThreadedCallback(callback,signal);
            (new Thread(thread, signalname.c_str()))->detach();
          }
          if (state == BaseTool::STOP_E) break;
        }
      }
      signal->set_signal_result(state);
      return signalh;
    }



    SignalThrower::SignalToAllCatchers_t
    SignalThrower::collapse_tree(SignalCatcher::TreeOfCatchers_t &tree) {
      SignalToAllCatchers_t allcatchers;
      SignalCatcher::TreeOfCatchers_t::reverse_iterator triter = tree.rbegin();
      SignalCatcher::TreeOfCatchers_t::reverse_iterator trend = tree.rend();
      for (;triter != trend; ++triter) {
        SignalCatcher::NodeCatchers_t &node_catchers = *triter;
        SignalCatcher::NodeCatchers_t::iterator nriter = node_catchers.begin();
        SignalCatcher::NodeCatchers_t::iterator nrend = node_catchers.end();
        for (;nriter != nrend; ++nriter) {
          SignalCatcher::CatcherTargetInfoBase* callback = *nriter;
          //ASSERT(callback.catcher_);
          //ASSERT(callback.function_);
          allcatchers[callback->targetname_].push_back(callback);
        }
      }
      return allcatchers;
    }
               
    SignalCallback::SignalCallback(Drawable *thrower,
                                   const string &signalname)
      : signal_(new Signal(signalname, thrower, thrower->get_vars())),
        signalh_(signal_),
        catchers_(),
        init_(false)
    {
    }

    SignalCallback::SignalCallback(Signal *signal)
      : signal_(signal),
        signalh_(signal),
        catchers_(),
        init_(false)
    {
    }

    SignalCallback::~SignalCallback()
    {
    }

    event_handle_t
    SignalCallback::operator()() {

      if (!init_) {
        SignalThrower::SignalToAllCatchers_t::iterator aciter = 
          signal_->get_signal_thrower()->all_catchers_.find
            (signal_->get_signal_name());
        
        if (aciter != signal_->get_signal_thrower()->all_catchers_.end()) {
          catchers_ = aciter->second;
        }
        init_ = true;
      }

      SignalThrower::AllSignalCatchers_t::iterator iter = catchers_.begin();
      SignalThrower::AllSignalCatchers_t::iterator iend = catchers_.end();
      BaseTool::propagation_state_e state = BaseTool::CONTINUE_E;
      for(;iter != iend; ++iter) {
        SignalCatcher::CatcherTargetInfoBase* callback = *iter;
        signal_->set_vars(callback->variables_);
        bool threaded = 
          callback->variables_ && callback->variables_->get_bool("threaded");

        if (!threaded) {
          state = callback->doCallback(signal_);
        } else {
          ThreadedCallback *thread = new ThreadedCallback(callback, signalh_);
          (new Thread(thread, signal_->get_signal_name().c_str()))->detach();
        }
        if (state == BaseTool::STOP_E) break;
      }

      signal_->set_signal_result(state);
      return signalh_;
    }      
  }
}
    
