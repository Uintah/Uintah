//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : Signals.cc
//    Author : McKay Davis
//    Date   : Thu Jun 29 18:14:47 2006


#include <Core/Skinner/Signals.h>
#include <Core/Skinner/Variables.h>
#include <Core/Util/Environment.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace SCIRun {
  namespace Skinner {
    Persistent *make_Signal() { return new Signal("",0); }
    PersistentTypeID Signal::type_id("Signal", "BaseEvent", &make_Signal);

    Signal::Signal(const string &name, SignalThrower *thrower, 
                   const string &data) :
      BaseEvent("", 0),
      signal_name_(name),
      signal_data_(data),
      thrower_(thrower)      
    {
    }

    Signal::~Signal() {}

    void
    Signal::io(Piostream &) {} 


    
    SignalCatcher::SignalCatcher() :
      catcher_targets_()
    {
    }

    SignalCatcher::~SignalCatcher() {  } 


    SignalCatcher::CatcherTargetInfo_t::CatcherTargetInfo_t() :
      catcher_(0),
      function_(0),
      data_(""),
      targetname_(""),
      threaded_(false)
    {
    }
      


    void
    SignalCatcher::register_target(const string &targetname,
                                   CatcherFunctionPtr function) 
    {
      ASSERT(function);
      CatcherTargetInfo_t callback;
      callback.catcher_ = this;
      callback.function_ = function;
      callback.data_ = "";
      callback.targetname_ = targetname;

      catcher_targets_.push_back(callback);
    }


    SignalThrower::SignalThrower() :
      all_catchers_()
    {
    }

    SignalThrower::~SignalThrower() {
    }
      

    event_handle_t
    SignalThrower::throw_signal(const string &signalname, Variables *vars) {
      event_handle_t signal = new Signal(signalname, this);
      return throw_signal(all_catchers_, signal, vars);
    }


    class ThreadedCallback : public Runnable {
    public:     
      ThreadedCallback() : callback_(0), signal_(0), state_(BaseTool::STOP_E){}
      virtual ~ThreadedCallback() { cerr << "threaded callback done\n"; }

      void      set_callback(SignalCatcher::CatcherTargetInfo_t *callback) {
        callback_ = callback;
      }

      void      set_signal(const event_handle_t &signal) {
        signal_ = signal;
      }
      
      event_handle_t get_signal() {
        return signal_;
      }

      virtual void run() {
        ASSERT(callback_);
        state_ = (callback_->catcher_->*callback_->function_)(signal_);
        signal_ = 0;
      }

      BaseTool::propagation_state_e     get_state() { return state_; }
    private:
      SignalCatcher::CatcherTargetInfo_t *      callback_;
      event_handle_t                            signal_;
      BaseTool::propagation_state_e             state_;
    };
      
      
      


    event_handle_t
    SignalThrower::throw_signal(SignalToAllCatchers_t &catchers,
                                event_handle_t &signalh,
                                Variables *vars)
    {

      Signal *signal = dynamic_cast<Signal *>(signalh.get_rep());
      ASSERT(signal);
      const string &signalname = signal->get_signal_name();
      //      cerr << "Throw signal2: " <<  signalname << std::endl;

      if (catchers.find(signalname) == catchers.end()) return signalh;
      BaseTool::propagation_state_e state = BaseTool::CONTINUE_E;
      SignalToAllCatchers_t::iterator iter = catchers.find(signalname);
      if (iter != catchers.end()) {
        AllSignalCatchers_t::iterator riter = iter->second.begin();
        AllSignalCatchers_t::iterator rend = iter->second.end();
        for(;riter != rend; ++riter) {
          SignalCatcher::CatcherTargetInfo_t &callback = *riter;

          ASSERT(callback.catcher_);
          ASSERT(callback.function_);

          string signaldata = callback.data_;
          if (vars) {
            signaldata = vars->dereference(signaldata);
          }
          signal->set_signal_data(signaldata);
          
          if (!callback.threaded_) {
            state = (callback.catcher_->*callback.function_)(signal);
          } else {
            cerr << "threading: " << signalname << std::endl;
            ThreadedCallback *threaded_callback = new ThreadedCallback();
            threaded_callback->set_callback(&callback);
            threaded_callback->set_signal(signal);
            new Thread(threaded_callback, signalname.c_str());
          }
          if (state == BaseTool::STOP_E) break;
        }
      }
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
          SignalCatcher::CatcherTargetInfo_t &callback = *nriter;
          ASSERT(callback.catcher_);
          ASSERT(callback.function_);
          allcatchers[callback.targetname_].push_back(callback);
        }
      }
      return allcatchers;
    }
               
  }
}
    
