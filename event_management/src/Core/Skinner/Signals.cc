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
  
    SignalCatcher::CatcherFunctionPtr
    SignalCatcher::get_catcher(const string &name)
    {
      if (catcher_targets_.find(name) != catcher_targets_.end()) {
        return catcher_targets_[name];
      }
      return 0;
    }


    SignalThrower::SignalThrower() :
      catchers_()
    {
    }

    SignalThrower::~SignalThrower() {
    }
      
    bool
    SignalThrower::hookup_signal_to_catcher_target
    (const string &signalname, const string &signaldata, 
     SignalCatcher *catcher, SignalCatcher::CatcherFunctionPtr func)
    {
      if (get_signal_id(signalname)) {
        catchers_[signalname].first.push_front(make_pair(catcher, func));
        catchers_[signalname].second = signaldata;
        return true;
      } 
      return false;
    }

    event_handle_t
    SignalThrower::throw_signal(SignalCatchers_t &catchers,
                                event_handle_t &signalh,
                                Variables *vars) {

      Signal *signal = dynamic_cast<Signal *>(signalh.get_rep());
      ASSERT(signal);
      BaseTool::propagation_state_e state;
      const string &signalname = signal->get_signal_name();
      //      cerr << "Throw signal: " <<  signalname << std::endl;
      if (catchers.find(signalname) == catchers.end()) return signalh;
      
      CatchersOnDeck_t &signal_catchers = catchers[signalname].first;
      for(CatchersOnDeck_t::iterator catcher_iter = 
            signal_catchers.begin();
          catcher_iter != signal_catchers.end(); ++catcher_iter) {

        string signaldata = catchers[signalname].second;
        if (vars) {
          signaldata = vars->dereference(signaldata);
        }
        signal->set_signal_data(signaldata);

        SignalCatcher *instance = catcher_iter->first;
        SignalCatcher::CatcherFunctionPtr catcher = catcher_iter->second;
        state = (instance->*catcher)(signal);
        if (state == BaseTool::STOP_E) break;        
      }
      return signal;
    }
      



    event_handle_t
    SignalThrower::throw_signal(const string &signalname, Variables *vars) {
      event_handle_t signal = new Signal(signalname, this);
      return throw_signal(catchers_, signal, vars);
    }


    SignalCatcher::TargetIDs_t
    SignalCatcher::get_all_target_ids() {
      TargetIDs_t ids;
      bool print = sci_getenv_p("SKINNER_XMLIO_DEBUG");
      if (print)
        cerr << "_ids: ";
      for(CatcherTargets_t::iterator titer = catcher_targets_.begin();
          titer != catcher_targets_.end(); ++titer) {
        if (print) 
          cerr << titer->first << ", ";
        ids.push_back(titer->first);
      }
      if (print)
        cerr << std::endl;
      return ids;
    }


  }
}
    
