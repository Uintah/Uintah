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
//    File   : Signals.h
//    Author : McKay Davis
//    Date   : Thu Jun 29 18:13:09 2006

#ifndef SKINNER_SIGNALS_H
#define SKINNER_SIGNALS_H

#include <Core/Events/BaseEvent.h>
#include <Core/Events/Tools/BaseTool.h>
#include <string>
#include <deque>
#include <vector>
using std::string;
using std::deque;
using std::vector;


#define REGISTER_CATCHER_TARGET(catcher_function) \
  catcher_functions_[#catcher_function] = \
    static_cast<SCIRun::Skinner::SignalCatcher::CatcherFunctionPtr> \
      (&catcher_function);


namespace SCIRun {
  namespace Skinner {

    class Signal; 
    class SignalThrower;
    class SignalCatcher;

    class Signal : public SCIRun::BaseEvent
    {
    public:
      Signal(const string &name, SignalThrower *);//const string &target);
      ~Signal();
      string              get_signal_name() { return signal_name_; }
      void                set_signal_name(const string &n) { signal_name_=n; }

      SignalThrower *     get_signal_thrower() { return thrower_; }
      void                set_signal_thrower(SignalThrower *t) { thrower_=t; }
      
      virtual Signal *    clone() { return new Signal(*this); }
      virtual void        io(Piostream&);
      static PersistentTypeID type_id;
    private:
      string              signal_name_; 
      SignalThrower *     thrower_;
    };
  
    class SignalCatcher {
    public:
      SignalCatcher();
      ~SignalCatcher();

      typedef 
      BaseTool::propagation_state_e 
      (SCIRun::Skinner::SignalCatcher::* CatcherFunctionPtr)(event_handle_t);


      typedef 
      BaseTool::propagation_state_e CatcherFunction_t(event_handle_t);


      typedef map<string, CatcherFunctionPtr> signal_slot_map_t;
      signal_slot_map_t   catcher_functions_;
      typedef vector<string> TargetIDs_t;
      TargetIDs_t        get_all_target_ids();
      CatcherFunctionPtr get_catcher(const string &name);
    };
  
  
    class SignalThrower {
    public:      
      SignalThrower();
      virtual ~SignalThrower();
      //      map<string, string>               signalname_catcher_map_t;
      //      signalname_catcher_map_t          catchers_;
      //      SignalCatcher::signal_slot_map_t  catcher_functions_;
      //      map<string, SignalCatcher *>      catcher_classes_;

      typedef pair<SignalCatcher*,SignalCatcher::CatcherFunctionPtr> Catcher_t;
      typedef deque<Catcher_t> CatchersOnDeck_t;
      typedef map<string, CatchersOnDeck_t> SignalCatchers_t;
      //      typedef vector<SignalCatcher *> SignalCatcherPointers_t;
      SignalCatchers_t  catchers_;

      event_handle_t  throw_signal(const string &);
    public:
      static event_handle_t throw_signal(SignalCatchers_t &catchers,
                                         event_handle_t &signal);
                                   
      virtual int     get_signal_id(const string &) const = 0;
      bool            add_catcher_function(const string &, 
                                           SignalCatcher *,
                                           SignalCatcher::CatcherFunctionPtr);

    };

    class Variables;
    class MakerSignal : public Signal
    {
      Variables * variables_;
    public:
      MakerSignal(const string &name, Variables *vars) : 
        Signal(name, 0), variables_(vars) {}
      Variables * get_vars() { return variables_; }
    };
  }
}

#endif
