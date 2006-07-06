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


#define REGISTER_CATCHER_TARGET(catcher_target_function_name) \
  this->register_target(#catcher_target_function_name, \
    static_cast<SCIRun::Skinner::SignalCatcher::CatcherFunctionPtr> \
      (&catcher_target_function_name));


namespace SCIRun {
  namespace Skinner {

    class Signal; 
    class SignalThrower;
    class SignalCatcher;
    class Variables;

    class Signal : public SCIRun::BaseEvent
    {
    public:
      Signal(const string &name, SignalThrower *, const string &data = "");
      ~Signal();
      string              get_signal_name() { return signal_name_; }
      void                set_signal_name(const string &n) { signal_name_=n; }

      const string &      get_signal_data() { return signal_data_; }
      void                set_signal_data(const string &d) { signal_data_=d; }

      SignalThrower *     get_signal_thrower() { return thrower_; }
      void                set_signal_thrower(SignalThrower *t) { thrower_=t; }
      
      virtual Signal *    clone() { return new Signal(*this); }
      virtual void        io(Piostream&);
      static PersistentTypeID type_id;
    private:
      string              signal_name_;
      string              signal_data_;
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


      struct CatcherTargetInfo_t { 
        SignalCatcher *         catcher_;
        CatcherFunctionPtr      function_;
        string                  data_;
        string                  targetname_;
      };

      typedef vector<CatcherTargetInfo_t> NodeCatchers_t;
      typedef vector<NodeCatchers_t> TreeOfCatchers_t;
      NodeCatchers_t            get_all_targets() { return catcher_targets_; }
    protected:
      void                      register_target(const string &targetname,
                                                CatcherFunctionPtr function);

    private:
      NodeCatchers_t            catcher_targets_;            
    };
  
  
    class SignalThrower {
    public:      
      SignalThrower();
      virtual ~SignalThrower();
      typedef vector<SignalCatcher::CatcherTargetInfo_t> AllSignalCatchers_t;
      typedef map<string, AllSignalCatchers_t> SignalToAllCatchers_t;
      SignalToAllCatchers_t all_catchers_;


      event_handle_t  throw_signal(const string &,
                                   Variables *vars = 0);


    public:
      static event_handle_t throw_signal(SignalToAllCatchers_t &catchers,
                                          event_handle_t &signal,
                                          Variables *vars = 0);

      static SignalToAllCatchers_t  collapse_tree(SignalCatcher::TreeOfCatchers_t &);

                                   
      virtual int     get_signal_id(const string &) const = 0;

      bool            hookup_signal_to_catcher_target(const string &,
                                                      const string &,
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
