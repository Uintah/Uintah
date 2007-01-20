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
//    File   : Signals.h
//    Author : McKay Davis
//    Date   : Thu Jun 29 18:13:09 2006

#ifndef SKINNER_SIGNALS_H
#define SKINNER_SIGNALS_H

#include <Core/Events/BaseEvent.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Malloc/Allocator.h>
#include <string>
#include <deque>
#include <vector>
using std::string;
using std::deque;
using std::vector;

#define REGISTER_CATCHER_TARGET(catcher_target_function_name) \
  this->register_default_thrower \
    (this->register_target(#catcher_target_function_name, this, \
       &catcher_target_function_name));

#define REGISTER_CATCHER_TARGET_BY_NAME(target_name, function_name) \
  this->register_default_thrower \
    (this->register_target(#target_name, this, &function_name));


#include <Core/Skinner/share.h>

namespace SCIRun {
  class PointerEvent;
  namespace Skinner {

    class Signal; 
    class SignalThrower;
    class SignalCatcher;
    class Variables;
    class Drawable;

    class SCISHARE Signal : public SCIRun::BaseEvent
    {
    public:
      Signal(const string &name, SignalThrower *, Variables *);
      Signal(const Signal &copy);
      ~Signal();
      string              get_signal_name() { return signal_name_; }
      void                set_signal_name(const string &n) { signal_name_=n; }

      Variables *         get_vars() { return variables_; }
      void                set_vars(Variables *vars) { variables_ = vars;}

      SignalThrower *     get_signal_thrower() { return thrower_; }
      void                set_signal_thrower(SignalThrower *t) { thrower_=t; }

      BaseTool::propagation_state_e     get_signal_result() { return result_; }
      void  set_signal_result(BaseTool::propagation_state_e r) { result_ = r; }
      
      virtual Signal *                  clone() { return new Signal(*this); }
      virtual void                      io(Piostream&);
      static PersistentTypeID           type_id;
    private:
      string                            signal_name_;
      Variables *                       variables_; 
      SignalThrower *                   thrower_;
      BaseTool::propagation_state_e     result_;
    };
    
    class SCISHARE SignalCatcher {
    public:
      SignalCatcher();
      ~SignalCatcher();

      // Typedefs 
      typedef 
      BaseTool::propagation_state_e CatcherFunction_t(event_handle_t);

      struct SCISHARE CatcherTargetInfoBase {
        virtual ~CatcherTargetInfoBase();
        CatcherTargetInfoBase() {}
        CatcherTargetInfoBase(string target, Variables* vars) : 
          targetname_(target), variables_(vars) {}
        string targetname_;
        Variables* variables_;
        virtual BaseTool::propagation_state_e doCallback(event_handle_t signal) = 0;
        virtual Drawable* getDrawable() = 0;
        virtual CatcherTargetInfoBase* clone() = 0;
      };

      template <class T>
      struct CatcherTargetInfo_t : public CatcherTargetInfoBase { 

        typedef BaseTool::propagation_state_e (T::*FuncPtr)(event_handle_t);

        CatcherTargetInfo_t();
        CatcherTargetInfo_t(const CatcherTargetInfo_t &copy);
        T*                      catcher_;
        FuncPtr function_;
        virtual Drawable* getDrawable() { return (Drawable*) catcher_; }
        virtual BaseTool::propagation_state_e doCallback(event_handle_t signal) {  
          ASSERT(catcher_); 
          ASSERT(function_);
          return (catcher_->*function_)(signal); 
        }
        virtual CatcherTargetInfoBase* clone() { return scinew CatcherTargetInfo_t<T>(*this); }
      };

      typedef vector<CatcherTargetInfoBase*> NodeCatchers_t;
      typedef vector<NodeCatchers_t> TreeOfCatchers_t;


      // Only method
      NodeCatchers_t            get_all_targets() { return catcher_targets_; }
    protected:
      template <class T>
      CatcherTargetInfoBase*    register_target(const string &targetname, T* caller, 
                                                typename CatcherTargetInfo_t<T>::FuncPtr function);

    private:
      NodeCatchers_t            catcher_targets_;            
    };
  
    template <class T>
    SignalCatcher::CatcherTargetInfoBase*
    SignalCatcher::register_target(const string &targetname, T* caller,
                                   typename SignalCatcher::CatcherTargetInfo_t<T>::FuncPtr function) 
    {
      ASSERT(function);
      CatcherTargetInfo_t<T>* callback = new CatcherTargetInfo_t<T>;
      callback->catcher_ = caller;
      callback->function_ = function;
      callback->targetname_ = targetname;
      callback->variables_ = 0;

      catcher_targets_.push_back(callback);
      return callback;
    }

    template <class T>
    SignalCatcher::CatcherTargetInfo_t<T>::CatcherTargetInfo_t() :
      SignalCatcher::CatcherTargetInfoBase("", 0),
      catcher_(0),
      function_(0)
    {
    }

    template <class T>
    SignalCatcher::CatcherTargetInfo_t<T>::CatcherTargetInfo_t
    (const CatcherTargetInfo_t &copy) :
      SignalCatcher::CatcherTargetInfoBase(copy.targetname_, copy.variables_),
      catcher_(copy.catcher_),
      function_(copy.function_) 
    {
    }

  
    class SCISHARE SignalThrower {
    public:      
      SignalThrower();
      virtual ~SignalThrower();

      // Typedefs
      typedef vector<SignalCatcher::CatcherTargetInfoBase*> AllSignalCatchers_t;
      typedef map<string, AllSignalCatchers_t> SignalToAllCatchers_t;

      event_handle_t                    throw_signal(const string &,
                                                     Variables *vars);

      event_handle_t                    throw_signal(event_handle_t &);

      static event_handle_t             throw_signal(SignalToAllCatchers_t &,
                                                     event_handle_t &signal);
      
      void                              register_default_thrower(SignalCatcher::CatcherTargetInfoBase*);

      static SignalToAllCatchers_t      collapse_tree(SignalCatcher::TreeOfCatchers_t &);
                                   
      virtual int                       get_signal_id(const string &) const = 0;

      bool                              expose_catcher(const string &);

      // Should be private
      SignalToAllCatchers_t all_catchers_;
    };



    class SCISHARE SignalCallback {
    public:
      SignalCallback(Drawable *, const string &signalname);
      SignalCallback(Signal *signal);
      virtual ~SignalCallback();
      event_handle_t                            operator()();
    private:
      Signal *                                  signal_;
      event_handle_t                            signalh_;
      SignalThrower::AllSignalCatchers_t        catchers_;
      bool                                      init_;
    };

      
        


    class Variables;

    class SCISHARE MakerSignal : public Signal
    {
      Variables * variables_;
    public:
      MakerSignal(const string &name, Variables *vars) : 
        Signal(name, 0, vars), variables_(vars) {}
      Variables * get_vars() { return variables_; }
    };

    class SCISHARE PointerSignal : public Signal
    {
      PointerEvent *pointer_;
    public:
      PointerSignal(const string &name, PointerEvent *pointer) :
        Signal(name, 0, 0), 
        pointer_(pointer)
      {}

      PointerEvent *get_pointer_event() { return pointer_; }
      void      set_pointer_event(PointerEvent *p) { pointer_ = p; }
    };

    class SCISHARE KeySignal : public Signal
    {
      KeyEvent *key_;
    public:
      KeySignal(const string &name, KeyEvent *key) :
        Signal(name, 0, 0), 
        key_(key)
      {}

      KeyEvent *get_key_event() { return key_; }
      void      set_key_event(KeyEvent *k) { key_ = k; }
    };

  }
}

#endif
