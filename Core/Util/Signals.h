/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  Signals.h: Signals and Slots Mechanism
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Signals_h
#define Signals_h


#include <vector>
#include <Core/Thread/Mutex.h>

namespace SCIRun {

/*
 * SlotBase & SignalBase
 */

class SlotBase {
private:
  int priority_;

public:
  SlotBase( int priority=0) : priority_(priority) {}
  virtual ~SlotBase() {}

  int priority() {return priority_;}
  virtual void send() {}
};


class SignalBase {
 protected:
  Mutex lock_;
  vector<SlotBase *> slot_;
  
 protected: //you can not allocated a SignalBase
  SignalBase() : lock_("signal lock") {} 
  
 public:
  void add( SlotBase *s )
    { 
      lock_.lock(); 
      // using insert sort
      slot_.push_back(s); 
      int priority = s->priority();
      int i=slot_.size();
      for (i--; i>0; i--) {
	if ( slot_[i-1]->priority() < priority )
	  slot_[i] = slot_[i-1];
	else
	  break;
      }
      slot_[i] = s;

      lock_.unlock();
    }
    
  template<class Slot>
  bool rem( const Slot &r) 
    { 
      lock_.lock();
      for (unsigned i=0; i<slot_.size(); i++) {
	Slot *s = dynamic_cast<Slot *>(slot_[i]);
	if ( s && (*s) == r ) {
	  delete slot_[i];
	  slot_.erase( slot_.begin()+i);
	  lock_.unlock();
	  return true;
	}
      }
      lock_.unlock();
      return false;
    }
};

//
// ************** no args
//

/*
 * Static Slot(void)
 */

class StaticSlot : public SlotBase {
private:
  void (*fun)();
public:
  StaticSlot( void (*fun)(), int priority=0 ) : SlotBase(priority), fun(fun) {}
  
  virtual void send() { (*fun)(); }
  bool operator==(const StaticSlot &s) { return fun == s.fun;}
};

/*
 * Slot(void)
 */

template<class Caller>
class Slot : public SlotBase {
private:
  Caller *caller_;
  void (Caller::*pmf_)();

public:
  Slot( Caller *caller, void (Caller::*pmf)(), int priority=0 ) 
    : SlotBase(priority), caller_(caller), pmf_(pmf) {}
  virtual void send() { (caller->*pmf)(); }
  bool operator==(const Slot &s) 
    { 
      return caller_ == s.caller_ && pmf_ == s.pmf_; 
    }
};


/*
 * Signal(void)
 */

class Signal : public SignalBase {
public:
  void operator()() 
    { lock_.lock();
    for (unsigned i=0; i<slot_.size(); i++) slot_[i]->send();
    lock_.unlock();
    }
};

/*
 * Connect(void)
 */
template<class T>
void connect( Signal &s, T &t, void (T::*fun)(), int priority=0)
{
  s.add( new Slot<T>(&t, fun, priority) );
}


template<class T>
void connect( Signal &s, T *t, void (T::*fun)(), int priority=0)
{
  s.add( new Slot<T>(t, fun, priority) );
}

inline
void connect( Signal &s, void (*fun)(), int priority=0 )
{
  s.add ( new StaticSlot( fun, priority ) );
}

/*
 * Disconnect (void)
 */

template<class T>
bool disconnect( Signal &s, T &t, void (T::*fun)())
{
  return s.rem(Slot<T>(&t, fun));
}

template<class T>
bool disconnect( Signal &s, T *t, void (T::*fun)())
{
  return s.rem(Slot<T>(t, fun));
}

inline
bool disconnect( Signal &s, void (*fun)() )
{
  return s.rem( StaticSlot( fun ) );
}
  
//
// ****************** single arg
//

/*
 * Slot(arg)
 */

template<class Arg1>
class SlotBase1 : public SlotBase  {
public:
  SlotBase1(int priority=0) : SlotBase(priority) {}
  virtual void send(Arg1) {}
};

template<class Caller, class Arg1>
class Slot1 : public SlotBase1<Arg1> {
private:
  Caller *caller_;
  void (Caller::*pmf_)(Arg1);
public:
  Slot1( Caller *caller, void (Caller::*pmf)(Arg1), int priority=0 )
    : SlotBase1<Arg1>(priority), caller_(caller), pmf_(pmf) {}
  void send(Arg1 arg) { (caller_->*pmf_)(arg); }
  bool operator==(const Slot1<Caller,Arg1> &s) 
    { return caller_ == s.caller_ && pmf_ == s.pmf_; }
};

/*
 * Static Slot(arg)
 */

template<class Arg1>
class StaticSlot1 : public SlotBase1<Arg1> {
private:
  void (*fun_)(Arg1);
public:
  StaticSlot1( void (*fun)(Arg1), int priority=0 ) 
    : SlotBase1<Arg1>(priority), fun_(fun) {}
  
  virtual void send(Arg1 a) { (*fun_)(a); }
  bool operator==(const StaticSlot1<Arg1> &s) { return fun_ == s.fun_;}
};

/*
 * Signal(arg)
 */

template<class Arg>
class Signal1  : public SignalBase {
public:
  void add( SlotBase1<Arg> *s) { SignalBase::add( s ); }
  void operator()(Arg a)
    {
      lock_.lock();
      for (unsigned i=0; i<slot_.size(); i++)
	static_cast<SlotBase1<Arg>*>(slot_[i])->send(a);
      lock_.unlock();
    }
};

/*
 * Connect (arg)
 */

template<class T, class Arg>
void connect( Signal1<Arg> &s, T &t, void (T::*fun)(Arg), int priority=0)
{
  s.add( new Slot1<T,Arg>(&t, fun, priority) );
}

template<class T, class Arg>
void connect( Signal1<Arg> &s, T *t, void (T::*fun)(Arg), int priority=0)
{
  s.add( new Slot1<T,Arg>(t, fun, priority) );
}


template<class Arg1>
void connect( Signal &s, void (*fun)(Arg1), int priority=0 )
{
  s.add( new StaticSlot1<Arg1>( fun, priority ));
}

/*
 * Disconnect (Arg)
 */

template<class T, class Arg>
bool disconnect( Signal1<Arg> &s, T &t, void (T::*fun)(Arg))
{
  return s.rem(Slot1<T,Arg>(&t, fun));
}

template<class T, class Arg>
bool disconnect( Signal1<Arg> &s, T *t, void (T::*fun)(Arg))
{
  return s.rem(Slot1<T,Arg>(t, fun));
}

template<class Arg>
bool disconnect( Signal1<Arg> &s, void (*fun)(Arg) )
{
  return s.rem( StaticSlot1<Arg>( fun ) );
}

//
// ***************** two args
//

/*
 * Slot(arg,arg)
 */

template<class Arg1,class Arg2>
class SlotBase2 :  public SlotBase {
public:
  virtual void send(Arg1,Arg2) {}
};

template<class Caller, class Arg1, class Arg2>
class Slot2 : public SlotBase2<Arg1,Arg2> {
private:
  Caller *caller_;
  void (Caller::*pmf_)(Arg1,Arg2);
public:
  Slot2( Caller *caller, void (Caller::*pmf)(Arg1,Arg2),int priority=0 )
    : SlotBase2<Arg1,Arg2>(priority), caller_(caller), pmf_(pmf) {}
  void send(Arg1 a, Arg2 b) { (caller_->*pmf_)(a,b); }
};

/*
 * Static Slot(arg)
 */

template<class Arg1, class Arg2>
class StaticSlot2 : public SlotBase2<Arg1,Arg2> {
private:
  void (*fun_)(Arg1,Arg2);
public:
  StaticSlot2( void (*fun)(Arg1,Arg2), int priority ) 
    : SlotBase2<Arg1,Arg2>(priority), fun_(fun) {}
  
  virtual void send(Arg1 a, Arg2 b) { (*fun_)(a,b); }
};


/*
 * Signal2(arg,arg)
 */

template<class Arg1,class Arg2>
class Signal2 : public SignalBase {
private:
  vector<SlotBase2<Arg1,Arg2> *> slot_;
public:
  void add( SlotBase2<Arg1,Arg2> *s) { SignalBase::add( s ); }
  void operator()(Arg1 a, Arg2 b )
    {
      lock_.lock();
      for (unsigned i=0; i<slot_.size(); i++)
	static_cast<SlotBase2<Arg1,Arg2>*>(s)->send(a,b);
      lock_.unlock();
    }
};

/*
 * Connect (arg1, arg2)
 */

template<class T, class Arg1, class Arg2>
void connect( Signal2<Arg1,Arg2> &s, T &t, void (T::*fun)(Arg1,Arg2), 
	      int priority=0)
{
  s.add( new Slot2<T,Arg1,Arg2>(&t, fun, priority) );
}

template<class T, class Arg1, class Arg2>
void connect( Signal2<Arg1,Arg2> &s, T *t, void (T::*fun)(Arg1,Arg2), 
	      int priority=0)
{
  s.add( new Slot2<T,Arg1,Arg2>(t, fun, priority) );
}

template<class Arg1, class Arg2>
void connect( Signal &s, void (*fun)(Arg1,Arg2), int priority=0 )
{
  s.add( new StaticSlot2<Arg1,Arg2>( fun, priority ) );
}

/*
 * Disconnect (Arg1, Arg2)
 */

template<class T, class Arg1, class Arg2>
bool disconnect( Signal2<Arg1, Arg2> &s, T &t, void (T::*fun)(Arg1, Arg2))
{
  return s.rem(Slot2<T,Arg1, Arg2>(&t, fun));
}

template<class T, class Arg1, class Arg2>
bool disconnect( Signal2<Arg1, Arg2> &s, T *t, void (T::*fun)(Arg1, Arg2))
{
  return s.rem(Slot2<T,Arg1, Arg2>(t, fun));
}

template<class Arg1, class Arg2>
bool disconnect( Signal2<Arg1, Arg2> &s, void (*fun)(Arg1, Arg2) )
{
  return s.rem( StaticSlot2<Arg1, Arg2>( fun ) );
}

}; // namespace SCIRun

#endif // Signals_h
