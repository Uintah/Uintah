/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */

#ifndef Signals_h
#define Signals_h

//#include <iostream>
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
  virtual ~SlotBase();

  int priority() {return priority_;}
  virtual void send();
};


class SignalBase {
 protected:
  Mutex lock_;
  std::vector<SlotBase *> slot_;
  
 protected: //you can not allocated a SignalBase
  SignalBase() : lock_("signal lock") {} 
  virtual ~SignalBase();
  
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
  virtual ~StaticSlot();
  
  virtual void send();
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
  ~Slot();

  virtual void send() { (caller_->*pmf_)(); }
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

  ~Signal();

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

//void connect( Signal &s, void (*fun)(), int priority=0 );

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

//bool disconnect( Signal &s, void (*fun)() );
  
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

template<class T, class Arg1>
void connect( Signal1<Arg1> &s, T &t, void (T::*fun)(Arg1), int priority=0)
{
  s.add( new Slot1<T,Arg1>(&t, fun, priority) );
}

template<class T, class Arg1>
void connect( Signal1<Arg1> &s, T *t, void (T::*fun)(Arg1), int priority=0)
{
  s.add( new Slot1<T,Arg1>(t, fun, priority) );
}


template<class Arg1>
void connect( Signal1<Arg1> &s, void (*fun)(Arg1), int priority=0 )
{
  s.add( new StaticSlot1<Arg1>( fun, priority ));
}

/*
 * Disconnect (Arg)
 */

template<class T, class Arg1>
bool disconnect( Signal1<Arg1> &s, T &t, void (T::*fun)(Arg1))
{
  return s.rem(Slot1<T,Arg1>(&t, fun));
}

template<class T, class Arg1>
bool disconnect( Signal1<Arg1> &s, T *t, void (T::*fun)(Arg1))
{
  return s.rem(Slot1<T,Arg1>(t, fun));
}

template<class Arg1>
bool disconnect( Signal1<Arg1> &s, void (*fun)(Arg1) )
{
  return s.rem( StaticSlot1<Arg1>( fun ) );
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
  SlotBase2(int priority=0) : SlotBase(priority) {}
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
public:
  void add( SlotBase2<Arg1,Arg2> *s) { SignalBase::add( s ); }
  void operator()(Arg1 a, Arg2 b )
    {
      lock_.lock();
      //cerr << "num connections: " << slot_.size() << endl;
      for (unsigned i=0; i<slot_.size(); i++)
	static_cast<SlotBase2<Arg1,Arg2>*>(slot_[i])->send(a,b);
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
void connect( Signal2<Arg1,Arg2> &s, void (*fun)(Arg1,Arg2), int priority=0 )
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

//
// ***************** three args
//

/*
 * Slot(arg,arg,arg)
 */

template<class Arg1,class Arg2,class Arg3>
class SlotBase3 :  public SlotBase {
public:
  SlotBase3(int priority=0) : SlotBase(priority) {}
  virtual void send(Arg1,Arg2,Arg3) {}
};

template<class Caller, class Arg1, class Arg2, class Arg3>
class Slot3 : public SlotBase3<Arg1,Arg2,Arg3> {
private:
  Caller *caller_;
  void (Caller::*pmf_)(Arg1,Arg2,Arg3);
public:
  Slot3( Caller *caller, void (Caller::*pmf)(Arg1,Arg2,Arg3),int priority=0 )
    : SlotBase3<Arg1,Arg2,Arg3>(priority), caller_(caller), pmf_(pmf) {}
  void send(Arg1 a, Arg2 b, Arg3 c) { (caller_->*pmf_)(a,b,c); }
};

/*
 * Static Slot(arg,arg,arg)
 */

template<class Arg1, class Arg2, class Arg3>
class StaticSlot3 : public SlotBase3<Arg1,Arg2,Arg3> {
private:
  void (*fun_)(Arg1,Arg2,Arg3);
public:
  StaticSlot3( void (*fun)(Arg1,Arg2,Arg3), int priority ) 
    : SlotBase3<Arg1,Arg2,Arg3>(priority), fun_(fun) {}
  
  virtual void send(Arg1 a, Arg2 b, Arg3 c) { (*fun_)(a,b,c); }
};


/*
 * Signal3(arg,arg,arg)
 */

template<class Arg1,class Arg2,class Arg3>
class Signal3 : public SignalBase {
public:
  void add( SlotBase3<Arg1,Arg2,Arg3> *s) { SignalBase::add( s ); }
  void operator()(Arg1 a, Arg2 b, Arg3 c)
    {
      lock_.lock();
      //cerr << "num connections: " << slot_.size() << endl;
      for (unsigned i=0; i<slot_.size(); i++)
	static_cast<SlotBase3<Arg1,Arg2,Arg3>*>(slot_[i])->send(a,b,c);
      lock_.unlock();
    }
};

/*
 * Connect (arg1, arg2, arg3)
 */

template<class T, class Arg1, class Arg2, class Arg3>
void connect( Signal3<Arg1,Arg2,Arg3> &s, T &t, void (T::*fun)(Arg1,Arg2,Arg3),
	      int priority=0)
{
  s.add( new Slot3<T,Arg1,Arg2,Arg3>(&t, fun, priority) );
}

template<class T, class Arg1, class Arg2, class Arg3>
void connect( Signal3<Arg1,Arg2,Arg3> &s, T *t, void (T::*fun)(Arg1,Arg2,Arg3),
	      int priority=0)
{
  s.add( new Slot3<T,Arg1,Arg2,Arg3>(t, fun, priority) );
}

template<class Arg1, class Arg2, class Arg3>
void connect( Signal3<Arg1,Arg2,Arg3> &s, void (*fun)(Arg1,Arg2,Arg3), 
	      int priority=0 )
{
  s.add( new StaticSlot3<Arg1,Arg2,Arg3>( fun, priority ) );
}


/*
 * Disconnect (Arg1, Arg2, Arg3)
 */

template<class T, class Arg1, class Arg2, class Arg3>
bool disconnect( Signal3<Arg1, Arg2, Arg3> &s, T &t, 
                 void (T::*fun)(Arg1, Arg2, Arg3))
{
  return s.rem(Slot3<T,Arg1, Arg2, Arg3>(&t, fun));
}

template<class T, class Arg1, class Arg2, class Arg3>
bool disconnect( Signal3<Arg1, Arg2, Arg3> &s, T *t, 
                 void (T::*fun)(Arg1, Arg2, Arg3))
{
  return s.rem(Slot3<T,Arg1, Arg2, Arg3>(t, fun));
}

template<class Arg1, class Arg2, class Arg3>
bool disconnect( Signal3<Arg1, Arg2, Arg3> &s, void (*fun)(Arg1, Arg2, Arg3) )
{
  return s.rem( StaticSlot3<Arg1, Arg2, Arg3>( fun ) );
}

//
// ***************** Four args
//

/*
 * Slot(arg,arg,arg,arg)
 */

template<class Arg1,class Arg2,class Arg3, class Arg4>
class SlotBase4 :  public SlotBase {
public:
  SlotBase4(int priority=0) : SlotBase(priority) {}
  virtual void send(Arg1,Arg2,Arg3,Arg4) {}
};

template<class Caller, class Arg1, class Arg2, class Arg3, class Arg4>
class Slot4 : public SlotBase4<Arg1,Arg2,Arg3,Arg4> {
private:
  Caller *caller_;
  void (Caller::*pmf_)(Arg1,Arg2,Arg3,Arg4);
public:
  Slot4( Caller *caller, void (Caller::*pmf)(Arg1,Arg2,Arg3,Arg4),
	 int priority=0 )
    : SlotBase4<Arg1,Arg2,Arg3,Arg4>(priority), caller_(caller), pmf_(pmf) {}
  void send(Arg1 a, Arg2 b, Arg3 c, Arg4 d) { (caller_->*pmf_)(a,b,c,d); }
};

/*
 * Static Slot(arg,arg,arg,arg)
 */

template<class Arg1, class Arg2, class Arg3, class Arg4>
class StaticSlot4 : public SlotBase4<Arg1,Arg2,Arg3,Arg4> {
private:
  void (*fun_)(Arg1,Arg2,Arg3,Arg4);
public:
  StaticSlot4( void (*fun)(Arg1,Arg2,Arg3,Arg4), int priority ) 
    : SlotBase4<Arg1,Arg2,Arg3,Arg4>(priority), fun_(fun) {}
  
  virtual void send(Arg1 a, Arg2 b, Arg3 c, Arg4 d) { (*fun_)(a,b,c,d); }
};


/*
 * Signal3(arg,arg,arg,arg)
 */

template<class Arg1,class Arg2,class Arg3,class Arg4>
class Signal4 : public SignalBase {
public:
  void add( SlotBase4<Arg1,Arg2,Arg3,Arg4> *s) { SignalBase::add( s ); }
  void operator()(Arg1 a, Arg2 b, Arg3 c, Arg4 d)
    {
      lock_.lock();
      //cerr << "num connections: " << slot_.size() << endl;
      for (unsigned i=0; i<slot_.size(); i++)
	static_cast<SlotBase4<Arg1,Arg2,Arg3,Arg4>*>(slot_[i])->send(a,b,c,d);
      lock_.unlock();
    }
};

/*
 * Connect (arg1, arg2, arg3, arg4)
 */

template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
void connect( Signal4<Arg1,Arg2,Arg3,Arg4> &s, T &t, 
	      void (T::*fun)(Arg1,Arg2,Arg3,Arg4), int priority=0)
{
  s.add( new Slot4<T,Arg1,Arg2,Arg3,Arg4>(&t, fun, priority) );
}

template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
void connect( Signal4<Arg1,Arg2,Arg3,Arg4> &s, T *t, 
	      void (T::*fun)(Arg1,Arg2,Arg3,Arg4), int priority=0)
{
  s.add( new Slot4<T,Arg1,Arg2,Arg3,Arg4>(t, fun, priority) );
}

template<class Arg1, class Arg2, class Arg3, class Arg4>
void connect( Signal4<Arg1,Arg2,Arg3,Arg4> &s, 
	      void (*fun)(Arg1,Arg2,Arg3,Arg4), int priority=0 )
{
  s.add( new StaticSlot4<Arg1,Arg2,Arg3,Arg4>( fun, priority ) );
}


/*
 * Disconnect (Arg1, Arg2, Arg3, Arg4)
 */

template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
bool disconnect( Signal4<Arg1, Arg2, Arg3, Arg4> &s, T &t, 
                 void (T::*fun)(Arg1, Arg2, Arg3, Arg4))
{
  return s.rem(Slot4<T,Arg1, Arg2, Arg3, Arg4>(&t, fun));
}

template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
bool disconnect( Signal4<Arg1, Arg2, Arg3, Arg4> &s, T *t, 
                 void (T::*fun)(Arg1, Arg2, Arg3, Arg4))
{
  return s.rem(Slot4<T,Arg1, Arg2, Arg3, Arg4>(t, fun));
}

template<class Arg1, class Arg2, class Arg3, class Arg4>
bool disconnect( Signal4<Arg1, Arg2, Arg3, Arg4> &s, 
		 void (*fun)(Arg1, Arg2, Arg3) )
{
  return s.rem( StaticSlot4<Arg1, Arg2, Arg3, Arg4>( fun ) );
}

} // namespace SCIRun


#endif // Signals_h
