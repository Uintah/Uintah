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

/*
 * SlotBase
 */

class SlotBase {
public:
  virtual ~SlotBase() {}

  virtual void send() {}
};

/*
 * Slot(void)
 */

template<class Caller>
class Slot : public SlotBase {
private:
  Caller *caller;
  void (Caller::*pmf)();

public:
  Slot( Caller *caller, void (Caller::*pmf)() ) : caller(caller), pmf(pmf) {}
  virtual void send() { (caller->*pmf)(); }
};


/*
 * Slot(arg)
 */

template<class Arg1>
class SlotBase1  {
public:
  virtual void send(Arg1) {}
};

template<class Caller, class Arg1>
class Slot1 : public SlotBase1<Arg1> {
private:
  Caller *caller;
  void (Caller::*pmf)(Arg1);
public:
  Slot1( Caller *caller, void (Caller::*pmf)(Arg1) )
    : caller(caller), pmf(pmf) {}
  void send(Arg1 arg) { (caller->*pmf)(arg); }
};

/*
 * Slot(arg,arg)
 */

template<class Arg1,class Arg2>
class SlotBase2  {
public:
  virtual void send(Arg1,Arg2) {}
};

template<class Caller, class Arg1, class Arg2>
class Slot2 : public SlotBase2<Arg1,Arg2> {
private:
  Caller *caller;
  void (Caller::*pmf)(Arg1,Arg2);
public:
  Slot2( Caller *caller, void (Caller::*pmf)(Arg1,Arg2) )
    : caller(caller), pmf(pmf) {}
  void send(Arg1 a, Arg2 b) { (caller->*pmf)(a,b); }
};


/*
 * Signal(void)
 */

class Signal {
private:
  vector<SlotBase *> slot;
public:
  void add( SlotBase *s) { slot.push_back(s); }
  void send() { for (int i=0; i<slot.size(); i++) slot[i]->send(); }
};

/*
 * Signal(arg)
 */

template<class Arg>
class Signal1  {
private:
  vector<SlotBase1<Arg> *> slot;
public:
  void add( SlotBase1<Arg> *s) { slot.push_back(s); }
  void send( Arg a) { for (int i=0; i<slot.size(); i++) slot[i]->send(a); }
};

/*
 * Signal(arg,arg)
 */

template<class Arg1,class Arg2>
class Signal2  {
private:
  vector<SlotBase2<Arg1,Arg2> *> slot;
public:
  void add( SlotBase2<Arg1,Arg2> *s) { slot.push_back(s); }
  void send( Arg1 a, Arg2 b) 
    { for (int i=0; i<slot.size(); i++) slot[i]->send(a,b); }
};

/*
 * Connect (arg1, arg2)
 */

template<class T, class Arg1, class Arg2>
void connect( Signal2<Arg1,Arg2> &s, T &t, void (T::*fun)(Arg1,Arg2))
{
  SlotBase2<Arg1,Arg2> *slot = new Slot2<T,Arg1,Arg2>(&t, fun);
  s.add( slot);
}

/*
 * Connect (arg)
 */

template<class T, class Arg>
void connect( Signal1<Arg> &s, T &t, void (T::*fun)(Arg))
{
  SlotBase1<Arg> *slot = new Slot1<T,Arg>(&t, fun);
  s.add( slot);
}

/*
 * Connect(void)
 */
template<class T>
void connect( Signal &s, T &t, void (T::*fun)())
{
  Slot<T> *slot = new Slot<T>(&t, fun);
  s.add( slot);
}


#endif // Signals_h
