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

#ifndef Calls_h
#define Calls_h

/*
 * ReplyBase
 */

template<class R>
class ReplyBase {
public:
  virtual ~ReplyBase() {}

  virtual R call() {}
};

/*
 * Reply(void)
 */

template<class R, class Caller>
class Reply : public ReplyBase<R> {
private:
  Caller *caller;
  R (Caller::*pmf)();

public:
  Reply( Caller *caller, R (Caller::*pmf)() ) : caller(caller), pmf(pmf) {}
  virtual R call() { return (caller->*pmf)(); }
};


/*
 * Reply(arg)
 */

template<class R, class Arg1>
class ReplyBase1  {
public:
  virtual R send(Arg1) {}
};

template<class R, class Caller, class Arg1>
class Reply1 : public ReplyBase1<R, Arg1> {
private:
  Caller *caller;
  R (Caller::*pmf)(Arg1);
public:
  Reply1( Caller *caller, R (Caller::*pmf)(Arg1) )
    : caller(caller), pmf(pmf) {}
  R call(Arg1 arg) { return (caller->*pmf)(arg); }
};

/*
 * Reply(arg,arg)
 */

template<class R, class Arg1,class Arg2>
class ReplyBase2  {
public:
  virtual R call(Arg1,Arg2) {}
};

template<class R, class Caller, class Arg1, class Arg2>
class Reply2 : public ReplyBase2<R, Arg1,Arg2> {
private:
  Caller *caller;
  R (Caller::*pmf)(Arg1,Arg2);
public:
  Reply2( Caller *caller, R (Caller::*pmf)(Arg1,Arg2) )
    : caller(caller), pmf(pmf) {}
  R send(Arg1 a, Arg2 b) { return (caller->*pmf)(a,b); }
};


/*
 * Signal(void)
 */

template<class R>
class Call {
private:
  ReplyBase<R> *slot;
public:
  Call() : slot(0) {}
  void add( ReplyBase<R> *s) { if (slot) delete slot; slot = s;}
  R operator()() { return slot->call(); }
};


/*
 * Signal(arg)
 */

template<class R, class Arg>
class Call1  {
private:
  ReplyBase1<R,Arg> *slot;
public:
  Call1() : slot(0) {}
  void add( ReplyBase1<R,Arg> *s) { if (slot) delete slot; slot=s; }
  R operator()(Arg a) { return slot->call(a); }
};

/*
 * Signal(arg,arg)
 */

template<class R, class Arg1,class Arg2>
class Call2  {
private:
  ReplyBase2<R,Arg1,Arg2> *slot;
public:
  Call2() : slot(0) {}
  void add( ReplyBase2<R,Arg1,Arg2> *s) {if (slot) delete slot; slot=s;}
  R operator()( Arg1 a, Arg2 b) { return slot->call(a,b); }
};

/*
 * Connect (arg1, arg2)
 */

template<class R, class T, class Arg1, class Arg2>
void connect( Call2<R,Arg1,Arg2> &s, T &t, R (T::*fun)(Arg1,Arg2))
{
  ReplyBase2<R,Arg1,Arg2> *slot = new Reply2<R,T,Arg1,Arg2>(&t, fun);
  s.add( slot);
}

/*
 * Connect (arg)
 */

template<class R, class T, class Arg>
void connect( Call1<R,Arg> &s, T &t, R (T::*fun)(Arg))
{
  ReplyBase1<R,Arg> *slot = new Reply1<R,T,Arg>(&t, fun);
  s.add( slot);
}


/*
 * Connect(void)
 */
template<class R,class T>
void connect( Call<R> &s, T &t, R (T::*fun)())
{
  Reply<R,T> *slot = new Reply<R,T>(&t, fun);
  s.add( slot);
}


#endif // Calls_h


