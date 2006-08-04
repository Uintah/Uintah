/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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


