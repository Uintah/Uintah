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
 *  AtomicCounter: Thread-safe integer variable
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <cstdio>
#include <cstdlib>

#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>

namespace SCIRun {
struct AtomicCounter_private {
  Mutex lock;
  long long value;
  AtomicCounter_private();
  ~AtomicCounter_private();
};
}

using SCIRun::AtomicCounter_private;
using SCIRun::AtomicCounter;

AtomicCounter_private::AtomicCounter_private()
    : lock("AtomicCounter lock")
{
}

AtomicCounter_private::~AtomicCounter_private()
{
}

AtomicCounter::AtomicCounter(const char* name)
    : name_(name)
{
  if(!Thread::isInitialized()){
    if(getenv("THREAD_SHOWINIT"))
      fprintf(stderr, "AtomicCounter: %s\n", name);
    Thread::initialize();
  }
  priv_=new AtomicCounter_private;
}

AtomicCounter::AtomicCounter(const char* name, long long value)
    : name_(name)
{
  priv_=new AtomicCounter_private;
  priv_->value=value;
}

AtomicCounter::~AtomicCounter()
{
  delete priv_;
  priv_=0;
}

AtomicCounter::operator long long() const
{
    return priv_->value;
}

long long
AtomicCounter::operator++()
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  long long ret=++priv_->value;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

long long
AtomicCounter::operator++(int)
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  long long ret=priv_->value++;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

long long
AtomicCounter::operator--()
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  long long ret=--priv_->value;	
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
}

long long
AtomicCounter::operator--(int)
{
  int oldstate = Thread::couldBlock(name_);
  priv_->lock.lock();
  long long ret=priv_->value--;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
  return ret;
} 

void
AtomicCounter::set(long long v)
{
  int oldstate=Thread::couldBlock(name_);
  priv_->lock.lock();
  priv_->value=v;
  priv_->lock.unlock();
  Thread::couldBlockDone(oldstate);
}
