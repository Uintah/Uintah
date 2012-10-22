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
#include <Core/Thread/AtomicCounter.h>

namespace SCIRun {
struct AtomicCounter_private {
};
}
 
using SCIRun::AtomicCounter;

AtomicCounter::AtomicCounter(const char* name)
    : name_(name)
{
}

AtomicCounter::AtomicCounter(const char* name, long long value)
    : name_(name)
{
  value_=value;
}

AtomicCounter::~AtomicCounter()
{
}

AtomicCounter::operator long long() const
{
    return value_;
}

long long
AtomicCounter::operator++()
{
  return __sync_add_and_fetch(&value_,1);
}

long long
AtomicCounter::operator++(int)
{
  return __sync_fetch_and_add(&value_,1);
}

long long
AtomicCounter::operator--()
{
  return __sync_sub_and_fetch(&value_,1);
}

long long
AtomicCounter::operator--(int)
{
  return __sync_fetch_and_sub(&value_,1);
} 

void
AtomicCounter::set(long long v)
{
  __sync_val_compare_and_swap(&value_, value_, v);
}

