/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  Datatype.cc: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Thread/AtomicCounter.h>

namespace SCIRun {

static AtomicCounter* current_generation = 0;
static Mutex init_lock("Datatypes generation counter initialization lock");


int
Datatype::compute_new_generation()
{
  if(!current_generation)
  {
    init_lock.lock();
    if(!current_generation)
    {
      current_generation = new AtomicCounter("Datatype generation counter", 1);
      init_lock.unlock();
    }
  }
  return (*current_generation)++;
}



Datatype::Datatype()
  : ref_cnt(0),
    lock("Datatype ref_cnt lock"),
    generation(compute_new_generation())
{
}

Datatype::Datatype(const Datatype&)
  : ref_cnt(0),
    lock("Datatype ref_cnt lock"),
    generation(compute_new_generation())
{
}

Datatype& Datatype::operator=(const Datatype&)
{
  ASSERT(ref_cnt == 1);
  generation = compute_new_generation();
  return *this;
}

Datatype::~Datatype()
{
}

} // End namespace SCIRun

