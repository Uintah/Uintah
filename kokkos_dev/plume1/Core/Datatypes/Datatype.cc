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
 *  Datatype.cc: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Thread/AtomicCounter.h>

namespace SCIRun {

static AtomicCounter* current_generation = 0;
static Mutex init_lock("Datatypes generation counter initialization lock");

Datatype::Datatype()
: lock("Datatype ref_cnt lock")
{
    ref_cnt=0;
    if(!current_generation){
      init_lock.lock();
      if(!current_generation)
	current_generation = new AtomicCounter("Datatypes generation counter", 1);
      init_lock.unlock();
    }
    generation=(*current_generation)++;
}

Datatype::Datatype(const Datatype&)
    : lock("Datatype ref_cnt lock")
{
    ref_cnt=0;
    generation=(*current_generation)++;
}

Datatype& Datatype::operator=(const Datatype&)
{
    // XXX:
    // Should probably throw an exception if ref_cnt is > 0 or
    // something.
    generation=(*current_generation)++;
    return *this;
}

Datatype::~Datatype()
{
}

} // End namespace SCIRun

