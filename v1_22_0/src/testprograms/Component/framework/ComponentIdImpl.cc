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

#include <Core/Thread/Mutex.h>
#include <Core/Thread/AtomicCounter.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>

#include <sstream>
#include <iostream>

#include <stdio.h>

namespace sci_cca {

using SCIRun::AtomicCounter;
using SCIRun::Mutex;
using std::cerr;
using std::ostringstream;

static AtomicCounter* generation = 0;
static Mutex lock("Component generation counter initialization lock");

ComponentIdImpl::ComponentIdImpl()
{
}

void
ComponentIdImpl::init( const string &host, const string &program )
{
  if (!generation) {
    lock.lock();
    if(!generation)
      generation = new AtomicCounter("Component generation counter", 1);
    lock.unlock();
  }

  number_ = (*generation)++;

  host_ = host;
  program_ = program;

  ostringstream tmp;
  tmp << host_ << "/" << program_ << "-" << number_; 
  
  id_ = tmp.str();
}

ComponentIdImpl::~ComponentIdImpl()
{
  cerr << "ComponenetIdImpl " << id_ << " destructor\n";
}

string
ComponentIdImpl::toString()
{
  return id_;
}

string
ComponentIdImpl::fullString()
{
  char full[ 1024 ];
  
  sprintf( full, "%s, %s, %d, %s", host_.c_str(), program_.c_str(), number_,
	   id_.c_str() );

  return string( full );
}

} // namespace sci_cca

