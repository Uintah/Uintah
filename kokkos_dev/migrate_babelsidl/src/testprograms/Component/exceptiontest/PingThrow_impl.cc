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
 *  PingThrow_impl.cc: Test client for PIDL
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   August 2003 
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <sci_defs/mpi_defs.h>
#include <testprograms/Component/exceptiontest/PingThrow_impl.h>
#include <Core/Util/NotFinished.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <sci_mpi.h>

using namespace PingThrow_ns;
using namespace std; 

PingThrow_impl::PingThrow_impl()
{
}

PingThrow_impl::~PingThrow_impl()
{
}

int PingThrow_impl::pingthrow(int arg)
{
  if(arg==1) {
    cout << "Throwing PP\n";
    PPException* ppe = new PPException();
    throw (ppe);
  }
  return arg;
}

int PingThrow_impl::pingthrow2(int arg)
{
  if(arg==1) {
    cout << "Throwing PP2\n";
    PP2Exception* ppe2 = new PP2Exception();
    throw (ppe2);
  }
  return arg;
}

void PingThrow_impl::getOX(OtherThrow::pointer& otptr)
{
  otptr = new OtherThrow_impl();
  otptr->addReference();
}

void PingThrow_impl::donone()
{
}

OtherThrow_impl::OtherThrow_impl()
{
}

OtherThrow_impl::~OtherThrow_impl()
{
}

int OtherThrow_impl::otherthrow()
{
  cout << "Throwing OX\n";
  OtherX* ox = new OtherX();
  throw (ox);
  return 0;
}
