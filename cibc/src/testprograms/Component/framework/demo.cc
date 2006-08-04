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




#include <unistd.h>
#include <iostream>
#include <Core/Thread/Thread.h>
#include <testprograms/Component/framework/cca.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <Core/CCA/PIDL/PIDL.h>

using namespace sci_cca;
using namespace SCIRun;

int
main( int argc, char *argv[] )
{
  bool stop = argc == 1;
  PIDL::PIDL::initialize();

  ComponentImpl *cp = argc == 1 ? 0 : new ComponentImpl;
  
  //  delete cp1;
  if ( stop ) {
    Semaphore wait("main wait",0);
    wait.down();
  }

  //delete cp;
  cerr << "main done\n";

  // never reached
  return 0;
}

