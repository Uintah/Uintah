/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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

#ifndef Framework_UnitTests_PIDLBootstrap_h
#define Framework_UnitTests_PIDLBootstrap_h

#include <Core/Util/Environment.h>
#include <Core/CCA/PIDL/PIDL.h>

// PIDL has it's own set of unit tests (see Core/CCA/PIDL/UnitTests)
// to verify the correctness and stability of the following functions:
void bootstrapPIDL()
{
  // create environment for tests
  SCIRun::create_sci_environment(0, 0);
  SCIRun::PIDL::initialize();
  SCIRun::PIDL::isfrwk = true;
  // Should initialize PRMI too:
  // all threads in the framework share the same invocation id.
  SCIRun::PRMI::setInvID(SCIRun::ProxyID(1,0));
}

void cleanupPIDL()
{
  SCIRun::PIDL::finalize();
}


#endif
