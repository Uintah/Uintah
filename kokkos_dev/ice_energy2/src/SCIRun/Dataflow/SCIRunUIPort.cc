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
 *  SCIRunUIPort.cc: CCA-style Interface to old TCL interfaces
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <SCIRun/Dataflow/SCIRunUIPort.h>
#include <SCIRun/Dataflow/SCIRunComponentInstance.h>
#include <Dataflow/Network/Module.h>
#include <iostream>

namespace SCIRun {

SCIRunUIPort::SCIRunUIPort(SCIRunComponentInstance* component)
  : component(component)
{
}

SCIRunUIPort::~SCIRunUIPort()
{
}

int SCIRunUIPort::ui()
{
  Module* module = component->getModule();
  module->popup_ui();
  //std::cerr << "Warning: need return correct value (0 success, -1 fatal error, other values for other errors !" << std::endl;
  // Module::popup_ui has void return value
  // TODO: report success or failure of function call
  return 0;
}

} // end namespace SCIRun
