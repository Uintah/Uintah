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
 *  CCAException.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <Core/CCA/spec/sci_sidl.h>
#include <SCIRun/Core/CCAException.h>
#include <Core/Util/NotFinished.h>

extern "C" {
  extern void backtrace_linux( int stack_depth, void *stack_addresses[] );
}

namespace SCIRun {

  CCAException::pointer CCAException::create(const std::string &msg, CCAExceptionType type)
  {
    return CCAException::pointer(new CCAException(msg, type));
  }

  CCAException::CCAException(const std::string &msg, CCAExceptionType type)
    : message(msg), type(type)
  {
    stack_depth = backtrace(stack_addresses, 1024);
    // FIXME [yarden] from SCIRun:
    // Omitting this will cause the framework to
    // segfault when an exception is thrown.
    //addReference();
  }
  
  CCAException::~CCAException()
  {
    show_stack();
    //deleteReference();
  }
  
  // TODO: implement stack trace
  std::string CCAException::getTrace()
  {
    NOT_FINISHED("string .SSIDL.BaseException.getTrace()");
    return std::string(0);
  }
  
  // TODO: implement add functions
  void CCAException::add(const std::string &traceline)
  {
    NOT_FINISHED("void .SSIDL.BaseException.add(in string traceline)");
  }
  
  void CCAException::add(const std::string &filename, int lineno, const std::string &methodname)
  {
    NOT_FINISHED("void .SSIDL.BaseException.add(in string filename, in int lineno, in string methodname)");
  }
  

  void CCAException::show_stack()
  {
    backtrace_linux( stack_depth, stack_addresses );
  }

} // end namespace SCIRun
