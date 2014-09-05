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

#include <Framework/Internal/EventServiceException.h>
#include <Core/Util/NotFinished.h>

namespace SCIRun {

// TODO: this code allows empty strings as message types.
// Check CCA spec to see if this is OK (probably not).
EventServiceException::EventServiceException(const std::string &msg, sci::cca::CCAExceptionType type)
  : message(msg), type(type)
{
  // If the message string supplied to the constructor is empty,
  // set a default message.
  // This could be an exception, however an empty message is not a serious
  // errror, so let's not bog down the framework with unnecessary error handling.
  if (message.empty()) {
    message = "EventServiceException: no message was created by the caller.";
  }

  // Omitting this will cause the framework to
  // segfault when an exception is thrown.
  addReference();
}

EventServiceException::~EventServiceException()
{
  deleteReference();
}


// TODO: implement stack trace
std::string EventServiceException::getTrace()
{
  NOT_FINISHED("string .SSIDL.BaseException.getTrace()");
  return std::string();
}

// TODO: implement add functions
void EventServiceException::add(const std::string &traceline)
{
  NOT_FINISHED("void .SSIDL.BaseException.add(in string traceline)");
}

void EventServiceException::add(const std::string &filename, int lineno, const std::string &methodname)
{
  NOT_FINISHED("void .SSIDL.BaseException.add(in string filename, in int lineno, in string methodname)");
}

}
