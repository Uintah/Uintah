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



#include <Core/CCA/SSIDL/sidl_sidl.h>
#include <Core/Util/NotFinished.h>

using SSIDL::BaseInterface;
using SSIDL::BaseClass;
using SSIDL::BaseException;
using SSIDL::SIDLException;


std::string
SIDLException::getNote()
{
  NOT_FINISHED("string SIDLException::getNote()");
  return "";
}

/**
 * Set the message associated with the exception.
 */
void
SIDLException::setNote(const std::string& /*message*/)
{
  NOT_FINISHED("string SIDLException::setNote()");
  return;
}

/**
 * Returns formatted string containing the concatenation of all 
 * tracelines.
 */
std::string
SIDLException::getTrace()
{
  NOT_FINISHED("string SIDLException::getTrace()");
  return "";
}

/**
 * Adds a stringified entry/line to the stack trace.
 */
void
// how to represent [Line] in C++: [Line] is extension for
// method overloading
//
// f_addLine???
//SIDLException::addLine(const string &/*traceline*/) ???
SIDLException::add(const std::string& /*traceline*/)
{
  NOT_FINISHED("string SIDLException::add(const std::string& traceline)");
  return;
}

/**
 * Formats and adds an entry to the stack trace based on the 
 * file name, line number, and method name.
 */
void
SIDLException::add(const std::string& /*filename*/,
			   int /*lineno*/, 
               const std::string& /*methodname*/)
{
  NOT_FINISHED("string SIDLException::add(const std::string& filename, int lineno, const std::string& methodname)");
  return;
}

