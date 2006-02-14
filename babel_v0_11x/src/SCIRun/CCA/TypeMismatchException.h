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

#ifndef SCIRun_CCA_TypeMismatchException_h
#define SCIRun_CCA_TypeMismatchException_h

#include <SCIRun/CCA/CCAException.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <string>

namespace SCIRun {

class TypeMismatchException : virtual public sci::cca::TypeMismatchException
{
public:
  TypeMismatchException(const std::string& msg, sci::cca::Type reqType, sci::cca::Type actualType);
  virtual ~TypeMismatchException();

  virtual sci::cca::CCAExceptionType getCCAExceptionType() { return type; }

  // .sci.cca.Type .sci.cca.TypeMismatchException.getRequestedType()
  virtual inline sci::cca::Type getRequestedType() { return requestType; }

  // .sci.cca.Type .sci.cca.TypeMismatchException.getActualType()
  virtual inline sci::cca::Type getActualType() { return actualType; }

  // string .SSIDL.BaseException.getNote()
  virtual inline std::string
  getNote() { return message; }

  // void .SSIDL.BaseException.setNote(in string message)
  virtual inline void
  setNote(const std::string& message) { this->message = message; }

  // string .SSIDL.BaseException.getTrace()
  virtual std::string getTrace();

  // void .SSIDL.BaseException.add(in string traceline)
  virtual void add(const ::std::string &traceline);

  // void .SSIDL.BaseException.add(in string filename, in int lineno, in string methodname)
  virtual void add(const std::string &filename, int lineno, const std::string &methodname);

private:
  sci::cca::Type requestType;
  sci::cca::Type actualType;
  mutable std::string message;
  mutable sci::cca::CCAExceptionType type;
};

} // end namespace SCIRun

#endif
