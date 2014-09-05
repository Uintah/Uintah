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
 *  CCAException.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_CCAException_h
#define SCIRun_Framework_CCAException_h

#include <Core/CCA/spec/sci_sidl.h>
#include <Core/CCA/SSIDL/sidl_sidl.h>
#include <string>

namespace SCIRun {

/**
 * \class CCAException
 *
 * An exception object for the CCA Component model.
 *
 */
class CCAException : public sci::cca::CCAException
{
public:
    CCAException(const std::string &msg,
                 sci::cca::CCAExceptionType type = sci::cca::Nonstandard);
    virtual ~CCAException();

    // .sci.cca.CCAExceptionType .sci.cca.CCAException.getCCAExceptionType()
    virtual inline sci::cca::CCAExceptionType
    getCCAExceptionType() { return type; }

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
    mutable std::string message;
    mutable sci::cca::CCAExceptionType type;
};

} // end namespace SCIRun

#endif
