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
 *  MalformedURL.h: Base class for PIDL Exceptions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_MalformedURL_h
#define CCA_PIDL_MalformedURL_h

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
/**************************************
 
CLASS
   MalformedURL
   
KEYWORDS
   Exception, Error, PIDL, URL
   
DESCRIPTION
   Exception class for an unintelligible URL.  This results from
   a syntax error in the URL.  See InvalidReference for a properly
   formed URL that doesn't map to a valid object.

****************************************/

class MalformedURL : public SCIRun::Exception {

  public:
    //////////
    // Contruct the object, giving the offending URL and an
    // explanation of the error
    MalformedURL(const std::string& url, const std::string& error);

    //////////
    // Copy ctor
    MalformedURL(const MalformedURL&);

    //////////
    // Destructor
    virtual ~MalformedURL();

    //////////
    // Return a human readable explanation
    virtual const char* message() const;

    //////////
    // Return the name of this class
    virtual const char* type() const;

  protected:
  private:
    //////////
    // The offending URL
    std::string d_url;

    //////////
    // The error explanation
    std::string d_error;

    //////////
    // The "complete" message
    std::string d_msg;

    MalformedURL& operator=(const MalformedURL&);
  };
} // End namespace SCIRun

#endif

