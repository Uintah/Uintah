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
 *  URL.h: Abstraction for a URL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_URL_h
#define CCA_PIDL_URL_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
/**************************************
 
CLASS
   URL
   
KEYWORDS
   URL, PIDL
   
DESCRIPTION
   An encapsulation of a URL for PIDL.
****************************************/
  class URL {
  public:
    //////////
    // Create a URL from the specified protocol, hostname,
    // port number and spec.  This will be of the form:
    // protocol://hostname:portnumber/spec
    URL(const std::string& protocol, const std::string& hostname,
	int portnumber, const std::string& spec);

    //////////
    // Create the URL from the specified string.  May throw
    // MalformedURL.
    URL(const std::string& url);

    //////////
    // Copy the URL
    URL(const URL&);

    //////////
    // Destructor
    ~URL();

    //////////
    // Return the entire URL as a single string.
    std::string getString() const;

    //////////
    // Return the protocol string.
    std::string getProtocol() const;

    //////////
    // Return the hostname.
    std::string getHostname() const;

    //////////
    // Return the port number (0 if one was not specified in
    // the URL.
    int getPortNumber() const;

    //////////
    // Return the URL spec (i.e. the filename part).  Returns
    // an empty string if a spec was not specified in the URL.
    std::string getSpec() const;

    long getIP();

  protected:
  private:
    //////////
    std::string d_protocol;

    //////////
    std::string d_hostname;

    //////////
    int d_portno;

    //////////
    std::string d_spec;
  };
} // End namespace SCIRun

#endif

