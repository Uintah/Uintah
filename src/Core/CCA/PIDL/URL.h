/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

#include <string>

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

