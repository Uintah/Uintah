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
#include <string>

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

