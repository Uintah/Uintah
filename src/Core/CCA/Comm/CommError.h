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



#ifndef CCA_Comm_CommError_h
#define CCA_Comm_CommError_h

#include <Core/Exceptions/Exception.h>
#include <string>

namespace SCIRun {
  /**************************************
 
  CLASS
     CommError
   
  KEYWORDS
     Exception, Error, PIDL
   
  DESCRIPTION
     Exception class for communication functions.  An unhandled negative return
     code from a socket function will get mapped to this exception.  The
     message is a description of the call, and the code is the result
     returned from the particular communication class.

  ****************************************/
  class CommError : public SCIRun::Exception {
  public:
    //////////
    // Construct the exception with the given reason and the
    // return code from the Communication class
    CommError(const std::string& msg, int code);

    //////////
    // Copy ctor
    CommError(const CommError&);

    //////////
    // Destructor
    virtual ~CommError();

    //////////
    // An explanation message, containing the msg string and the
    // return code passed into the constructor.
    const char* message() const;

    //////////
    // The name of this class
    const char* type() const;
  protected:
  private:
    //////////
    // The explanation string (usually the name of the offending
    // call).
    std::string d_msg;

    //////////
    // The globus error code.
    int d_code;

    CommError& operator=(const CommError&);
  };
}

#endif
