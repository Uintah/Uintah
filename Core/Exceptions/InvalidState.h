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
 *  InvalidState.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_InvalidState_h
#define Core_Exceptions_InvalidState_h

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class InvalidState : public Exception {
  public:
    InvalidState(const std::string&);
    InvalidState(const InvalidState&);
    virtual ~InvalidState();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string message_;
    InvalidState& operator=(const InvalidState&);
  };
} // End namespace SCIRun

#endif


