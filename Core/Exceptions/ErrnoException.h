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
 *  ErrnoException.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Core_Exceptions_ErrnoException_h
#define Core_Exceptions_ErrnoException_h

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class ErrnoException : public Exception {
public:
  ErrnoException(const std::string&, int err);
  ErrnoException(const ErrnoException&);
  virtual ~ErrnoException();
  virtual const char* message() const;
  virtual const char* type() const;
	 
  int getErrno() const;
protected:
private:
  std::string message_;
  int errno_;

  ErrnoException& operator=(const ErrnoException&);
};

} // End namespace SCIRun

#endif
