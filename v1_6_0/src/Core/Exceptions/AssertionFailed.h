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
 *  AssertionFailed.h: Exception for a failed assertion.  Note - this
 *    version takes only a char*.  There is a FancyAssertionFailed that
 *    takes std::string.  This is done to prevent include file pollution
 *    with std::string
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_AssertionFailed_h
#define Core_Exceptions_AssertionFailed_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
class AssertionFailed : public Exception {
public:
  AssertionFailed(const char* msg,
		  const char* file,
		  int line);
  AssertionFailed(const AssertionFailed&);
  virtual ~AssertionFailed();
  virtual const char* message() const;
  virtual const char* type() const;
protected:
private:
  char* message_;
  AssertionFailed& operator=(const AssertionFailed&);
};
} // End namespace SCIRun

#endif


