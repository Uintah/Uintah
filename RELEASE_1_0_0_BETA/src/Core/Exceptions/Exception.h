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
 *  Exception.h: Base class for all SCI Exceptions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_Exception_h
#define Core_Exceptions_Exception_h

#include <sci_config.h>

#if USE_SCI_THROW
#define SCI_THROW(exc) do {SCIRun::Exception::sci_throw(exc);throw exc;} while(SCIRun::Exception::alwaysFalse())
#else
#define SCI_THROW(exc) throw exc
#endif

namespace SCIRun {
      class Exception {
      public:
	 Exception();
	 Exception(const Exception&);
	 virtual ~Exception();
	 virtual const char* message() const=0;
	 virtual const char* type() const=0;
	 const char* stackTrace() const {
	    return stacktrace_;
	 }

	 static void sci_throw(const Exception& exc);
	 static bool alwaysFalse();
      protected:
      private:
	 Exception& operator=(const Exception&);
	 const char* stacktrace_;
      };
} // End namespace SCIRun

#endif
