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
