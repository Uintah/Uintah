
// $Id$

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

#ifndef SCICore_Exceptions_Exception_h
#define SCICore_Exceptions_Exception_h

#include <sci_config.h>

#if USE_SCI_THROW
#define SCI_THROW(exc) do {SCICore::Exceptions::Exception::sci_throw(exc);throw exc;} while(SCICore::Exceptions::Exception::alwaysFalse())
#else
#define SCI_THROW(exc) throw exc
#endif

namespace SCICore {
    namespace Exceptions {
	class Exception {
	public:
	    Exception();
	    virtual ~Exception();
	    virtual const char* message() const=0;
	    virtual const char* type() const=0;

	    static void sci_throw(const Exception& exc);
	    static bool alwaysFalse();
	protected:
	private:
	    Exception(const Exception&);
	    Exception& operator=(const Exception&);
	};
    }
}

#endif
