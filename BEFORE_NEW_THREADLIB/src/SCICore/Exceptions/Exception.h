
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

#include <string>

namespace SCICore {
    namespace Exceptions {
	class Exception {
	public:
	    virtual std::string message() const=0;
	protected:
	private:
	};
    }
}

#endif
