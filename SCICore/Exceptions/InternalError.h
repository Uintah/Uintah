
// $Id$

/*
 *  InternalError.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCICore_Exceptions_InternalError_h
#define SCICore_Exceptions_InternalError_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace SCICore {
    namespace Exceptions {
	class InternalError : public Exception {
	public:
	    InternalError(const std::string&);
	    InternalError(const InternalError&);
	    virtual ~InternalError();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    std::string d_message;
	    InternalError& operator=(const InternalError&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.3  2000/03/23 20:43:10  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.2  2000/03/23 10:25:41  sparker
// New exception facility - retired old "Exception.h" classes
//
// Revision 1.1  1999/08/25 19:03:16  sparker
// Exception base class and generic error class
//
//

