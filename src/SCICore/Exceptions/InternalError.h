
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

namespace SCICore {
    namespace Exceptions {
	class InternalError : public Exception {
	public:
	    InternalError(const std::string&);
	    virtual ~InternalError();
	    virtual std::string message() const;
	protected:
	private:
	    std::string d_message;
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  1999/08/25 19:03:16  sparker
// Exception base class and generic error class
//
//

