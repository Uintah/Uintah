
// $Id$

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

#ifndef SCICore_Exceptions_AssertionFailed_h
#define SCICore_Exceptions_AssertionFailed_h

#include <SCICore/Exceptions/Exception.h>

namespace SCICore {
    namespace Exceptions {
	class AssertionFailed : public Exception {
	public:
	    AssertionFailed(const char* msg,
			    const char* file,
			    int line);
	    virtual ~AssertionFailed();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    char* d_message;
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  2000/03/23 10:25:40  sparker
// New exception facility - retired old "Exception.h" classes
//
//

