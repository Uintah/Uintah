
/*
 *  ArrayIndexOutOfBounds.h: Exception to indicate a failed bounds check
 * $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCICore_Exceptions_ArrayIndexOutOfBounds_h
#define SCICore_Exceptions_ArrayIndexOutOfBounds_h

#include <SCICore/Exceptions/Exception.h>

namespace SCICore {
    namespace Exceptions {
	class ArrayIndexOutOfBounds : public Exception {
	public:
	    ArrayIndexOutOfBounds(long value, long lower, long upper);
	    virtual ~ArrayIndexOutOfBounds();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    long value, lower, upper;
	    char* msg;
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  2000/03/23 10:29:49  sparker
// Part of new exceptions
//
//

