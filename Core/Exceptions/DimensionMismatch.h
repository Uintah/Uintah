
/*
 *  DimensionMismatch.h: Exception to indicate a failed bounds check
 * $Id$
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCICore_Exceptions_DimensionMismatch_h
#define SCICore_Exceptions_DimensionMismatch_h

#include <SCICore/Exceptions/Exception.h>

namespace SCICore {
    namespace Exceptions {
	class DimensionMismatch : public Exception {
	public:
	    DimensionMismatch(long value, long expected);
	    DimensionMismatch(const DimensionMismatch&);
	    virtual ~DimensionMismatch();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    long value, expected;
	    char* msg;

	    DimensionMismatch& operator=(const DimensionMismatch);
	};
    }
}

#endif  // SCICore_Exceptions_DimensionMismatch_h


