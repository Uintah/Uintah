
/*
 *  ArrayIndexOutOfBounds.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_ArrayIndexOutOfBounds_h
#define Core_Exceptions_ArrayIndexOutOfBounds_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
	class ArrayIndexOutOfBounds : public Exception {
	public:
	    ArrayIndexOutOfBounds(long value, long lower, long upper);
	    ArrayIndexOutOfBounds(const ArrayIndexOutOfBounds&);
	    virtual ~ArrayIndexOutOfBounds();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    long value, lower, upper;
	    char* msg;

	    ArrayIndexOutOfBounds& operator=(const ArrayIndexOutOfBounds);
	};
} // End namespace SCIRun

#endif


