
/*
 *  DimensionMismatch.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_DimensionMismatch_h
#define Core_Exceptions_DimensionMismatch_h

#include <Core/Exceptions/Exception.h>

namespace SCIRun {
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
} // End namespace SCIRun

#endif  // Core_Exceptions_DimensionMismatch_h


