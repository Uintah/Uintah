

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

#ifndef Core_Exceptions_InternalError_h
#define Core_Exceptions_InternalError_h

#include <Core/Exceptions/Exception.h>
#include <string>

namespace SCIRun {
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
} // End namespace SCIRun

#endif


