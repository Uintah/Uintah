
// $Id$

/*
 *  ParameterNotFound.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Uintah_Exceptions_ParameterNotFound_h
#define Uintah_Exceptions_ParameterNotFound_h

#include <Uintah/Exceptions/ProblemSetupException.h>
#include <string>

namespace Uintah {
    namespace Exceptions {
	class ParameterNotFound : public ProblemSetupException {
	public:
	    ParameterNotFound(const std::string&);
	    ParameterNotFound(const ParameterNotFound&);
	    virtual ~ParameterNotFound();
	    virtual const char* type() const;
	protected:
	private:
	    ParameterNotFound& operator=(const ParameterNotFound&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.1  2000/04/12 22:57:47  sparker
// Added new exception classes
//
//

