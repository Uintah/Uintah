
// $Id$

/*
 *  InvalidValue.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Uintah_Exceptions_InvalidValue_h
#define Uintah_Exceptions_InvalidValue_h

#include <Uintah/Exceptions/ProblemSetupException.h>
#include <string>

namespace Uintah {
    namespace Exceptions {
	class InvalidValue : public ProblemSetupException {
	public:
	    InvalidValue(const std::string&);
	    InvalidValue(const InvalidValue&);
	    virtual ~InvalidValue();
	    virtual const char* type() const;
	protected:
	private:
	    InvalidValue& operator=(const InvalidValue&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.4  2000/04/11 07:10:44  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
//

