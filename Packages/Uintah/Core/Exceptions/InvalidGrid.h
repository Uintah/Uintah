
// $Id$

/*
 *  InvalidGrid.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Uintah_Exceptions_InvalidGrid_h
#define Uintah_Exceptions_InvalidGrid_h

#include <SCICore/Exceptions/Exception.h>
#include <string>

namespace Uintah {
    namespace Exceptions {
	class InvalidGrid : public SCICore::Exceptions::Exception {
	public:
	    InvalidGrid(const std::string& msg);
	    InvalidGrid(const InvalidGrid&);
	    virtual ~InvalidGrid();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    std::string d_msg;
	    InvalidGrid& operator=(const InvalidGrid&);
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

