
/*
 *  InvalidReference.h: A "bad" reference to an object
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_PIDL_InvalidReference_h
#define Component_PIDL_InvalidReference_h

#include <Component/PIDL/PIDLException.h>

namespace Component {
    namespace PIDL {
	class InvalidReference : public PIDLException {
	public:
	    InvalidReference(const std::string&);
	    virtual ~InvalidReference();
	    std::string message() const;
	protected:
	private:
	    std::string d_msg;
	};
    }
}

#endif
//
// $Log$
// Revision 1.1  1999/08/30 17:39:45  sparker
// Updates to configure script:
//  rebuild configure if configure.in changes (Bug #35)
//  Fixed rule for rebuilding Makefile from Makefile.in (Bug #36)
//  Rerun configure if configure changes (Bug #37)
//  Don't build Makefiles for modules that aren't --enabled (Bug #49)
//  Updated Makfiles to build sidl and Component if --enable-parallel
// Updates to sidl code to compile on linux
// Imported PIDL code
// Created top-level Component directory
// Added ProcessManager class - a simpler interface to fork/exec (not finished)
//
//
