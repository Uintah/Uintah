
/*
 *  URL.h: Abstraction for a URL
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

#ifndef Component_PIDL_URL_h
#define Component_PIDL_URL_h

#include <string>

namespace Component {
    namespace PIDL {
	class URL {
	public:
	    URL(const std::string&, const std::string&, int portno,
		const std::string&);
	    URL(const std::string&);
	    URL(const URL&);
	    ~URL();

	    URL& operator=(const URL*);

	    std::string getString() const;
	    std::string getProtocol() const;
	    std::string getHostname() const;
	    int getPortNumber() const;
	    std::string getSpec() const;

	protected:
	private:
	    std::string d_protocol;
	    std::string d_hostname;
	    int d_portno;
	    std::string d_spec;
	};
    }
}

#endif

//
// $Log$
// Revision 1.2  1999/09/17 05:08:11  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.1  1999/08/30 17:39:49  sparker
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
