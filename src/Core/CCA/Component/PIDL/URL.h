
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
/**************************************
 
CLASS
   URL
   
KEYWORDS
   URL, PIDL
   
DESCRIPTION
   An encapsulation of a URL for PIDL.
****************************************/
	class URL {
	public:
	    //////////
	    // Create a URL from the specified protocol, hostname,
	    // port number and spec.  This will be of the form:
	    // protocol://hostname:portnumber/spec
	    URL(const std::string& protocol, const std::string& hostname,
		int portnumber, const std::string& spec);

	    //////////
	    // Create the URL from the specified string.  May throw
	    // MalformedURL.
	    URL(const std::string& url);

	    //////////
	    // Copy the URL
	    URL(const URL&);

	    //////////
	    // Destructor
	    ~URL();

	    //////////
	    // Copy the URL.
	    URL& operator=(const URL&);

	    //////////
	    // Return the entire URL as a single string.
	    std::string getString() const;

	    //////////
	    // Return the protocol string.
	    std::string getProtocol() const;

	    //////////
	    // Return the hostname.
	    std::string getHostname() const;

	    //////////
	    // Return the port number (0 if one was not specified in
	    // the URL.
	    int getPortNumber() const;

	    //////////
	    // Return the URL spec (i.e. the filename part).  Returns
	    // an empty string if a spec was not specified in the URL.
	    std::string getSpec() const;

	protected:
	private:
	    //////////
	    std::string d_protocol;

	    //////////
	    std::string d_hostname;

	    //////////
	    int d_portno;

	    //////////
	    std::string d_spec;
	};
    }
}

#endif

//
// $Log$
// Revision 1.3  1999/09/24 20:03:39  sparker
// Added cocoon documentation
//
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
