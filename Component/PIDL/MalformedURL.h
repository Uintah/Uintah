
/*
 *  MalformedURL.h: Base class for PIDL Exceptions
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

#ifndef Component_PIDL_MalformedURL_h
#define Component_PIDL_MalformedURL_h

#include <SCICore/Exceptions/Exception.h>

namespace Component {
    namespace PIDL {
	class MalformedURL : public SCICore::Exceptions::Exception {
	public:
	    MalformedURL(const std::string& url, const std::string& error);
	    virtual ~MalformedURL();
	    virtual std::string message() const;
	protected:
	private:
	    std::string d_url;
	    std::string d_error;
	};
    }
}

#endif

//
// $Log$
// Revision 1.3  1999/09/17 05:08:07  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:00  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:46  sparker
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
