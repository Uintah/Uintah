
/*
 *  Reference.h: A serializable "pointer" to an object
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

#include <Component/PIDL/Reference.h>
#include <Component/PIDL/GlobusError.h>
#include <Component/PIDL/Startpoint.h>
#include <Component/PIDL/URL.h>
#include <globus_nexus.h>

using Component::PIDL::Reference;

Reference::Reference()
{
    d_startpoint=0;
}

Reference::Reference(const Reference& copy)
{
    if(copy.d_startpoint){
	d_startpoint=new Startpoint;
	if(int gerr=globus_nexus_startpoint_copy(&d_startpoint->d_sp,
						 &copy.d_startpoint->d_sp))
	    throw GlobusError("startpoint_copy", gerr);
    } else {
	d_startpoint=0;
    }
}

Reference::Reference(const URL& url)
{
    std::string s(url.getString());
    d_startpoint=new Startpoint;
    char* str=const_cast<char*>(s.c_str());
    if(int gerr=globus_nexus_attach(str, &d_startpoint->d_sp)){
	delete d_startpoint;
	d_startpoint=0;
	throw GlobusError("nexus_attach", gerr);
    }
}

Reference::~Reference()
{
    if(d_startpoint){
	if(int gerr=globus_nexus_startpoint_destroy(&d_startpoint->d_sp))
	    throw GlobusError("startpoint_destroy", gerr);
	delete d_startpoint;
    }
}

Reference& Reference::operator=(const Reference& copy)
{
    if(this == &copy)
	return *this;
    if(d_startpoint){
	if(int gerr=globus_nexus_startpoint_destroy(&d_startpoint->d_sp))
	    throw GlobusError("startpoint_destroy", gerr);
    }
    if(copy.d_startpoint) {
	if(!d_startpoint)
	    d_startpoint=new Startpoint;
	if(int gerr=globus_nexus_startpoint_copy(&d_startpoint->d_sp,
					      &copy.d_startpoint->d_sp))
	    throw GlobusError("startpoint_copy", gerr);
    } else {
	if(d_startpoint)
	    delete d_startpoint;
    }
    return *this;
}
//
// $Log$
// Revision 1.2  1999/08/31 08:59:02  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:47  sparker
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
