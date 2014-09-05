
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

#include <Component/PIDL/MalformedURL.h>

using Component::PIDL::MalformedURL;

MalformedURL::MalformedURL(const std::string& url,
			   const std::string& error)
    : d_url(url), d_error(error)
{
    d_msg = "Malformed URL: "+d_url+" ("+d_error+")"; 
}

MalformedURL::MalformedURL(const MalformedURL& copy)
    : d_url(copy.d_url), d_error(copy.d_error)
{
}

MalformedURL::~MalformedURL()
{
}

const char* MalformedURL::message() const
{
    return d_msg.c_str();
}

const char* MalformedURL::type() const
{
    return "Component::PIDL::MalformedURL";
}

//
// $Log$
// Revision 1.4  2000/03/23 20:43:07  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.3  2000/03/23 10:27:36  sparker
// Added "name" method to match new Exception base class
//
// Revision 1.2  1999/08/31 08:59:00  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
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
