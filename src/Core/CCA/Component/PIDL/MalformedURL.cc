
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

MalformedURL::MalformedURL(const std::string& url,
					 const std::string& error)
    : d_url(url), d_error(error)
{
}

MalformedURL::~MalformedURL()
{
}

std::string MalformedURL::message() const
{
    return "Malformed URL: "+d_url+" ("+d_error+")";
}

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
