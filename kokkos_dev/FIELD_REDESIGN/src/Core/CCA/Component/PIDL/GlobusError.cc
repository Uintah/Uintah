
/*
 *  GlobusError.h: Erros due to globus calls
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

#include <Component/PIDL/GlobusError.h>
#include <sstream>

using Component::PIDL::GlobusError;

GlobusError::GlobusError(const std::string& msg, int code)
    : d_msg(msg), d_code(code)
{
    std::ostringstream o;
    o << d_msg << "(return code=" << d_code << ")";
    d_msg = o.str();
}

GlobusError::GlobusError(const GlobusError& copy)
    : d_msg(copy.d_msg), d_code(copy.d_code)
{
}

GlobusError::~GlobusError()
{
}

const char* GlobusError::message() const
{
    return d_msg.c_str();
}

const char* GlobusError::type() const
{
    return "Component::PIDL::GlobusError";
}

//
// $Log$
// Revision 1.5  2000/03/23 20:43:06  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.4  2000/03/23 10:27:35  sparker
// Added "name" method to match new Exception base class
//
// Revision 1.3  1999/09/17 05:08:07  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:00  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:44  sparker
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
