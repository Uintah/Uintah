
/* REFERENCED */
static char *id="$Id$";

/*
 *  InternalError.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Exceptions/InternalError.h>

using SCICore::Exceptions::InternalError;

InternalError::InternalError(const std::string& message)
    : d_message(message)
{
}

InternalError::InternalError(const InternalError& copy)
    : d_message(copy.d_message)
{
}

InternalError::~InternalError()
{
}

const char* InternalError::message() const
{
    return d_message.c_str();
}

const char* InternalError::type() const
{
    return "SCICore::Exceptions::InternalError";
}

//
// $Log$
// Revision 1.4  2000/03/23 20:43:10  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.3  2000/03/23 10:25:41  sparker
// New exception facility - retired old "Exception.h" classes
//
// Revision 1.2  1999/08/31 08:59:04  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/25 19:03:16  sparker
// Exception base class and generic error class
//
//
