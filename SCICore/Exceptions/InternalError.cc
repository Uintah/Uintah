
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

InternalError::~InternalError()
{
}

std::string InternalError::message() const
{
    return d_message;
}

//
// $Log$
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
