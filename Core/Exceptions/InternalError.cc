
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

SCICore::Exceptions::InternalError::InternalError(const std::string& message)
    : d_message(message)
{
}

std::string SCICore::Exceptions::InternalError::message() const
{
    return d_message;
}

//
// $Log$
// Revision 1.1  1999/08/25 19:03:16  sparker
// Exception base class and generic error class
//
//
