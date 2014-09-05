
/*
 *  PIDLException.h: Base class for PIDL Exceptions
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Component/PIDL/GlobusError.h>

using Component::PIDL::PIDLException;

PIDLException::PIDLException()
{
}

PIDLException::PIDLException(const PIDLException&)
{
}


//
// $Log$
// Revision 1.1  2000/03/23 20:43:07  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
//
