
/*
 *  AssertionFailed.h: Generic exception for internal errors
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

#include <SCICore/Exceptions/AssertionFailed.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

using SCICore::Exceptions::AssertionFailed;

AssertionFailed::AssertionFailed(const char* message,
				 const char* file,
				 int lineno)
{
    size_t len = strlen(message)+strlen(file)+100;
    d_message = (char*)malloc(len);
    sprintf(d_message, "%s (file: %s, line: %d)", message, file, lineno);
}

AssertionFailed::AssertionFailed(const AssertionFailed& copy)
    : d_message(strdup(copy.d_message))
{
}

AssertionFailed::~AssertionFailed()
{
    free(d_message);
}

const char* AssertionFailed::message() const
{
    return d_message;
}

const char* AssertionFailed::type() const
{
    return "SCICore::Exceptions::AssertionFailed";
}

//
// $Log$
// Revision 1.2  2000/03/23 20:43:09  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.1  2000/03/23 10:25:40  sparker
// New exception facility - retired old "Exception.h" classes
//
//
