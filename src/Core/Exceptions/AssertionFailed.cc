
/*
 *  AssertionFailed.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Exceptions/AssertionFailed.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

namespace SCIRun {

AssertionFailed::AssertionFailed(const char* message,
				 const char* file,
				 int lineno)
{
    size_t len = strlen(message)+strlen(file)+100;
    message_ = (char*)malloc(len);
    sprintf(message_, "%s (file: %s, line: %d)", message, file, lineno);
}

AssertionFailed::AssertionFailed(const AssertionFailed& copy)
    : message_(strdup(copy.message_))
{
}

AssertionFailed::~AssertionFailed()
{
    free(message_);
}

const char* AssertionFailed::message() const
{
    return message_;
}

const char* AssertionFailed::type() const
{
    return "AssertionFailed";
}

} // End namespace SCIRun
