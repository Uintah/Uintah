
/*
 *  ArrayIndexOutOfBounds.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/Exceptions/ArrayIndexOutOfBounds.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using SCICore::Exceptions::ArrayIndexOutOfBounds;


ArrayIndexOutOfBounds::ArrayIndexOutOfBounds(long value, long lower, long upper)
    : value(value), lower(lower), upper(upper)
{
    // Format the message now...
    char buf[120];
    sprintf(buf, "Array index %ld out of range [%ld, %ld)",
	    value, lower, upper);
    msg=strdup(buf);
}

ArrayIndexOutOfBounds::ArrayIndexOutOfBounds(const ArrayIndexOutOfBounds& copy)
   : msg(strdup(copy.msg)), value(copy.value), lower(copy.lower), upper(copy.upper)
{
}
    
ArrayIndexOutOfBounds::~ArrayIndexOutOfBounds()
{
    free(msg);
}

const char* ArrayIndexOutOfBounds::message() const
{
    return msg;
}

const char* ArrayIndexOutOfBounds::type() const
{
    return "SCICore::Exceptions::ArrayIndexOutOfBounds";
}

