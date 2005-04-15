
/*
 *  ArrayIndexOutOfBounds.h: Exception to indicate a failed bounds check
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

//
// $Log$
// Revision 1.2.2.3  2000/10/26 17:51:52  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.3  2000/09/25 17:58:57  sparker
// Do not call variables errno due to #defines on some systems (linux)
// Correctly implemented copy CTORs
//
// Revision 1.2  2000/03/23 20:43:09  sparker
// Added copy ctor to all exception classes (for Linux/g++)
//
// Revision 1.1  2000/03/23 10:25:40  sparker
// New exception facility - retired old "Exception.h" classes
//
//
