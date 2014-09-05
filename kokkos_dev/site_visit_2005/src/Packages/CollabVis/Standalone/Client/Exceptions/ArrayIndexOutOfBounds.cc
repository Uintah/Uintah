/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Exceptions/ArrayIndexOutOfBounds.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace SCIRun {

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
  : value(copy.value), lower(copy.lower), upper(copy.upper), msg(strdup(copy.msg))
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
    return "ArrayIndexOutOfBounds";
}

} // End namespace SCIRun
