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
