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
 *  DimensionMismatch.h: Exception to indicate a failed bounds check
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Exceptions/DimensionMismatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace SCIRun {

DimensionMismatch::DimensionMismatch(long value, long expected)
    : value(value), expected(expected)
{
    // Format the message now...
    char buf[120];
    sprintf(buf, "Dimension mismatch, got %ld, expeced %ld", value, expected);
    msg=strdup(buf);
}

DimensionMismatch::DimensionMismatch(const DimensionMismatch& copy)
    : msg(strdup(copy.msg))
{
}
    
DimensionMismatch::~DimensionMismatch()
{
    free(msg);
}

const char* DimensionMismatch::message() const
{
    return msg;
}

const char* DimensionMismatch::type() const
{
    return "DimensionMismatch";
}
} // End namespace SCIRun
