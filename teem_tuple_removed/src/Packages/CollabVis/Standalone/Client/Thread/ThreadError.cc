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
 *  ThreadError: Exception class for unusual errors in the thread library
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: August 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Thread/ThreadError.h>
namespace SCIRun {


ThreadError::ThreadError(const std::string& message)
    : message_(message)
{
}

ThreadError::ThreadError(const ThreadError& copy)
    : message_(copy.message_)
{
}

ThreadError::~ThreadError()
{
}

const char*
ThreadError::message() const
{
    return message_.c_str();
}

const char*
ThreadError::type() const
{
    return "ThreadError";
}


} // End namespace SCIRun
