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
 *  FileNotFound.h: Generic exception for internal errors
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Exceptions/FileNotFound.h>

namespace SCIRun {

FileNotFound::FileNotFound(const std::string& message)
    : message_(message)
{
}

FileNotFound::FileNotFound(const FileNotFound& copy)
    : message_(copy.message_)
{
}

FileNotFound::~FileNotFound()
{
}

const char* FileNotFound::message() const
{
    return message_.c_str();
}

const char* FileNotFound::type() const
{
    return "FileNotFound";
}

} // End namespace SCIRun
