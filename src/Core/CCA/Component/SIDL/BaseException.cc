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
 *  BaseException: Implementation of SIDL.BaseException
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 2003 
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/SIDL/sidl_sidl.h>
#include <Core/Util/NotFinished.h>

using SIDL::BaseException;
std::string BaseException::getMessage(){
    NOT_FINISHED("string BaseException::getMessage()");
    return "";
}

/**
 * Set the message associated with the exception.
 */
void BaseException::setMessage(const std::string &message){
    NOT_FINISHED("string BaseException::setMessage()");
    return;
}

/**
 * Returns formatted string containing the concatenation of all 
 * tracelines.
 */
std::string BaseException::getTrace(){
    NOT_FINISHED("string BaseException::getTrace()");
    return "";
}

/**
 * Adds a stringified entry/line to the stack trace.
 */
void BaseException::addToStackTrace(const std::string &traceline){
    NOT_FINISHED("string BaseException::addToStackTrace()");
    return;
}

/**
 * Formats and adds an entry to the stack trace based on the 
 * file name, line number, and method name.
 */
void BaseException::addToTrace(const std::string &filename, int lineno, const std::string &methodname){
    NOT_FINISHED("string BaseException::addToTrace()");
    return;
}



