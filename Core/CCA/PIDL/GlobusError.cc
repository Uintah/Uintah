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
 *  GlobusError.h: Erros due to globus calls
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/PIDL/GlobusError.h>
#include <sstream>
using namespace SCIRun;

GlobusError::GlobusError(const std::string& msg, int code)
    : d_msg(msg), d_code(code)
{
    std::ostringstream o;
    o << d_msg << "(return code=" << d_code << ")";
    d_msg = o.str();
}

GlobusError::GlobusError(const GlobusError& copy)
    : d_msg(copy.d_msg), d_code(copy.d_code)
{
}

GlobusError::~GlobusError()
{
}

const char* GlobusError::message() const
{
    return d_msg.c_str();
}

const char* GlobusError::type() const
{
    return "SCIRun::GlobusError";
}

