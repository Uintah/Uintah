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


#include "CommError.h"
#include <iostream>
#include <sstream>
using namespace std;
using namespace SCIRun;

CommError::CommError(const string& msg, int code)
    : d_msg(msg), d_code(code)
{
    ostringstream o;
    o << d_msg << "(return code=" << d_code << ")";
    d_msg = o.str();
    cerr << "CommError: " << d_msg << '\n';
}

CommError::CommError(const CommError& copy)
    : d_msg(copy.d_msg), d_code(copy.d_code)
{
}

CommError::~CommError()
{
}

const char* CommError::message() const
{
    return d_msg.c_str();
}

const char* CommError::type() const
{
    return "PIDL::CommError";
}

