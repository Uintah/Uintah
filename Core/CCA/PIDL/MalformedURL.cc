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
 *  MalformedURL.h: Base class for PIDL Exceptions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include "MalformedURL.h"
using namespace SCIRun;

MalformedURL::MalformedURL(const std::string& url,
			   const std::string& error)
    : d_url(url), d_error(error)
{
    d_msg = "Malformed URL: "+d_url+" ("+d_error+")"; 
}

MalformedURL::MalformedURL(const MalformedURL& copy)
    : d_url(copy.d_url), d_error(copy.d_error)
{
}

MalformedURL::~MalformedURL()
{
}

const char* MalformedURL::message() const
{
    return d_msg.c_str();
}

const char* MalformedURL::type() const
{
    return "SCIRun::MalformedURL";
}

