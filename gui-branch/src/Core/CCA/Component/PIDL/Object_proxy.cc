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
 *  Object.h: Base class for all PIDL distributed objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/Object_proxy.h>
#include <Core/CCA/Component/PIDL/GlobusError.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <iostream>
#include <string>

using namespace std;

using PIDL::GlobusError;
using PIDL::Object_proxy;
using PIDL::TypeInfo;

Object_proxy::Object_proxy(const Reference& ref)
    : ProxyBase(ref)
{
}

Object_proxy::Object_proxy(const URL& url)
    : ProxyBase(Reference())
{
  std::string s(url.getString());
  d_ref.d_vtable_base=TypeInfo::vtable_methods_start;
  char* str=const_cast<char*>(s.c_str());
  if(int gerr=globus_nexus_attach(str, &d_ref.d_sp)){
    d_ref.d_vtable_base=TypeInfo::vtable_invalid;
    cout << "nexus_attach error\n";
    
    throw GlobusError("nexus_attach", gerr);
  }
  attach_done();
}

Object_proxy::~Object_proxy()
{
}

