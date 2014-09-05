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
 *  Object_proxy.h
 *
 *  
 */

#include <Core/CCA/Component/PIDL/Object_proxy.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <iostream>
#include <string>

using namespace std;
using namespace SCIRun;

Object_proxy::Object_proxy(const Reference& ref)
    : ProxyBase(ref)
{
}

Object_proxy::Object_proxy(const URL& url)
  : ProxyBase(*(new Reference()))
{
  d_ref.chan->openConnection(url);
  d_ref.d_vtable_base=TypeInfo::vtable_methods_start;

  attach_done();
}

Object_proxy::~Object_proxy()
{
}















