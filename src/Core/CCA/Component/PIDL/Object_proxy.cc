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
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <iostream>
#include <string>

using namespace std;
using namespace SCIRun;

Object_proxy::Object_proxy(const Reference& ref)
    : ProxyBase(ref)
{
  rm.localSize = 1;
  rm.localRank = 0;
}

Object_proxy::Object_proxy(const URL& url)
  : ProxyBase(*(new Reference()))
{
  rm.d_ref[0].chan->openConnection(url);
  rm.d_ref[0].d_vtable_base=TypeInfo::vtable_methods_start;
  rm.localSize = 1;
  rm.localRank = 0;
}

Object_proxy::Object_proxy(const int urlc, const URL urlv[], int mysize, int myrank)
{
  for(int i=0; i < urlc; i++) {
    Reference *ref = new Reference();
    ref->chan->openConnection(urlv[i]);
    ref->d_vtable_base=TypeInfo::vtable_methods_start;
    rm.insertReference(*ref);
  }
  rm.localSize = mysize;
  rm.localRank = myrank;
  //Exchange parallel IDs among processes
  rm.intracomm = PIDL::getIntraComm();
}

Object_proxy::Object_proxy(const std::vector<URL>& urlv, int mysize, int myrank)
{
  std::vector<URL>::const_iterator iter = urlv.begin();
  for(unsigned int i=0; i < urlv.size(); i++, iter++) {
    Reference *ref = new Reference();
    ref->chan->openConnection(*iter);
    ref->d_vtable_base=TypeInfo::vtable_methods_start;
    rm.insertReference(*ref);
  }
  rm.localSize = mysize;
  rm.localRank = myrank;
  //Exchange parallel IDs among processes
  rm.intracomm = PIDL::getIntraComm();
}

Object_proxy::~Object_proxy()
{
  ::std::cout << "~Object_proxy\n";
  if(rm.localSize > 1)
    delete (rm.intracomm);
}















