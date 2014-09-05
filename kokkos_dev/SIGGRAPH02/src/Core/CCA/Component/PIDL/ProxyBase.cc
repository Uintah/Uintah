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
 *  ProxyBase.cc: Base class for all PIDL proxies
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */
#include <Core/Thread/Thread.h>
#include <Core/CCA/Component/PIDL/ProxyBase.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <iostream>
using namespace SCIRun;

ProxyBase::ProxyBase() { }

ProxyBase::ProxyBase(const Reference& ref)
  : d_ref(ref), attached_( false )
{ }

ProxyBase::~ProxyBase()
{
  if( attached_ )
    {
      d_ref.chan->closeConnection();
    }
}

void ProxyBase::_proxyGetReference(Reference& ref, bool copy) const
{
  if (copy) {
    if (ref.chan != NULL) {
      delete (ref.chan);
      ref.chan = NULL;
    }
    ref.chan = (d_ref.chan)->SPFactory(true);
  }
  else {
    ref = d_ref;
  }
}






