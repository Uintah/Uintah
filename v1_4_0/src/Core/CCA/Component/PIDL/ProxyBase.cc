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
 *  ProxyBase.h: Base class for all PIDL proxies
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
#include <Core/CCA/Component/PIDL/GlobusError.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>

#include <globus_nexus.h>

#include <iostream>

using namespace std;

using PIDL::ProxyBase;
using PIDL::Reference;
using SCIRun::Thread;

ProxyBase::ProxyBase(const Reference& ref)
  : d_ref(ref), attached_( false )
{
}

ProxyBase::~ProxyBase()
{
  if( attached_ )
    {
      int size=0;
      globus_nexus_buffer_t buffer;
      if(int gerr=globus_nexus_buffer_init(&buffer, size, 0))
	{
	  throw GlobusError("buffer_init", gerr);
	}

      // Send the message
      Reference ref;
      _proxyGetReference(ref, false);
      int handler=TypeInfo::vtable_deleteReference_handler;
      if(int gerr=globus_nexus_send_rsr(&buffer, &ref.d_sp,
					handler, GLOBUS_TRUE, GLOBUS_FALSE))
	{
	  throw GlobusError("ProxyBase: send_rsr", gerr);
	}
      // No reply is sent for this
      if(d_ref.d_vtable_base != TypeInfo::vtable_invalid){
	if(int gerr=globus_nexus_startpoint_destroy_and_notify(&d_ref.d_sp)){
	  throw GlobusError("nexus_startpoint_destroy_and_notify", gerr);
	}
      }
    }
}

void ProxyBase::_proxyGetReference(Reference& ref, bool copy) const
{
  ref=d_ref;
  if( copy ){
    if( int gerr = globus_nexus_startpoint_copy(
						&ref.d_sp, const_cast<globus_nexus_startpoint_t*>(&d_ref.d_sp)))
      throw GlobusError("startpoint_copy", gerr);
  }
}

