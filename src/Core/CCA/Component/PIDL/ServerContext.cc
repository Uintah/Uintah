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
 *  ServerContext.cc: Local class for PIDL that holds the context
 *                   for server objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/ServerContext.h>
#include <Core/CCA/Component/PIDL/GlobusError.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <Core/CCA/Component/PIDL/TypeInfo_internal.h>
#include <Core/Util/NotFinished.h>
#include <iostream>
using std::cerr;

using PIDL::ServerContext;

static void unknown_handler(globus_nexus_endpoint_t* endpoint,
			    globus_nexus_buffer_t* buffer,
			    int handler_id)
{
  cerr << "handler_id=" << handler_id << '\n';
  NOT_FINISHED("unknown handler");
}

void
ServerContext::activateEndpoint()
{
  globus_nexus_endpointattr_t attr;
  if(int gerr=globus_nexus_endpointattr_init(&attr))
    throw GlobusError("endpointattr_init", gerr);
  if(int gerr=globus_nexus_endpointattr_set_handler_table(&attr,
							  d_typeinfo->d_priv->table,
							  d_typeinfo->d_priv->tableSize))
    throw GlobusError("endpointattr_set_handler_table", gerr);
  if(int gerr=globus_nexus_endpointattr_set_unknown_handler(&attr,
							    unknown_handler,
							    GLOBUS_NEXUS_HANDLER_TYPE_THREADED))
    throw GlobusError("endpointattr_set_unknown_handler", gerr);
  if(int gerr=globus_nexus_endpoint_init(&d_endpoint, &attr))
    throw GlobusError("endpoint_init", gerr);    
  globus_nexus_endpoint_set_user_pointer(&d_endpoint, (void*)this);
  if(int gerr=globus_nexus_endpointattr_destroy(&attr))
    throw GlobusError("endpointattr_destroy", gerr);
  
  d_endpoint_active=true;
}

