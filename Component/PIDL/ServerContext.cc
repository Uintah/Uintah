
/*
 *  ServerContext.cc: Local class for PIDL that holds the context
 *                   for server objects
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Component/PIDL/ServerContext.h>
#include <Component/PIDL/GlobusError.h>
#include <Component/PIDL/TypeInfo.h>
#include <Component/PIDL/TypeInfo_internal.h>
#include <SCICore/Util/NotFinished.h>

using Component::PIDL::ServerContext;

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
}

//
// $Log$
// Revision 1.1  1999/09/24 06:26:25  sparker
// Further implementation of new Component model and IDL parser, including:
//  - fixed bugs in multiple inheritance
//  - added test for multiple inheritance
//  - fixed bugs in object reference send/receive
//  - added test for sending objects
//  - beginnings of support for separate compilation of sidl files
//  - beginnings of CIA spec implementation
//  - beginnings of cocoon docs in PIDL
//  - cleaned up initalization sequence of server objects
//  - use globus_nexus_startpoint_eventually_destroy (contained in
// 	the globus-1.1-utah.patch)
//
//
