
/*
 *  Object.cc: Base class for all PIDL distributed objects
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Component/PIDL/Object.h>

#include <Component/PIDL/Dispatch.h>
#include <Component/PIDL/GlobusError.h>
#include <Component/PIDL/InvalidReference.h>
#include <Component/PIDL/PIDL.h>
#include <Component/PIDL/ServerContext.h>
#include <Component/PIDL/URL.h>
#include <Component/PIDL/Wharehouse.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Util/NotFinished.h>
#include <sstream>

using Component::PIDL::Object_interface;
using Component::PIDL::URL;
using SCICore::Exceptions::InternalError;

static void unknown_handler(globus_nexus_endpoint_t* endpoint,
			    globus_nexus_buffer_t* buffer,
			    int handler_id)
{
    NOT_FINISHED("unknown handler");
}

Object_interface::Object_interface(const TypeInfo* typeinfo,
						  Dispatch* dispatch,
						  void* ptr)
{
    d_serverContext=new ServerContext;
    d_serverContext->d_typeinfo=typeinfo;
    globus_nexus_endpointattr_t attr;
    if(int gerr=globus_nexus_endpointattr_init(&attr))
	throw GlobusError("endpointattr_init", gerr);
    if(int gerr=globus_nexus_endpointattr_set_handler_table(&attr,
							    dispatch->table,
							    dispatch->tableSize))
	throw GlobusError("endpointattr_set_handler_table", gerr);
    if(int gerr=globus_nexus_endpointattr_set_unknown_handler(&attr,
							      unknown_handler,
							      GLOBUS_NEXUS_HANDLER_TYPE_THREADED))
	throw GlobusError("endpointattr_set_unknown_handler", gerr);
    if(int gerr=globus_nexus_endpoint_init(&d_serverContext->d_endpoint,
					   &attr))
	throw GlobusError("endpoint_init", gerr);    
    globus_nexus_endpoint_set_user_pointer(&d_serverContext->d_endpoint, ptr);
    if(int gerr=globus_nexus_endpointattr_destroy(&attr))
	throw GlobusError("endpointattr_destroy", gerr);

    Wharehouse* wharehouse=PIDL::getWharehouse();
    d_serverContext->d_objid=wharehouse->registerObject(this);
}

Object_interface::Object_interface()
{
    d_serverContext=0;
}

Object_interface::~Object_interface()
{
    if(d_serverContext){
	Wharehouse* wharehouse=PIDL::getWharehouse();
	if(wharehouse->unregisterObject(d_serverContext->d_objid) != this)
	    throw InternalError("Corruption in object wharehouse");
	//d_serverContext->d_endpoint->shutdown();
	NOT_FINISHED("Object_interface::~Object_interface");
	delete d_serverContext;
    } else {
	NOT_FINISHED("Object::~Object");
    }
}

URL Object_interface::getURL() const
{
    if(d_serverContext){
	std::ostringstream o;
	o << PIDL::getBaseURL() << d_serverContext->d_objid;
	return o.str();
    } else {
	NOT_FINISHED("Object::getURL");
    }
}

//
// $Log$
// Revision 1.2  1999/08/31 08:59:01  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:46  sparker
// Updates to configure script:
//  rebuild configure if configure.in changes (Bug #35)
//  Fixed rule for rebuilding Makefile from Makefile.in (Bug #36)
//  Rerun configure if configure changes (Bug #37)
//  Don't build Makefiles for modules that aren't --enabled (Bug #49)
//  Updated Makfiles to build sidl and Component if --enable-parallel
// Updates to sidl code to compile on linux
// Imported PIDL code
// Created top-level Component directory
// Added ProcessManager class - a simpler interface to fork/exec (not finished)
//
//
