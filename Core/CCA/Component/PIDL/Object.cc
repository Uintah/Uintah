
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

#include <Component/PIDL/GlobusError.h>
#include <Component/PIDL/InvalidReference.h>
#include <Component/PIDL/PIDL.h>
#include <Component/PIDL/Reference.h>
#include <Component/PIDL/ServerContext.h>
#include <Component/PIDL/TypeInfo.h>
#include <Component/PIDL/TypeInfo_internal.h>
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
    cerr << "handler_id=" << handler_id << '\n';
    NOT_FINISHED("unknown handler");
}

Object_interface::Object_interface()
    : d_serverContext(0)
{
}

void Object_interface::initializeServer(const TypeInfo* typeinfo, void* ptr)
{
    if(d_serverContext){
	// This happens because initializeServer gets called by
	// all of the parent interfaces.  We have no way of knowing
	// which one is the last, so we overwrite the old
	// endpoint
	if(int gerr=globus_nexus_endpoint_destroy(&d_serverContext->d_endpoint))
	    throw GlobusError("endpoint_destroy", gerr);
    } else {
	d_serverContext=new ServerContext;
	Wharehouse* wharehouse=PIDL::getWharehouse();
	d_serverContext->d_objid=wharehouse->registerObject(this);
    }
    d_serverContext->d_typeinfo=typeinfo;
    d_serverContext->d_ptr=ptr;
    d_serverContext->d_objptr=this;
    globus_nexus_endpointattr_t attr;
    if(int gerr=globus_nexus_endpointattr_init(&attr))
	throw GlobusError("endpointattr_init", gerr);
    if(int gerr=globus_nexus_endpointattr_set_handler_table(&attr,
							    typeinfo->d_priv->table,
							    typeinfo->d_priv->tableSize))
	throw GlobusError("endpointattr_set_handler_table", gerr);
    if(int gerr=globus_nexus_endpointattr_set_unknown_handler(&attr,
							      unknown_handler,
							      GLOBUS_NEXUS_HANDLER_TYPE_THREADED))
	throw GlobusError("endpointattr_set_unknown_handler", gerr);
    if(int gerr=globus_nexus_endpoint_init(&d_serverContext->d_endpoint,
					   &attr))
	throw GlobusError("endpoint_init", gerr);    
    globus_nexus_endpoint_set_user_pointer(&d_serverContext->d_endpoint,
					   (void*)d_serverContext);
    if(int gerr=globus_nexus_endpointattr_destroy(&attr))
	throw GlobusError("endpointattr_destroy", gerr);
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

void Object_interface::_getReference(Reference& ref, bool copy) const
{
    if(!d_serverContext)
	throw SCICore::Exceptions::InternalError("Object_interface::getReference called for a non-server object");
    if(!copy){
	throw SCICore::Exceptions::InternalError("Object_interface::getReference called with copy=false");
    }
    if(int gerr=globus_nexus_startpoint_bind(&ref.d_sp, &d_serverContext->d_endpoint))
	throw GlobusError("startpoint_bind", gerr);
    ref.d_vtable_base=TypeInfo::vtable_methods_start;
}

//
// $Log$
// Revision 1.4  1999/09/21 06:12:59  sparker
// Fixed bugs in multiple inheritance
// Added round-trip optimization
// To support this, we store Startpoint* in the endpoint instead of the
//    object final type.
//
// Revision 1.3  1999/09/17 05:08:08  sparker
// Implemented component model to work with sidl code generator
//
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
