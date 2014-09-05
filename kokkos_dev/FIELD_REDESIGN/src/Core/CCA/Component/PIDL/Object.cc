
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
#include <Component/PIDL/PIDL.h>
#include <Component/PIDL/Reference.h>
#include <Component/PIDL/ServerContext.h>
#include <Component/PIDL/TypeInfo.h>
#include <Component/PIDL/URL.h>
#include <Component/PIDL/Wharehouse.h>
#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Thread/MutexPool.h>
#include <sstream>

using Component::PIDL::Object_interface;
using Component::PIDL::URL;
using SCICore::Exceptions::InternalError;
using SCICore::Thread::Mutex;
using SCICore::Thread::MutexPool;

Object_interface::Object_interface()
    : d_serverContext(0)
{
    ref_cnt=0;
    mutex_index=getMutexPool()->nextIndex();
}

void Object_interface::initializeServer(const TypeInfo* typeinfo, void* ptr)
{
    if(!d_serverContext){
	d_serverContext=new ServerContext;
	d_serverContext->d_endpoint_active=false;
	d_serverContext->d_objptr=this;
    } else if(d_serverContext->d_endpoint_active){
	throw InternalError("Server re-initialized while endpoint already active?");
    } else if(d_serverContext->d_objptr != this){
	throw InternalError("Server re-initialized with a different base class ptr?");
    }
    //
    // This may happen multiple times, due to multiple inheritance.  It
    // is a "last one wins" approach - the last CTOR to call this function
    // is the most derived type.
    d_serverContext->d_typeinfo=typeinfo;
    d_serverContext->d_ptr=ptr;
}

Object_interface::~Object_interface()
{
    if(ref_cnt != 0)
	throw InternalError("Object delete while reference count != 0");
    if(d_serverContext){
	Wharehouse* wharehouse=PIDL::getWharehouse();
	if(wharehouse->unregisterObject(d_serverContext->d_objid) != this)
	    throw InternalError("Corruption in object wharehouse");
	if(int gerr=globus_nexus_endpoint_destroy(&d_serverContext->d_endpoint))
	    throw GlobusError("endpoint_destroy", 0);
	delete d_serverContext;
    }
}

URL Object_interface::getURL() const
{
    std::ostringstream o;
    if(d_serverContext){
	if(!d_serverContext->d_endpoint_active)
	    activateObject();
	o << PIDL::getBaseURL() << d_serverContext->d_objid;
    } else {
	// TODO - send a message to get the URL
	o << "getURL() doesn't (yet) work for proxy objects";
    }
    return o.str();
}

void Object_interface::_getReference(Reference& ref, bool copy) const
{
    if(!d_serverContext)
	throw SCICore::Exceptions::InternalError("Object_interface::getReference called for a non-server object");
    if(!copy){
	throw SCICore::Exceptions::InternalError("Object_interface::getReference called with copy=false");
    }
    if(!d_serverContext->d_endpoint_active)
	activateObject();
    if(int gerr=globus_nexus_startpoint_bind(&ref.d_sp, &d_serverContext->d_endpoint))
	throw GlobusError("startpoint_bind", gerr);
    ref.d_vtable_base=TypeInfo::vtable_methods_start;
}

void Object_interface::_addReference()
{
    Mutex* m=getMutexPool()->getMutex(mutex_index);
    m->lock();
    ref_cnt++;
    m->unlock();
}

void Object_interface::_deleteReference()
{
    Mutex* m=getMutexPool()->getMutex(mutex_index);
    m->lock();
    ref_cnt--;
    bool del;
    if(ref_cnt == 0)
	del=true;
    else
	del=false;
    m->unlock();

    // We must delete outside of the lock to prevent deadlock
    // conditions with the mutex pool, but we must check the condition
    // inside the lock to prevent race conditions with other threads
    // simultaneously releasing a reference.
    if(del)
	delete this;
}

void Object_interface::activateObject() const
{
    Wharehouse* wharehouse=PIDL::getWharehouse();
    d_serverContext->d_objid=wharehouse->registerObject(const_cast<Object_interface*>(this));
    d_serverContext->activateEndpoint();
}

MutexPool* Object_interface::getMutexPool()
{
    static MutexPool* pool=0;
    if(!pool){
	// TODO - make this threadsafe.  This can leak if two threads
	// happen to request the first pool at the same time.  I doubt
	// it will ever happen - sparker.
	pool=new MutexPool("Component::PIDL::Object_interface mutex pool", 63);
    }
    return pool;
}

//
// $Log$
// Revision 1.6  1999/09/26 06:12:55  sparker
// Added (distributed) reference counting to PIDL objects.
// Began campaign against memory leaks.  There seem to be no more
//   per-message memory leaks.
// Added a test program to flush out memory leaks
// Fixed other Component testprograms so that they work with ref counting
// Added a getPointer method to PIDL handles
//
// Revision 1.5  1999/09/24 06:26:25  sparker
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
