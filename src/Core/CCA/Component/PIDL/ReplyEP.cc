
/*
 *  ReplyEP.h:  A set of prebuilt endpoints for replies
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

#include <Component/PIDL/ReplyEP.h>
#include <Component/PIDL/GlobusError.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Util/NotFinished.h>
#include <globus_nexus.h>
#include <vector>

using Component::PIDL::ReplyEP;
using SCICore::Thread::Mutex;

static Mutex mutex("ReplyEP pool lock");
static std::vector<ReplyEP*> pool;

void ReplyEP::reply_handler(globus_nexus_endpoint_t* ep,
			    globus_nexus_buffer_t* buffer,
			    globus_bool_t)
{
    globus_nexus_buffer_save(buffer);
    void* p=globus_nexus_endpoint_get_user_pointer(ep);
    ReplyEP* r=(ReplyEP*)p;
    r->d_bufferHandoff=*buffer;
    r->d_sema.up();
}

static globus_nexus_handler_t reply_table[] =
{
    {GLOBUS_NEXUS_HANDLER_TYPE_NON_THREADED, ReplyEP::reply_handler}
};

static void unknown_handler(globus_nexus_endpoint_t* endpoint,
			    globus_nexus_buffer_t* buffer,
			    int handler_id)
{
    NOT_FINISHED("unknown handler");
}

ReplyEP::ReplyEP()
    : d_sema("Reply wait semaphore", 0)
{
}

ReplyEP::~ReplyEP()
{
}

ReplyEP* ReplyEP::acquire()
{
    mutex.lock();
    if(pool.size() == 0){
	mutex.unlock();
	ReplyEP* r=new ReplyEP;
	globus_nexus_endpointattr_t attr;
	if(int gerr=globus_nexus_endpointattr_init(&attr))
	    throw GlobusError("endpointattr_init", gerr);
	if(int gerr=globus_nexus_endpointattr_set_handler_table(&attr,
								reply_table,
								1))
	    throw GlobusError("endpointattr_set_handler_table", gerr);
	if(int gerr=globus_nexus_endpointattr_set_unknown_handler(&attr,
								  unknown_handler,
								  GLOBUS_NEXUS_HANDLER_TYPE_THREADED))
	    throw GlobusError("endpointattr_set_unknown_handler", gerr);
	if(int gerr=globus_nexus_endpoint_init(&r->d_ep, &attr))
	    throw GlobusError("endpoint_init", gerr);
	globus_nexus_endpoint_set_user_pointer(&r->d_ep, r);
	if(int gerr=globus_nexus_endpointattr_destroy(&attr))
	    throw GlobusError("endpointattr_destroy", gerr);

	if(int gerr=globus_nexus_startpoint_bind(&r->d_sp, &r->d_ep))
	    throw GlobusError("bind_startpoint", gerr);

	return r;
    } else {
	ReplyEP* r=pool.back();
	pool.pop_back();
	mutex.unlock();
	return r;
    }
}

void ReplyEP::release(ReplyEP* r)
{
    mutex.lock();
    pool.push_back(r);
    mutex.unlock();
}

void ReplyEP::get_startpoint(globus_nexus_startpoint_t* spp)
{
    *spp=d_sp;
}

globus_nexus_buffer_t ReplyEP::wait()
{
    d_sema.down();
    return d_bufferHandoff;
}

//
// $Log$
// Revision 1.4  1999/09/21 06:13:00  sparker
// Fixed bugs in multiple inheritance
// Added round-trip optimization
// To support this, we store Startpoint* in the endpoint instead of the
//    object final type.
//
// Revision 1.3  1999/09/17 05:08:10  sparker
// Implemented component model to work with sidl code generator
//
// Revision 1.2  1999/08/31 08:59:02  sparker
// Configuration and other updates for globus
// First import of beginnings of new component library
// Added yield to Thread_irix.cc
// Added getRunnable to Thread.{h,cc}
//
// Revision 1.1  1999/08/30 17:39:48  sparker
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
