
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

#ifndef Component_PIDL_ReplyEP_h
#define Component_PIDL_ReplyEP_h

#include <SCICore/Thread/Semaphore.h>
#include <globus_nexus.h>

namespace Component {
    namespace PIDL {
	class ReplyEP {
	public:
	    static ReplyEP* acquire();
	    static void release(ReplyEP*);
	    globus_nexus_buffer_t wait();
	    void get_startpoint(globus_nexus_startpoint_t*);


	    static void reply_handler(globus_nexus_endpoint_t* ep,
				      globus_nexus_buffer_t* buffer,
				      globus_bool_t threadedHandler);
	protected:
	    ReplyEP();
	    ~ReplyEP();
	    ReplyEP(const ReplyEP&);
	    globus_nexus_endpoint_t d_ep;
	    globus_nexus_startpoint_t d_sp;
	    SCICore::Thread::Semaphore d_sema;
	    globus_nexus_buffer_t d_bufferHandoff;
	};
    }
}

#endif
//
// $Log$
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
