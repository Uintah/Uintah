
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
/**************************************
 
CLASS
   ReplyEP
   
KEYWORDS
   Endpoint, PIDL
   
DESCRIPTION
   Class to provide efficient access to a transient endpoint, typically
   used for a reply.  The static methods acquire and release are used
   to obtain a ReplyEP for use in a reply.  The wait() method blocks
   until the reply has been sent to the reply handler (handler number
   0), and returns the nexus buffer.
****************************************/
	class ReplyEP {
	public:
	    //////////
	    // Acquire a ReplyEP.  Creates a new one if none are
	    // ready to be used.  Eventually, the caller should call
	    // release() on this object.
	    static ReplyEP* acquire();

	    //////////
	    // Release the ReplyEP.
	    static void release(ReplyEP*);

	    //////////
	    // Wait for the reply and return the associated buffer.  The
	    // caller should eventually call globus_nexus_buffer_destroy
	    // on this buffer.
	    globus_nexus_buffer_t wait();

	    //////////
	    // Return the startpoint associated with this object.  This
	    // is used only in sidl-generated stub code.  It does NOT
	    // call globus_nexus_startpoint_copy.  That should be done
	    // in the generated code if necessary.
	    void get_startpoint(globus_nexus_startpoint_t*);


	    //////////
	    // The nexus handler for this object.  This is for internal
	    // use only.
	    static void reply_handler(globus_nexus_endpoint_t* ep,
				      globus_nexus_buffer_t* buffer,
				      globus_bool_t threadedHandler);
	private:
	    //////////
	    // Create the ReplyEP object. Used internally
	    ReplyEP();

	    //////////
	    // Destructor
	    ~ReplyEP();

	    //////////
	    // Copy construct is private to prevent accidental use.
	    ReplyEP(const ReplyEP&);

	    //////////
	    // The endpoint
	    globus_nexus_endpoint_t d_ep;

	    //////////
	    // The startpoint associated with this endpoint
	    globus_nexus_startpoint_t d_sp;

	    //////////
	    // The semaphore used for blocking
	    SCICore::Thread::Semaphore d_sema;

	    //////////
	    // A place to bounce the buffer from the handler thread to
	    // the waiter.
	    globus_nexus_buffer_t d_bufferHandoff;
	};
    }
}

#endif

//
// $Log$
// Revision 1.4  1999/09/24 20:03:37  sparker
// Added cocoon documentation
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
