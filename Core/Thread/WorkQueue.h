
// $Id$

/*
 *  WorkQueue: Manage assignments of work
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCICore_Thread_WorkQueue_h
#define SCICore_Thread_WorkQueue_h

/**************************************
 
CLASS
   WorkQueue
   
KEYWORDS
   Thread, Work
   
DESCRIPTION
   Doles out work assignment to various worker threads.  Simple
   attempts are made at evenly distributing the workload.
   Initially, assignments are relatively large, and will get smaller
   towards the end in an effort to equalize the total effort.
   
PATTERNS


WARNING
   
****************************************/

#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/Mutex.h>
#include <vector>

namespace SCICore {
    namespace Thread {
	struct WorkQueue_private;
	
	class WorkQueue {
	public:
	    //////////
	    // Make an empty work queue with no assignments.
	    WorkQueue(const char* name);
	    
	    //////////
	    // Fill the work queue with the specified total number of work
	    // assignments.  <i>num_threads</i> specifies the approximate
	    // number of threads which will be working from this queue.
	    // The optional <i>granularity</i> specifies the degree to
	    // which the tasks are divided.  A large granularity will
	    // create more assignments with smaller assignments.  A
	    // granularity of zero will recieve a single assignment of
	    // approximately uniform size.  <i>name</i> should be a static
	    // string which describes the primitive for debugging purposes.
	    void refill(int totalAssignments, int nthreads,
			bool dynamic, int granularity=5);
	    
	    //////////
	    // Destroy the work queue.  Any unassigned work will be lost.  
	    ~WorkQueue();
	    
	    //////////
	    // Called by each thread to get the next assignment.  If
	    // <i>nextAssignment</i> returns true, the thread has a valid
	    // assignment, and then would be responsible for the work 
	    // from the returned <i>start</i> through <i>end-l</i>.
	    // Assignments can range from 0 to  <i>totalAssignments</i>-1.
	    // When <i>nextAssignment</i> returns false, all work has
	    // been completed (dynamic=true), or has been assigned
	    // (dynamic=false).
	    bool nextAssignment(int& start, int& end);
	    
	    //////////
	    // Increase the work to be done.  <i>dynamic</i> as provided
	    // to the constructor MUST be true. Work should only be added
	    // by the workers.
	    void addWork(int nassignments);

	    //////////
	    // Block until the queue is empty
	    void waitForEmpty();

	private:
	    WorkQueue_private* d_priv;
	    const char* d_name;
	    int d_num_threads;
	    int d_total_assignments;
	    int d_granularity;
	    std::vector<int> d_assignments;
	    
	    int d_num_waiting;
	    bool d_done;
	    bool d_dynamic;
	    
	    void init();
	    void fill();
	    WorkQueue(const WorkQueue& copy);
	    WorkQueue& operator=(const WorkQueue&);
	};
    }
}

#endif

//
// $Log$
// Revision 1.5  1999/08/25 19:00:53  sparker
// More updates to bring it up to spec
// Factored out common pieces in Thread_irix and Thread_pthreads
// Factored out other "default" implementations of various primitives
//
// Revision 1.4  1999/08/25 02:38:03  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//
