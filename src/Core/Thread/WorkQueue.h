
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

#ifndef Core_Thread_WorkQueue_h
#define Core_Thread_WorkQueue_h

#include <Core/share/share.h>

#include <Core/Thread/AtomicCounter.h>
#include <vector>

namespace SCIRun {
	struct WorkQueue_private;
	
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
   
****************************************/
	class SCICORESHARE WorkQueue {
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
			int granularity=5);
	    
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
	    // been assigned.
	    bool nextAssignment(int& start, int& end);
	    
	private:
	    const char* d_name;
	    int d_num_threads;
	    int d_total_assignments;
	    int d_granularity;
	    std::vector<int> d_assignments;
	    AtomicCounter d_current_assignment;
	    bool d_done;
	    bool d_dynamic;
	    
	    void fill();

	    // Cannot copy them
	    WorkQueue(const WorkQueue& copy);
	    WorkQueue& operator=(const WorkQueue&);
	};
} // End namespace SCIRun

#endif

