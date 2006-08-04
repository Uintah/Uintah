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
	    const char* name_;
	    int num_threads_;
	    int total_assignments_;
	    int granularity_;
	    std::vector<int> assignments_;
	    AtomicCounter current_assignment_;
	    bool done_;
	    bool dynamic_;
	    
	    void fill();

	    // Cannot copy them
	    WorkQueue(const WorkQueue& copy);
	    WorkQueue& operator=(const WorkQueue&);
	};
} // End namespace SCIRun

#endif

