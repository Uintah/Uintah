
#ifndef SCI_THREAD_WORKQUEUE_H
#define SCI_THREAD_WORKQUEUE_H 1

/**************************************
 
CLASS
   WorkQueue
   
KEYWORDS
   WorkQueue
   
DESCRIPTION
   Doles out work assignment to various worker threads.  Simple
   attempts are made at evenly distributing the workload.
   Initially, assignments are relatively large, and will get smaller
   towards the end in an effort to equalize the total effort.
   
PATTERNS


WARNING
   
****************************************/

#include "Mutex.h"
#include "ConditionVariable.h"
#include <string>
#include <vector>
struct WorkQueue_private;

class WorkQueue {
    WorkQueue_private* d_priv;
    std::string d_name;
    int d_num_threads;
    int d_total_assignments;
    int d_granularity;
    vector<int> d_assignments;

    int d_num_waiting;
    bool d_done;
    bool d_dynamic;

    void init();
    void fill();
    WorkQueue(const WorkQueue& copy);
    WorkQueue& operator=(const WorkQueue&);
public:

    //////////
    // Create the work queue with the specified total number of work
    // assignments.  <i>nthreads</i> specifies the approximate number
    // of threads which will be working from this queue.  The optional
    // <i>granularity</i> specifies the degree to which the tasks are
    // divided.  A large granularity will create more assignments with
    // smaller assignments.  A granularity of zero will recieve a single
    // assignment of approximately uniform size.  <i>name</i> should be
    // a string which describes the primitive for debugging
    // purposes.
    WorkQueue(const std::string& name, int totalAssignments, int nthreads,
	      bool dynamic, int granularity=5);
    void refill(int totalAssignments, int nthreads,
		bool dynamic, int granularity=5);

    //////////
    // Make an empty work queue with no assignments.
    WorkQueue(const std::string& name);

    //////////
    // Destroy the work queue.  Any unassigned work will be lost.  
    ~WorkQueue();

    //////////
    // Called by each thread to get the next assignment.  If
    // <i>nextAssignment</i> returns true, the thread has a valid
    // assignment, and then would be responsible for the work from the 
    // returned <i>start</i> through <i>end-l</i>.  Assignments can range
    // from 0 to  <i>totalAssignments</i>-1.  When <i>nextAssignment</i>
    // returns false, all work has been completed (dynamic=true), or has
    // been assigned (dynamic=false).
    bool nextAssignment(int& start, int& end);

    //////////
    // Increase the work to be done.  <i>dynamic</i> as provided to the
    // constructor MUST be true. Work should only be added by the workers.
    void addWork(int nassignments);


    //////////
    // Block until the queue is empty
    void waitForEmpty();
};

#endif
