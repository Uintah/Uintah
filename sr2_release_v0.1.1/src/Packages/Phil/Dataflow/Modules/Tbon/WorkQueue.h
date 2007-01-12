
/*
  WorkQueue.h
  list of jobs for parallel processes
  (most of this derives from Steve's rtrt)

  Packages/Philip Sutton
  July 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __TBON_WORKQUEUE_H__
#define __TBON_WORKQUEUE_H__

extern "C" {
#include <sys/pmo.h>
#include <fetchop.h>
}
#include <iostream>

#include <Core/Thread/Mutex.h>
#include <Core/Thread/Barrier.h>

namespace Phil {
using namespace SCIRun;
using namespace std;

// this will be 1 on an R10k, where fetchop commands are available.
// you have to use semaphores, etc. if on another platform.
#define USE_FETCHOP 1

struct Job {
  int index;
  int x, y, z;
};

class WorkQueue {
public:
  WorkQueue( int np, int maxjobs );
  ~WorkQueue();

  void addjob( int i, int index, int x, int y, int z ) {
    jobs[i].index = index; jobs[i].x = x; jobs[i].y = y; jobs[i].z = z;
  }

  int getWork( int& start, int& end );
  void prepare( int n );

  Job* jobs;

protected:
private:
  int numProcessors;
  int numJobs;
  atomic_var_t *atom;
  atomic_reservoir_t reservoir;
  int locked_atom;
  Mutex* mutex;

  int nassignments;
  int* assignments;

  static const int THRESHOLD;
};

// define constants

// this will be the fewest number of jobs given out.
// useful values are around 2..16 for most data sets.
const int WorkQueue::THRESHOLD = 16; // must be >= 2

// constructor
WorkQueue::WorkQueue( int np, int maxjobs ) {
  // create new queue
  numProcessors = np;
  numJobs = 0;
  jobs = new Job[maxjobs];
  mutex = new Mutex("WorkQueue mutex");

  // ready the atomic variable
#if USE_FETCHOP
  reservoir = atomic_alloc_reservoir(USE_DEFAULT_PM, 10, NULL);
  if( !reservoir ) {
    cerr << "Error - cannot allocate reservoir!" << endl;
    return;
  }
  atom = atomic_alloc_variable(reservoir, NULL);
#else
  // do nothing
#endif
}

// clean up
WorkQueue::~WorkQueue() {
  delete [] jobs;
  delete [] assignments;
#if USE_FETCHOP
  atomic_free_reservoir(reservoir);
#endif
}

// prepare for a search
void
WorkQueue::prepare( int n ) {
  int last = 0;

  numJobs = n;
  assignments = new int[n+1];

  nassignments = 0;

  int unitsize = numJobs / (3*numProcessors);
  while( last < numJobs && unitsize > THRESHOLD ) {
    for( int counter = 0; 
	 counter < numProcessors && last < numJobs; 
	 counter++ ) {
      assignments[nassignments++] = last;
      last += unitsize;
      if( last > numJobs )
	last = numJobs;
    }
    unitsize = unitsize >> 1;
  } // while( unitsize > THRESHOLD )

  while( last < numJobs ) {
    assignments[nassignments++] = last;
    last += unitsize;
  }
  assignments[nassignments] = numJobs;

  //  cout << "nassignments = " << nassignments << endl;
#if USE_FETCHOP
  atomic_store( atom, 0 );
#else
  locked_atom = 0;
#endif
}

int 
WorkQueue::getWork( int& start, int& end ) {
  int i;
#if USE_FETCHOP
  i = (int)atomic_fetch_and_increment( atom );
#else  
  mutex->lock();
  i = locked_atom;
  locked_atom++;
  mutex->unlock();
#endif

  if( i >= nassignments )
    return 0;
  start = assignments[i];
  end = assignments[i+1];
  return 1;
}
} // End namespace Phil



#endif


