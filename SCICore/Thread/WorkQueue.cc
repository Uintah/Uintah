
/* REFERENCED */
static char *id="$Id$";

/*
 *  WorkQueue.h: Manage assignments of work
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */


#include "WorkQueue.h"
<<<<<<< WorkQueue.cc

void WorkQueue::fill()
{
    if(d_totalAssignments==0){
	d_assignments.resize(0);
	return;
    }
    d_assignments.reserve(d_totalAssignments+1);
    int current_assignment=0;
    int current_assignmentsize=(2*d_totalAssignments)/(d_numThreads*(d_granularity+1));
    int decrement=current_assignmentsize/d_granularity;
    if(current_assignmentsize==0)
	current_assignmentsize=1;
    if(decrement==0)
	decrement=1;
    for(int i=0;i<d_granularity;i++){
	for(int j=0;j<d_numThreads;j++){
	    d_assignments.push_back(current_assignment);
	    current_assignment+=current_assignmentsize;
	    if(current_assignment >= d_totalAssignments){
		break;
	    }
	}
	if(current_assignment >= d_totalAssignments){
	  break;
	}
	current_assignmentsize-=decrement;
	if(current_assignmentsize<1)
	    current_assignmentsize=1;
	if(current_assignment >= d_totalAssignments){
	    break;
	}
    }
    while(current_assignment < d_totalAssignments){
	d_assignments.push_back(current_assignment);
	current_assignment+=current_assignmentsize;
    }
    d_assignments.push_back(d_totalAssignments);
    d_numWaiting=0;
    d_done=false;
}

=======
#include "Thread.h"
#include <iostream.h>
>>>>>>> 1.8
//
// $Log$
// Revision 1.3  1999/08/25 02:38:03  sparker
// Added namespaces
// General cleanups to prepare for integration with SCIRun
//
//

