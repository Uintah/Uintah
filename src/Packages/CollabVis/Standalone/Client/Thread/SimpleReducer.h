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
 *  SimpleReducer: A barrier with reduction operations
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_SimpleReducer_h
#define Core_Thread_SimpleReducer_h

#include <Core/share/share.h>

#include <Core/Thread/Barrier.h>

namespace SCIRun {

/**************************************
 
CLASS
   SimpleReducer
   
KEYWORDS
   Thread
   
DESCRIPTION
   Perform reduction operations over a set of threads.  Reduction
   operations include things like global sums, global min/max, etc.
   In these operations, a local sum (operation) is performed on each
   thread, and these sums are added together.
   
****************************************/
	class SCICORESHARE SimpleReducer : public Barrier {
	public:
	    //////////
	    // Create a <b> SimpleReducer</i>.
	    // At each operation, a barrier wait is performed, and the
	    // operation will be performed to compute the global balue.
	    // <i>name</i> should be a static string which describes
	    // the primitive for debugging purposes.
	    SimpleReducer(const char* name);

	    //////////
	    // Destroy the SimpleReducer and free associated memory.
	    virtual ~SimpleReducer();

	    //////////
	    // Performs a global sum over all of the threads.  As soon as each
	    // thread has called sum with their local sum, each thread will
	    // return the same global sum.
	    double sum(int myrank, int numThreads, double mysum);

	    //////////
	    // Performs a global max over all of the threads.  As soon as each
	    // thread has called max with their local max, each thread will
	    // return the same global max.
	    double max(int myrank, int numThreads, double mymax);

	    //////////
	    // Performs a global min over all of the threads.  As soon as each
	    // thread has called min with their local max, each thread will
	    // return the same global max.
	    double min(int myrank, int numThreads, double mymax);

	private:
	    struct data {
		double d_;
	    };
	    struct joinArray {
		data d_;
		// Assumes 128 bytes in a cache line...
		char filler_[128-sizeof(data)];
	    };
	    struct pdata {
		int buf_;
		char filler_[128-sizeof(int)];	
	    };
	    joinArray* join_[2];
	    pdata* p_;
	    int array_size_;
	    void collectiveResize(int proc, int numThreads);

	    // Cannot copy them
	    SimpleReducer(const SimpleReducer&);
	    SimpleReducer& operator=(const SimpleReducer&);
	};
} // End namespace SCIRun

#endif


