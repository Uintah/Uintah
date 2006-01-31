/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  MutexPool: A set of mutex objects
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Thread_MutexPool_h
#define Core_Thread_MutexPool_h

#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>

#include <Core/Thread/share.h>

namespace SCIRun {
/**************************************
 
CLASS
   MutexPool
   
KEYWORDS
   Thread, Mutex
   
DESCRIPTION
   A container class for a set of Mutex objects.  This can be used to
   limit the number of active mutexes down to a more reasonable set.
   However, this must be used very carefully, as it becomes easy to
   create a hold-and-wait condition.
****************************************/
	class SCISHARE MutexPool {
	public:
	    //////////
	    // Create the mutex pool with size mutex objects.
	    MutexPool(const char* name, int size);

	    //////////
	    // Destroy the mutex pool and all mutexes in it.
	    ~MutexPool();

	    //////////
	    // return the next index in a round-robin fashion
	    int nextIndex();

	    //////////
	    // return the idx'th mutex.
	    Mutex* getMutex(int idx);

	    //////////
	    // lock the idx'th mutex.
	    void lockMutex(int idx);

	    //////////
	    // unlock the idx'th mutex.
	    void unlockMutex(int idx);
	private:
	    //////////
	    // The next ID
	    AtomicCounter nextID_;

 	    //////////
	    // The number of Mutexes in the pool
	    int size_;

 	    //////////
	    // The array of Mutex objects.
	    Mutex** pool_;

 	    //////////
	    // Private copy ctor to prevent accidental copying
	    MutexPool(const MutexPool&);
 	    //////////
	    // Private assignment operator to prevent accidental assignment
	    MutexPool& operator=(const MutexPool&);
	};
} // End namespace SCIRun

#endif

