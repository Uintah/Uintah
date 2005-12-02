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
 *  Parallel: Automatically instantiate several threads
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Thread_Parallel_h
#define Core_Thread_Parallel_h

#include <Core/Thread/ParallelBase.h>
#include <Core/Thread/Semaphore.h>

namespace SCIRun {
/**************************************
 
				       CLASS
				       Parallel
   
				       KEYWORDS
				       Thread
   
				       DESCRIPTION
				       Helper class to make instantiating threads to perform a parallel
				       task easier.
   
****************************************/
template<class T> class Parallel  : public ParallelBase {
public:
  //////////
  // Create a parallel object, using the specified member
  // function instead of <i>parallel</i>.  This will
  // typically be used like:
  // <b><pre>Thread::parallel(Parallel&lt;MyClass&gt;(this, &amp;MyClass::mymemberfn), nthreads);</pre></b>
  Parallel(T* obj, void (T::*pmf)(int));
	    
  //////////
  // Destroy the Parallel object - the threads will remain alive.
  virtual ~Parallel();
  T* obj_;
  void (T::*pmf_)(int);
protected:
  virtual void run(int proc);
private:
  // Cannot copy them
  Parallel(const Parallel&);
  Parallel<T>& operator=(const Parallel<T>&);
};

template<class T>
void
Parallel<T>::run(int proc)
{
    // Copy out do make sure that the call is atomic
    T* obj=obj_;
    void (T::*pmf)(int) = pmf_;
    if(wait_)
	wait_->up();
    (obj->*pmf)(proc);
    // Cannot do anything here, since the object may be deleted by the
    // time we return
}

template<class T>
Parallel<T>::Parallel(T* obj, void (T::*pmf)(int))
    : obj_(obj), pmf_(pmf)
{
    wait_=0; // This may be set by Thread::parallel
} // End namespace SCIRun

template<class T>
Parallel<T>::~Parallel()
{
}

} //End namespace SCIRun

#endif




