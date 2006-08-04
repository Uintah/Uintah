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
 *  Parallel3: Automatically instantiate several threads, with 3 arguments
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 3997
 *
 *  Copyright (C) 3997 SCI Group
 */

#ifndef Core_Thread_Parallel3_h
#define Core_Thread_Parallel3_h

#include <Core/Thread/ParallelBase.h>
#include <Core/Thread/Semaphore.h>

namespace SCIRun {
/**************************************

 CLASS
 Parallel3

 KEYWORDS
 Thread

 DESCRIPTION
 Helper class to make instantiating threads to perform a parallel
 task easier.
   
****************************************/
template<class T, class Arg1, class Arg2, class Arg3> class Parallel3  : public ParallelBase {
public:
  //////////
  // Create a Parallel3 object, using the specified member
  // function instead of <i>Parallel3</i>.  This will
  // typically be used like:
  // <b><pre>Thread::Parallel3(Parallel3&lt;MyClass&gt;(this, &amp;MyClass::mymemberfn), nthreads);</pre></b>
  Parallel3(T* obj, void (T::*pmf)(int, Arg1, Arg2, Arg3), Arg1 a1, Arg2 a2, Arg3 a3);
	    
  //////////
  // Destroy the Parallel3 object - the threads will remain alive.
  virtual ~Parallel3();
  T* obj_;
  void (T::*pmf_)(int, Arg1, Arg2, Arg3);
  Arg1 a1;
  Arg2 a2;
  Arg3 a3;
protected:
  virtual void run(int proc);
private:
  // Cannot copy them
  Parallel3(const Parallel3&);
  Parallel3<T, Arg1, Arg2, Arg3>& operator=(const Parallel3<T, Arg1, Arg2, Arg3>&);
};

template<class T, class Arg1, class Arg2, class Arg3>
void
Parallel3<T, Arg1, Arg2, Arg3>::run(int proc)
{
    // Copy out do make sure that the call is atomic
    T* obj=obj_;
    void (T::*pmf)(int, Arg1, Arg2, Arg3) = pmf_;
    if(wait_)
	wait_->up();
    (obj->*pmf)(proc, a1, a2, a3);
    // Cannot do anything here, since the object may be deleted by the
    // time we return
}

template<class T, class Arg1, class Arg2, class Arg3>
Parallel3<T, Arg1, Arg2, Arg3>::Parallel3(T* obj,
							   void (T::*pmf)(int, Arg1, Arg2, Arg3),
							   Arg1 a1, Arg2 a2, Arg3 a3)
    : obj_(obj), pmf_(pmf), a1(a1), a2(a2), a3(a3)
{
    wait_=0; // This may be set by Thread::parallel
} // End namespace SCIRun

template<class T, class Arg1, class Arg2, class Arg3>
Parallel3<T, Arg1, Arg2, Arg3>::~Parallel3()
{
}

} //End namespace SCIRun

#endif


