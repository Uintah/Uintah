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

#include <Core/Thread/MutexPool.h>
namespace SCIRun {


MutexPool::MutexPool(const char* name, int size)
    :  nextID_("MutexPool ID lock"), size_(size)
{
    // Mutex has no default CTOR so we must allocate them
    // indepdently.
    pool_=new Mutex*[size_];
    for(int i=0;i<size_;i++)
	pool_[i]=new Mutex(name);
}

MutexPool::~MutexPool()
{
    for(int i=0;i<size_;i++)
	delete pool_[i];
    delete[] pool_;
}

int MutexPool::nextIndex()
{
    for(;;) {
	int next=nextID_++;
	if(next < size_)
	    return next;
	// The above is atomic, but if it exceeds size, we need to
	// reset it.
	nextID_.set(0);
    }
}

Mutex* MutexPool::getMutex(int idx)
{
    return pool_[idx];
}

void MutexPool::lockMutex(int idx)
{
    pool_[idx]->lock();
}

void MutexPool::unlockMutex(int idx)
{
    pool_[idx]->unlock();
}


} // End namespace SCIRun
