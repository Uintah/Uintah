
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
