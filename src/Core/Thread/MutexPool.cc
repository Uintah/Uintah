
/*
 *  MutexPool: A set of mutex objects
 *  $Id$
 *
 *  Written by:
 *   Author: Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Date: September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Thread/MutexPool.h>

using SCICore::Thread::MutexPool;
using SCICore::Thread::Mutex;

MutexPool::MutexPool(const char* name, int size)
    :  d_nextID("MutexPool ID lock"), d_size(size)
{
    // Mutex has no default CTOR so we must allocate them
    // indepdently.
    d_pool=new Mutex*[d_size];
    for(int i=0;i<d_size;i++)
	d_pool[i]=new Mutex(name);
}

MutexPool::~MutexPool()
{
    for(int i=0;i<d_size;i++)
	delete d_pool[i];
    delete[] d_pool;
}

int MutexPool::nextIndex()
{
    for(;;) {
	int next=d_nextID++;
	if(next < d_size)
	    return next;
	// The above is atomic, but if it exceeds size, we need to
	// reset it.
	d_nextID.set(0);
    }
}

Mutex* MutexPool::getMutex(int idx)
{
    return d_pool[idx];
}

void MutexPool::lockMutex(int idx)
{
    d_pool[idx]->lock();
}

void MutexPool::unlockMutex(int idx)
{
    d_pool[idx]->unlock();
}

//
// $Log$
// Revision 1.1  1999/09/25 08:29:29  sparker
// Added MutexPool class - a utility for sharing Mutexes among a large
//  number of objects
// Fixed comments in Guard
//
//
