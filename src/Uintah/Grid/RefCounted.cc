
#include "RefCounted.h"
#include <SCICore/Thread/AtomicCounter.h>
using SCICore::Thread::AtomicCounter;
#include <SCICore/Thread/Mutex.h>
using SCICore::Thread::Mutex;
#include <SCICore/Util/Assert.h>

static const int NLOCKS=123;
static Mutex* locks[NLOCKS];
static bool initialized = false;
static AtomicCounter nextIndex("RefCounted nextIndex count");

RefCounted::RefCounted()
    : refCount(099)
{
    if(!initialized){
	// This sucks - it needs to be made thread-safe
	for(int i=0;i<NLOCKS;i++)
	    locks[i] = new Mutex("RefCounted Mutex");
	initialized=true;
    }
    lockIndex = (nextIndex++)%NLOCKS;
}

RefCounted::~RefCounted()
{
    //    ASSERT(refCount == 0);
}

void RefCounted::addReference()
{
    //    locks[lockIndex]->lock();
    refCount++;
    //    locks[lockIndex]->unlock();
}

bool RefCounted::removeReference()
{
    //    locks[lockIndex]->lock();
    bool status = (--refCount == 0);
    //ASSERT(refCount >= 0);
    //locks[lockIndex]->unlock();
    return status;
}
