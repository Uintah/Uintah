
#include "RefCounted.h"
#include <Core/Thread/AtomicCounter.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
using namespace SCIRun;

static const int    NLOCKS=123;
static       Mutex* locks[NLOCKS];
static       bool   initialized = false;

static AtomicCounter nextIndex("RefCounted nextIndex count", 0);
#include <iostream>
using namespace std;

RefCounted::RefCounted()
    : d_refCount(0)
{
    if(!initialized){
	// This sucks - it needs to be made thread-safe
	for(int i=0;i<NLOCKS;i++)
	    locks[i] = scinew Mutex("RefCounted Mutex");
	initialized=true;
    }
    d_lockIndex = (nextIndex++)%NLOCKS;
    ASSERT(d_lockIndex >= 0);
}

RefCounted::~RefCounted()
{
    ASSERTEQ(d_refCount, 0);
}

void RefCounted::addReference()
{
    locks[d_lockIndex]->lock();
    d_refCount++;
    locks[d_lockIndex]->unlock();
}

bool RefCounted::removeReference()
{
    locks[d_lockIndex]->lock();
    bool status = (--d_refCount == 0);
    ASSERT(d_refCount >= 0);
    locks[d_lockIndex]->unlock();
    return status;
}

