/* REFERENCED */
static char *id="@(#) $Id$";

#include "RefCounted.h"
#include <SCICore/Thread/AtomicCounter.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Util/Assert.h>
using namespace Uintah;
using namespace SCICore::Thread;

static const int    NLOCKS=123;
static       Mutex* locks[NLOCKS];
static       bool   initialized = false;

static AtomicCounter nextIndex("RefCounted nextIndex count");

RefCounted::RefCounted()
    : d_refCount(0)
{
    if(!initialized){
	// This sucks - it needs to be made thread-safe
	for(int i=0;i<NLOCKS;i++)
	    locks[i] = new Mutex("RefCounted Mutex");
	initialized=true;
    }
    d_lockIndex = (nextIndex++)%NLOCKS;
}

RefCounted::~RefCounted()
{
    ASSERT(d_refCount == 0);
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

//
// $Log$
// Revision 1.4  2000/04/26 06:48:53  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
