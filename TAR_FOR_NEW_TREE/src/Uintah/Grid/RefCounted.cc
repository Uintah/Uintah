//
// $Id$
//

#include "RefCounted.h"
#include <SCICore/Thread/AtomicCounter.h>
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Util/FancyAssert.h>
#include <SCICore/Malloc/Allocator.h>
using namespace Uintah;
using namespace SCICore::Thread;

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

//
// $Log$
// Revision 1.8  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.7  2000/08/21 23:27:07  sparker
// Added getReferenceCount() method to RefCounted
// Correctly maintain ref counts on neighboring particle subsets in ParticleSubset
//
// Revision 1.6  2000/06/22 21:56:30  sparker
// Changed variable read/write to fortran order
//
// Revision 1.5  2000/05/30 20:19:33  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.4  2000/04/26 06:48:53  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
