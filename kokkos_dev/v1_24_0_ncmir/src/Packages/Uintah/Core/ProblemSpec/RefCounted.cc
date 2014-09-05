
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
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
static Mutex initlock("RefCounted initialization lock");

static AtomicCounter* nextIndex;
static AtomicCounter* freeIndex;
#include <iostream>
using namespace std;

RefCounted::RefCounted()
    : d_refCount(0)
{
  if(!initialized){
    initlock.lock();
    if(!initialized){
      for(int i=0;i<NLOCKS;i++)
	locks[i] = scinew Mutex("RefCounted Mutex");
      nextIndex=new AtomicCounter("RefCounted nextIndex count", 0);
      freeIndex=new AtomicCounter("RefCounted freeIndex count", 0);
      initialized=true;
    }
    initlock.unlock();
  }
  d_lockIndex = ((*nextIndex)++)%NLOCKS;
  ASSERT(d_lockIndex >= 0);
}

RefCounted::~RefCounted()
{
  ASSERTEQ(d_refCount, 0);
  int index = ++(*freeIndex);
  if(index == *nextIndex){
    initlock.lock();
    if(*freeIndex == *nextIndex){
      initialized = false;
      for(int i=0;i<NLOCKS;i++){
	delete locks[i];
	locks[i]=0;
      }
      delete nextIndex;
      nextIndex=0;
      delete freeIndex;
      freeIndex=0;
    }
    initlock.unlock();
  }
}

void RefCounted::addReference() const
{
    locks[d_lockIndex]->lock();
    d_refCount++;
    locks[d_lockIndex]->unlock();
}

bool RefCounted::removeReference() const
{
    locks[d_lockIndex]->lock();
    bool status = (--d_refCount == 0);
    ASSERT(d_refCount >= 0);
    locks[d_lockIndex]->unlock();
    return status;
}
