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


#include "Object.h"

#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/Reference.h>
#include <Core/CCA/PIDL/ServerContext.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/CCA/PIDL/Warehouse.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/MutexPool.h>
#include <sstream>
#include <iostream>

using namespace std;
using namespace SCIRun;

Object::Object()
    : d_serverContext(0)
{
  ref_cnt=0;
  mutex_index=getMutexPool()->nextIndex();
}

void
Object::initializeServer(const TypeInfo* typeinfo, void* ptr, EpChannel* epc)
{
  if(!d_serverContext){
    d_serverContext=new ServerContext;
    d_serverContext->d_endpoint_active=false;
    d_serverContext->d_objptr=this;
    d_serverContext->d_objid=-1;
    d_serverContext->d_sched = 0;
  } else if(d_serverContext->d_endpoint_active){
    throw InternalError("Server reinitialized while endpoint already active?");
  } else if(d_serverContext->d_objptr != this){
    throw InternalError("Server reinitialized with a different base class ptr?");
  }
  // This may happen multiple times, due to multiple inheritance.  It
  // is a "last one wins" approach - the last CTOR to call this function
  // is the most derived type.
  d_serverContext->chan = epc;
  d_serverContext->chan->openConnection();
  d_serverContext->d_typeinfo=typeinfo;
  d_serverContext->d_ptr=ptr;
  d_serverContext->storage = new HandlerStorage();
  d_serverContext->gatekeeper = new HandlerGateKeeper();
}

Object::~Object()
{
  if(ref_cnt != 0)
    throw InternalError("Object delete while reference count != 0");
  if(d_serverContext){
    Warehouse* warehouse=PIDL::getWarehouse();
    if(d_serverContext->d_endpoint_active){
      if(warehouse->unregisterObject(d_serverContext->d_objid) != this)
	throw InternalError("Corruption in object warehouse");
      d_serverContext->chan->closeConnection(); 
    }
    delete d_serverContext->chan;
    if(d_serverContext->d_sched)
      delete d_serverContext->d_sched;
    if(d_serverContext->storage)
      delete d_serverContext->storage;
    delete d_serverContext;
  }
}

URL Object::getURL() const
{
  std::ostringstream o;
  if(d_serverContext){
    if(!d_serverContext->d_endpoint_active)
      activateObject();
    o << d_serverContext->chan->getUrl() 
      << d_serverContext->d_objid;
  } else {
    // TODO - send a message to get the URL
    o << "getURL() doesn't (yet) work for proxy objects";
  }
  return (o.str());
}

void
Object::_getReferenceCopy(ReferenceMgr** refM) const
{
  if(!d_serverContext)
    throw InternalError("Object::getReference called for a non-server object");
  if(!d_serverContext->d_endpoint_active)
    activateObject();
  (*refM) = new ReferenceMgr();
  Reference* ref = new Reference();
  ref->d_vtable_base=TypeInfo::vtable_methods_start;
  d_serverContext->bind(*ref);
  (*refM)->insertReference(ref);
}

void
Object::addReference()
{
  _addReference();
}

void
Object::deleteReference()
{
  _deleteReference();
}

void
Object::_addReference()
{
  Mutex* m=getMutexPool()->getMutex(mutex_index);
  m->lock();
  ref_cnt++;
  //::std::cout << "Reference count is now " << ref_cnt << "\n";
  m->unlock();
}

void
Object::_deleteReference()
{
  Mutex* m=getMutexPool()->getMutex(mutex_index);
  m->lock();
  ref_cnt--;
  //::std::cout << "Reference count is now " << ref_cnt << "\n";
  bool del;
  if(ref_cnt == 0)
    del=true;
  else
    del=false;
  m->unlock();
  
  // We must delete outside of the lock to prevent deadlock
  // conditions with the mutex pool, but we must check the condition
  // inside the lock to prevent race conditions with other threads
  // simultaneously releasing a reference.
  if(del)
    delete this;
}

void
Object::activateObject() const
{
  Warehouse* warehouse=PIDL::getWarehouse();
  if(PIDL::isNexus()){
    d_serverContext->d_objid=warehouse->registerObject(const_cast<Object*>(this));
    d_serverContext->activateEndpoint();
  }
  else{
    d_serverContext->activateEndpoint();
    warehouse->registerObject(d_serverContext->d_objid, const_cast<Object*>(this));
  }

}

MutexPool*
Object::getMutexPool()
{
  static MutexPool* pool=0;
  if(!pool){
    // TODO - make this threadsafe.  This can leak if two threads
    // happen to request the first pool at the same time.  I doubt
    // it will ever happen - sparker.
    pool=new MutexPool("Core/CCA::PIDL::Object mutex pool", 63);
  }
  return pool;
}

void Object::createScheduler()
{
  d_serverContext->d_sched = new SCIRun::MxNScheduler();
}

void Object::setCalleeDistribution(std::string distname, 
				   MxNArrayRep* arrrep) 
{
  d_serverContext->d_sched->setCalleeRepresentation(distname,arrrep);
}

void Object::createSubset(int localsize, int remotesize)
{
  ::std::cerr << "createSubset(int,int) has no effect on this object\n";
}

void Object::getException() {}

int Object::getRefCount()
{
  return ref_cnt;
}
