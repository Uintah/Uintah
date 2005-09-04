/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



#include "Object.h"

#include <sci_defs/mpi_defs.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Comm/DT/DTAddress.h>
#include <Core/CCA/PIDL/Reference.h>
#include <Core/CCA/PIDL/ServerContext.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/CCA/PIDL/Warehouse.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/MutexPool.h>
#include <sstream>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <Core/CCA/PIDL/ProxyBase.h>
#include <Core/CCA/spec/sci_sidl.h>
#include <sci_defs/mpi_defs.h> 
#include <sci_mpi.h>

using namespace std;
using namespace SCIRun;

//#define TRACED_OBJ sci::cca::TypeMap
//#define TRACED_OBJ_NAME "TypeMap"

int Object::objcnt(0);
Mutex Object::sm("test mutex");

Object::Object()
    : d_serverContext(0)
{
  firstTime=true;
  sm.lock();
  objid=objcnt++;
  ref_cnt=0;

#ifdef TRACED_OBJ
  if(dynamic_cast<TRACED_OBJ *>(this)){
    int mpi_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    ::std::cout << "----------------------------------\n";
    ::std::cout << "Object ("<<TRACED_OBJ_NAME<<" "<<mpi_rank<<") ref_cnt is now " << ref_cnt << "\n";
  }
#endif
  //mutex_index=getMutexPool()->nextIndex();
  sm.unlock();
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
    throw InternalError("Server reinitialized while endpoint already active?", __FILE__, __LINE__);
  } else if(d_serverContext->d_objptr != this){
    throw InternalError("Server reinitialized with a different base class ptr?", __FILE__, __LINE__);
  }
  // This may happen multiple times, due to multiple inheritance.  It
  // is a "last one wins" approach - the last CTOR to call this function
  // is the most derived type.
  d_serverContext->chan = epc;
  d_serverContext->chan->openConnection();
  d_serverContext->d_typeinfo=typeinfo;
  d_serverContext->d_ptr=ptr;
  //TODO: possible memory leak if this method is called multiple times.
  d_serverContext->storage = new HandlerStorage();
  // gatekeeper is not used anymore.
  //  d_serverContext->gatekeeper = new HandlerGateKeeper();
}

Object::~Object()
{
  if(ref_cnt != 0) {
    throw InternalError("Object delete while reference count != 0", __FILE__, __LINE__);
  }
  if(d_serverContext){
    Warehouse* warehouse=PIDL::getWarehouse();
    if(d_serverContext->d_endpoint_active){
      if(warehouse->unregisterObject(d_serverContext->d_objid) != this)
	throw InternalError("Corruption in object warehouse", __FILE__, __LINE__);
      //EpChannel->closeConnection() does nothing, so far.
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
    throw InternalError("Object::getURL called for a non-server object", __FILE__, __LINE__);
  }
  return (o.str());
}

void
Object::_getReferenceCopy(ReferenceMgr** refM) const
{
  if(!d_serverContext)
    throw InternalError("Object::getReference called for a non-server object", __FILE__, __LINE__);
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
  //Mutex* m=getMutexPool()->getMutex(mutex_index);
  sm.lock();
  ref_cnt++;
#ifdef TRACED_OBJ  
  if(dynamic_cast<TRACED_OBJ *>(this)){
    int mpi_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    ::std::cout << "----------------------------------\n";
    ::std::cout << "_addReference ("<<TRACED_OBJ_NAME<<" "<<mpi_rank<<") ref_cnt is now " << ref_cnt << "\n";
  }
  firstTime=false;
#endif
  sm.unlock();
}

void
Object::_deleteReference()
{
  //Mutex* m=getMutexPool()->getMutex(mutex_index);
  sm.lock();

  ref_cnt--;
#ifdef TRACED_OBJ
  firstTime=false;  
  if(dynamic_cast<sci::cca::ports::UIPort*>(this)){
    int mpi_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    ::std::cout << "----------------------------------\n";
    ::std::cout << "_delReference ("<<TRACED_OBJ_NAME<<" "<<mpi_rank<<") ref_cnt is now " << ref_cnt << "\n";
  }
#endif  
  bool del;
  if(ref_cnt == 0)
    del=true;
  else
    del=false;
  sm.unlock();
  
  // We must delete outside of the lock to prevent deadlock
  // conditions with the mutex pool, but we must check the condition
  // inside the lock to prevent race conditions with other threads
  // simultaneously releasing a reference.
  //TODO: this is just to verify that reference counting is a problem.

  if(del){
    //TODO: 
    //    try to cast the pointer to possible objects such as Typemap, UIPort, goPort etc to find out which one caused the refcnt problem. Review the PHello program and addProvidesPort and related programs first. It may save some time. 
#ifdef TRACED_OBJ
    int mpi_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
    if(dynamic_cast<TRACED_OBJ *>(this)){
      ::std::cout << "############# "<<TRACED_OBJ_NAME<<"("<<mpi_rank<<") deleted \n";
    }
#endif
    delete this;
  }
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
  d_serverContext->d_sched = new SCIRun::MxNScheduler(callee);
}

void Object::setCalleeDistribution(std::string distname, 
				   MxNArrayRep* arrrep) 
{
  //Clear existing distribution
  d_serverContext->d_sched->clear(distname, ProxyID(), callee);
  //Reset distribution
  d_serverContext->d_sched->setCalleeRepresentation(distname,arrrep);
}

void Object::createSubset(int localsize, int remotesize)
{
  ::std::cerr << "createSubset(int,int) has no effect on this object\n";
}

void Object::setRankAndSize(int rank, int size)
{
  ::std::cerr << "setRankAndSize(int,int) has no effect on this object\n";
}

void Object::resetRankAndSize()
{
  ::std::cerr << "resetRankAndSize() has no effect on this object\n";
}

void Object::getException() {}

int Object::getRefCount()
{
  return ref_cnt;
}
