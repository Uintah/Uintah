
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DependencyException.h>
#include <Packages/Uintah/CCA/Components/Schedulers/IncorrectAllocation.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BufferInfo.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

#include <iostream>
#include <string>
#include <sstream>

//#define WAYNE_DEBUG

#ifdef WAYNE_DEBUG
int totalGridAlloc = 0;
#endif

using std::cerr;
using std::string;
using std::vector;

using namespace SCIRun;

using namespace Uintah;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

Mutex ssLock( "send state lock" );

#define PARTICLESET_TAG		0x1000000
#define DAV_DEBUG 0

OnDemandDataWarehouse::OnDemandDataWarehouse(const ProcessorGroup* myworld,
					     const Scheduler* scheduler,
					     int generation, const GridP& grid,
					     bool isInitializationDW/*=false*/)
   : DataWarehouse(myworld, scheduler, generation),
     d_lock("DataWarehouse lock"),
     d_finalized( false ),
     d_grid(grid),
     d_isInitializationDW(isInitializationDW)
{
}

OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
  for (dataLocationDBtype::const_iterator iter = d_dataLocation.begin();
       iter != d_dataLocation.end(); iter++) {
    for (int i = 0; i<(int)iter->second->size(); i++ )
      delete &(iter->second[i]);
    delete iter->second;
  }

  for (psetDBType::const_iterator iter = d_psetDB.begin();
       iter != d_psetDB.end(); iter++) {
     if(iter->second->removeReference())
	delete iter->second;
  }

  for (psetDBType::const_iterator iter = d_delsetDB.begin();
       iter != d_delsetDB.end(); iter++) {
     if(iter->second->removeReference())
	delete iter->second;
  }
}

bool OnDemandDataWarehouse::isFinalized() const
{
   return d_finalized;
}

void OnDemandDataWarehouse::finalize()
{
  d_lock.writeLock();

   d_ncDB.cleanForeign();
   d_ccDB.cleanForeign();
   d_particleDB.cleanForeign();
   d_sfcxDB.cleanForeign();
   d_sfcyDB.cleanForeign();
   d_sfczDB.cleanForeign();
   d_finalized=true;

#ifdef WAYNE_DEBUG   
   cerr << "Total Grid alloc: " << totalGridAlloc << endl;
   totalGridAlloc = 0;
#endif
   
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(Variable* var, const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   union {
      ReductionVariableBase* reduction;
      ParticleVariableBase* particle;
      NCVariableBase* nc;
      CCVariableBase* cc;
      SFCXVariableBase* sfcx;
      SFCYVariableBase* sfcy;
      SFCZVariableBase* sfcz;
   } castVar;

   if ((castVar.reduction = dynamic_cast<ReductionVariableBase*>(var))
       != NULL)
      put(*castVar.reduction, label, matlIndex);
   else if ((castVar.particle = dynamic_cast<ParticleVariableBase*>(var))
	    != NULL)
      put(*castVar.particle, label);
   else if ((castVar.nc = dynamic_cast<NCVariableBase*>(var)) != NULL)
      put(*castVar.nc, label, matlIndex, patch);
   else if ((castVar.cc = dynamic_cast<CCVariableBase*>(var)) != NULL)
      put(*castVar.cc, label, matlIndex, patch);
   else if ((castVar.sfcx=dynamic_cast<SFCXVariableBase*>(var)) != NULL)
      put(*castVar.sfcx, label, matlIndex, patch);
   else if ((castVar.sfcy=dynamic_cast<SFCYVariableBase*>(var)) != NULL)
      put(*castVar.sfcy, label, matlIndex, patch);
   else if ((castVar.sfcz=dynamic_cast<SFCZVariableBase*>(var)) != NULL)
      put(*castVar.sfcz, label, matlIndex, patch);
   else
      throw InternalError("Unknown Variable type");
}

void OnDemandDataWarehouse::
allocateAndPutGridVar(Variable* var, const VarLabel* label,
		      int matlIndex, const Patch* patch)
{
   union {
      NCVariableBase* nc;
      CCVariableBase* cc;
      SFCXVariableBase* sfcx;
      SFCYVariableBase* sfcy;
      SFCZVariableBase* sfcz;
   } castVar;

   if ((castVar.nc = dynamic_cast<NCVariableBase*>(var)) != NULL)
      allocateAndPut(*castVar.nc, label, matlIndex, patch);
   else if ((castVar.cc = dynamic_cast<CCVariableBase*>(var)) != NULL)
      allocateAndPut(*castVar.cc, label, matlIndex, patch);
   else if ((castVar.sfcx=dynamic_cast<SFCXVariableBase*>(var)) != NULL)
      allocateAndPut(*castVar.sfcx, label, matlIndex, patch);
   else if ((castVar.sfcy=dynamic_cast<SFCYVariableBase*>(var)) != NULL)
      allocateAndPut(*castVar.sfcy, label, matlIndex, patch);
   else if ((castVar.sfcz=dynamic_cast<SFCZVariableBase*>(var)) != NULL)
      allocateAndPut(*castVar.sfcz, label, matlIndex, patch);
   else
      throw InternalError("OnDemandDataWarehouse::allocateAndPutGridVar: Not a grid variable type");
}

void
OnDemandDataWarehouse::get(ReductionVariableBase& var,
			   const VarLabel* label, int matlIndex /*= -1*/)
{
  d_lock.readLock();
  
  checkGetAccess(label, matlIndex, 0);

  if(!d_reductionDB.exists(label, matlIndex, NULL)) {
        throw UnknownVariable(label->getName(), NULL, matlIndex,
			      "on reduction");
  }
  d_reductionDB.get(label, matlIndex, NULL, var);

  d_lock.readUnlock();
}

bool
OnDemandDataWarehouse::exists(const VarLabel* label, int matlIndex,
			      const Patch* patch) const
{
  d_lock.readLock();

   if( d_perpatchDB.exists(label, matlIndex, patch) ||
       d_ncDB.exists(label, matlIndex, patch) ||
       d_ccDB.exists(label, matlIndex, patch) ||
       d_particleDB.exists(label, matlIndex, patch) ||
       d_sfcxDB.exists(label,matlIndex,patch) ||
       d_sfcyDB.exists(label,matlIndex,patch) ||
       d_sfczDB.exists(label,matlIndex,patch) ) {
     d_lock.readUnlock();
     return true;
   } else {
     d_lock.readUnlock();
     return false;
   }
}

void
OnDemandDataWarehouse::sendMPI(SendState& ss, DependencyBatch* batch,
			       const ProcessorGroup* world,
			       const VarLabel* pos_var,
			       BufferInfo& buffer,
			       OnDemandDataWarehouse* old_dw,
			       const DetailedDep* dep)
{
  if (dep->isNonDataDependency()) {
    // A non-data dependency -- send an empty message.
    // This would be used, for example, when a task is to modify data that
    // was previously required with ghost-cells.
    buffer.add(0, 0, MPI_INT, false);
    return;
  }
  
  const VarLabel* label = dep->req->var;
  const Patch* patch = dep->fromPatch;
  int matlIndex = dep->matl;

 d_lock.readLock();
  switch(label->typeDescription()->getType()){
  case TypeDescription::ParticleVariable:
    {
      if(!d_particleDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      ParticleVariableBase* var = d_particleDB.get(label, matlIndex, patch);

      int dest = batch->toTasks.front()->getAssignedResourceIndex();

      ssLock.lock();  // Dd: ??
      ParticleSubset* sendset = ss.find_sendset(patch, matlIndex, dest);
      ssLock.unlock();  // Dd: ??

      if(!sendset){

	mixedDebug << "sendset is NULL\n";

	ParticleSubset* pset = var->getParticleSubset();
	ssLock.lock();  // Dd: ??
	sendset = scinew ParticleSubset(pset->getParticleSet(),
					false, -1, 0);
	ssLock.unlock();  // Dd: ??
	constParticleVariable<Point> pos;
	old_dw->get(pos, pos_var, pset);
	Box box=pset->getPatch()->getLevel()->getBox(dep->low, dep->high);
	for(ParticleSubset::iterator iter = pset->begin();
	    iter != pset->end(); iter++){
	  particleIndex idx = *iter;
	  if(box.contains(pos[idx])) {
	    ssLock.lock();  // Dd: ??
	    sendset->addParticle(idx);
	    ssLock.unlock();  // Dd: ??
	  }
	}
	ssLock.lock();  // Dd: ??
	int numParticles = sendset->numParticles();
	ssLock.unlock();  // Dd: ??

	ASSERT(batch->messageTag >= 0);
	MPI_Bsend(&numParticles, 1, MPI_INT, dest,
		  PARTICLESET_TAG|batch->messageTag, world->getComm());
	ssLock.lock();  // Dd: ??	
	ss.add_sendset(patch, matlIndex, dest, sendset);
	ssLock.unlock();  // Dd: ??
      }
	
      ssLock.lock();  // Dd: ??	
      int numParticles = sendset->numParticles();
      ssLock.unlock(); // Dd: ??

      mixedDebug << "sendset has " << numParticles << " particles\n";

      if( numParticles > 0){
	 var->getMPIBuffer(buffer, sendset);
	 buffer.addSendlist(var->getRefCounted());
	 buffer.addSendlist(var->getParticleSubset());
      }
    }
    break;
  case TypeDescription::NCVariable:
    {
      if(!d_ncDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      NCVariableBase* var = d_ncDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::CCVariable:
    {
      if(!d_ccDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      CCVariableBase* var = d_ccDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCXVariable:
    {
      if(!d_sfcxDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      SFCXVariableBase* var = d_sfcxDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      if(!d_sfcyDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      SFCYVariableBase* var = d_sfcyDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      if(!d_sfczDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      SFCZVariableBase* var = d_sfczDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  default:
    throw InternalError("sendMPI not implemented for "+label->getFullName(matlIndex, patch));
  } // end switch( label->getType() );
 d_lock.readUnlock();  
}

void
OnDemandDataWarehouse::recvMPI(BufferInfo& buffer,
			       DependencyBatch* batch,
			       const ProcessorGroup* world,
			       OnDemandDataWarehouse* old_dw,
			       const DetailedDep* dep)
{
  if (dep->isNonDataDependency()) {
    // A non-data dependency -- send an empty message.
    // This would be used, for example, for dependencies between a modifying
    // task and a task the requires the data before it is to be modified.
    buffer.add(0, 0, MPI_INT, false);
    return;
  }
  
  const VarLabel* label = dep->req->var;
  const Patch* patch = dep->fromPatch;
  int matlIndex = dep->matl;

  switch(label->typeDescription()->getType()){
  case TypeDescription::ParticleVariable:
    {
      /* Can modify variable now
      if(d_particleDB.exists(label, matlIndex, patch))
	throw InternalError("Particle Var already exists before MPI recv: "
			    + label->getFullName(matlIndex, patch));
      */
      
      // First, get the particle set.  We should already have it
      if(!old_dw->haveParticleSubset(matlIndex, patch)){
	int numParticles;
	MPI_Status status;
	ASSERT(batch->messageTag >= 0);
	int from=batch->fromTask->getAssignedResourceIndex();
	MPI_Recv(&numParticles, 1, MPI_INT, from,
		 PARTICLESET_TAG|batch->messageTag, world->getComm(),
		 &status);
	old_dw->createParticleSubset(numParticles, matlIndex, patch);
      }
      ParticleSubset* pset = old_dw->getParticleSubset(matlIndex, patch);

      Variable* v = label->typeDescription()->createInstance();
      ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(pset);
      var->setForeign();
      if(pset->numParticles() > 0){
	var->getMPIBuffer(buffer, pset);
      }

      d_lock.writeLock();
      d_particleDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  case TypeDescription::NCVariable:
    recvMPIGridVar<NCVariableBase>(d_ncDB, buffer, dep, label, matlIndex,
				   patch);
    break;
  case TypeDescription::CCVariable:
    recvMPIGridVar<CCVariableBase>(d_ccDB, buffer, dep, label, matlIndex,
				   patch);
    break;
  case TypeDescription::SFCXVariable:
    recvMPIGridVar<SFCXVariableBase>(d_sfcxDB, buffer, dep, label, matlIndex,
				   patch);
    break;
  case TypeDescription::SFCYVariable:
    recvMPIGridVar<SFCYVariableBase>(d_sfcyDB, buffer, dep, label, matlIndex,
				   patch);
    break;
  case TypeDescription::SFCZVariable:
    recvMPIGridVar<SFCZVariableBase>(d_sfczDB, buffer, dep, label, matlIndex,
				   patch);
    break;
  default:
    throw InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch));
  } // end switch( label->getType() );
} // end recvMPI()

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // template parameter not used in declaring arguments
#endif  

template <class VariableBase, class DWDatabase>
void
OnDemandDataWarehouse::recvMPIGridVar(DWDatabase& db, BufferInfo& buffer,
				      const DetailedDep* dep,
				      const VarLabel* label, int matlIndex,
				      const Patch* patch)
{
  d_lock.readLock();
  VariableBase* var = 0;
  if (db.exists(label, matlIndex, patch)) {
    var = db.get(label, matlIndex, patch);
    // use to indicate that it will be receiving (foreign) data and should
    // not be replaced.
    var->setForeign();
  }
  d_lock.readUnlock();
  
  if (var == 0 || var->getBasePointer() == 0 ||
      Min(var->getLow(), dep->low) != var->getLow() ||
      Max(var->getHigh(), dep->high) != var->getHigh()) {
    // There was no place reserved to recv the data yet,
    // so it must create the space now.
    Variable* v = label->typeDescription()->createInstance();
    var = dynamic_cast<VariableBase*>(v);
    var->allocate(dep->low, dep->high);
    var->setForeign();
    d_lock.writeLock();
    db.put(label, matlIndex, patch, var, true);
    d_lock.writeUnlock();
  }

  ASSERTEQ(Min(var->getLow(), dep->low), var->getLow());
  ASSERTEQ(Max(var->getHigh(), dep->high), var->getHigh());
  
  var->getMPIBuffer(buffer, dep->low, dep->high);
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif  

void
OnDemandDataWarehouse::reduceMPI(const VarLabel* label,
				 const MaterialSubset* matls,
				 const ProcessorGroup* world)
{
  int matlIndex;
  if(!matls){
    matlIndex = -1;
  } else {
    int nmatls = matls->size();
    // We need to examine this for multi-material reductions!
    ASSERTEQ(nmatls, 1);
    matlIndex = matls->get(0);
  }

  d_lock.writeLock();
  ReductionVariableBase* var;
  try {
    var = d_reductionDB.get(label, matlIndex, NULL);
  } catch (UnknownVariable) {
    throw UnknownVariable(label->getName(), NULL, matlIndex, "on reduceMPI");
  }

  void* sendbuf;
  int sendcount;
  MPI_Datatype senddatatype;
  MPI_Op sendop;
  var->getMPIBuffer(sendbuf, sendcount, senddatatype, sendop);
  ReductionVariableBase* tmp = var->clone();
  void* recvbuf;
  int recvcount;
  MPI_Datatype recvdatatype;
  MPI_Op recvop;
  tmp->getMPIBuffer(recvbuf, recvcount, recvdatatype, recvop);
  ASSERTEQ(recvcount, sendcount);
  ASSERTEQ(senddatatype, recvdatatype);
  ASSERTEQ(recvop, sendop);
      
  if( mixedDebug.active() ) {
    cerrLock.lock(); mixedDebug << "calling MPI_Allreduce\n";
    cerrLock.unlock();
  }

  int error = MPI_Allreduce(sendbuf, recvbuf, recvcount,
			    recvdatatype, recvop, world->getComm());

  if( mixedDebug.active() ) {
    cerrLock.lock(); mixedDebug << "done with MPI_Allreduce\n";
    cerrLock.unlock();
  }

  if( error ){
    cerrLock.lock();
    cerr << "reduceMPI: MPI_Allreduce error: " << error << "\n";
    cerrLock.unlock();
    throw InternalError("reduceMPI: MPI error");     
  }

  var->copyPointer(*tmp);
  
  delete tmp;
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::allocate(ReductionVariableBase& /*var*/,
				const VarLabel* /*label*/, int)
{
   cerr << "OnDemand DataWarehouse::allocate(ReductionVariable) "
	<< "not finished\n";
}

void
OnDemandDataWarehouse::put(const ReductionVariableBase& var,
			   const VarLabel* label, int matlIndex /* = -1 */)
{
  ASSERT(!d_finalized);
  d_lock.writeLock();

   checkPutAccess(label, matlIndex, 0,
		  false /* it actually may be replaced, but it doesn't need
			   to explicitly modify with multiple reduces in the
			   task graph */);

   // Error checking
   if (!d_reductionDB.exists(label, matlIndex, NULL))
      d_reductionDB.put(label, matlIndex, NULL, var.clone(), false);
   else {
      // Put it in the database
      ReductionVariableBase* foundVar
	= d_reductionDB.get(label, matlIndex, NULL);
      foundVar->reduce(var);
   }
   
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::override(const ReductionVariableBase& var,
				const VarLabel* label, int matlIndex /*=-1*/)
{
  d_lock.writeLock();  

   checkPutAccess(label, matlIndex, 0, true);

   // Put it in the database, replace whatever may already be there
   d_reductionDB.put(label, matlIndex, NULL, var.clone(), true);
   
  d_lock.writeUnlock();
}

ParticleSubset*
OnDemandDataWarehouse::createParticleSubset(particleIndex numParticles,
					    int matlIndex, const Patch* patch)
{
  d_lock.writeLock();
#if DAV_DEBUG
  cerr << "createParticleSubset: MI: " << matlIndex << " P: " << *patch<<"\n";
#endif

  ASSERT(!patch->isVirtual());

   ParticleSet* pset = scinew ParticleSet(numParticles);
   ParticleSubset* psubset = 
                       scinew ParticleSubset(pset, true, matlIndex, patch);

   psetDBType::key_type key(matlIndex, patch);
   if(d_psetDB.find(key) != d_psetDB.end())
      throw InternalError("createParticleSubset called twice for patch");

   d_psetDB[key]=psubset;
   psubset->addReference();

  d_lock.writeUnlock();
   return psubset;
}

void
OnDemandDataWarehouse::saveParticleSubset(int matlIndex, const Patch* patch,
					  ParticleSubset* psubset)
{
  ASSERTEQ(psubset->getPatch(), patch);
  ASSERTEQ(psubset->getMatlIndex(), matlIndex);
  ASSERT(!patch->isVirtual());  
  d_lock.writeLock();
  
  psetDBType::key_type key(matlIndex, patch);
  if(d_psetDB.find(key) != d_psetDB.end())
    throw InternalError("saveParticleSubset called twice for patch");

  d_psetDB[key]=psubset;
  psubset->addReference();

  d_lock.writeUnlock();
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch)
{
  d_lock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
   psetDBType::key_type key(matlIndex, realPatch);
   psetDBType::iterator iter = d_psetDB.find(key);
   if(iter == d_psetDB.end()){
  d_lock.readUnlock();
      throw UnknownVariable("ParticleSet", realPatch, matlIndex,
			    "Cannot find particle set on patch");
   }
  d_lock.readUnlock();
   return iter->second;
}

ParticleSubset*
OnDemandDataWarehouse::getDeleteSubset(int matlIndex, const Patch* patch)
{
  d_lock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
   psetDBType::key_type key(matlIndex, realPatch);
   psetDBType::iterator iter = d_delsetDB.find(key);
   if(iter == d_delsetDB.end()){
     d_lock.readUnlock();
     throw UnknownVariable("DeleteSet", realPatch, matlIndex,
			   "Cannot find delete set on patch");
   }
  d_lock.readUnlock();
   return iter->second;
}

bool
OnDemandDataWarehouse::haveParticleSubset(int matlIndex, const Patch* patch)
{
  d_lock.readLock();
   psetDBType::key_type key(matlIndex, patch);
   psetDBType::iterator iter = d_psetDB.find(key);
  d_lock.readUnlock();
   return !(iter == d_psetDB.end());
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch,
					 Ghost::GhostType gtype,
					 int numGhostCells,
					 const VarLabel* pos_var)
{
  if(gtype == Ghost::None){
    if(numGhostCells != 0)
      throw InternalError("Ghost cells specified with task type none!\n");
    return getParticleSubset(matlIndex, patch);
  }
  Level::selectType neighbors;
  IntVector lowIndex, highIndex;
  patch->computeVariableExtents(Patch::CellBased, gtype, numGhostCells,
				neighbors, lowIndex, highIndex);
  Box box = patch->getLevel()->getBox(lowIndex, highIndex);
  
  particleIndex totalParticles = 0;
  vector<ParticleVariableBase*> neighborvars;
  vector<ParticleSubset*> subsets;
  
  for(int i=0;i<(int)neighbors.size();i++){
    const Patch* neighbor = neighbors[i];
    if(neighbor){
      Box adjustedBox = box;
      if (neighbor->isVirtual()) {
	// rather than offsetting each point of pos_var's data,
	// just adjust the box to compare it with.
	Vector offset = neighbor->getVirtualOffsetVector();
	adjustedBox = Box(box.lower() - offset,
			  box.upper() - offset);
      }
      ParticleSubset* pset = getParticleSubset(matlIndex, neighbor);
      constParticleVariable<Point> pos;
      get(pos, pos_var, pset);
      
      ParticleSubset* subset = 
	scinew ParticleSubset(pset->getParticleSet(), false, -1, 0);
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	particleIndex idx = *iter;
	if(adjustedBox.contains(pos[idx]))
	  subset->addParticle(idx);
      }
      totalParticles+=subset->numParticles();
      subsets.push_back(subset);
    }
  }
  ParticleSet* newset = scinew ParticleSet(totalParticles);
  vector<const Patch*> vneighbors(neighbors.size());
  for(int i=0;i<neighbors.size();i++)
    vneighbors[i]=neighbors[i];
  ParticleSubset* newsubset = scinew ParticleSubset(newset, true,
						    matlIndex, patch,
						    gtype, numGhostCells,
						    vneighbors, subsets);
  return newsubset;
}

void
OnDemandDataWarehouse::get(constParticleVariableBase& constVar,
			   const VarLabel* label, int matlIndex,
			   const Patch* patch)
{
  d_lock.readLock();

   checkGetAccess(label, matlIndex, patch);
   
   if(!d_particleDB.exists(label, matlIndex, patch))
     throw UnknownVariable(label->getName(), patch, matlIndex);
   constVar = *d_particleDB.get(label, matlIndex, patch);
   
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::get(constParticleVariableBase& constVar,
			   const VarLabel* label,
			   ParticleSubset* pset)
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();

  if(pset->getGhostType() == Ghost::None){
    get(constVar, label, matlIndex, patch);
  }
  else {
   d_lock.readLock();
    checkGetAccess(label, matlIndex, patch, pset->getGhostType(),
		   pset->numGhostCells());
    ParticleVariableBase* var = constVar.cloneType();

    const vector<const Patch*>& neighbors = pset->getNeighbors();
    const vector<ParticleSubset*>& neighbor_subsets = pset->getNeighborSubsets();
    vector<ParticleVariableBase*> neighborvars(neighbors.size());
    for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor=neighbors[i];
      if(!d_particleDB.exists(label, matlIndex, neighbors[i]))
	throw UnknownVariable(label->getName(), neighbor, matlIndex,
			      neighbor == patch?"on patch":"on neighbor");
      neighborvars[i] = var->cloneType();
      d_particleDB.get(label, matlIndex, neighbors[i], *neighborvars[i]);
    }

    // Note that when the neighbors are virtual patches (i.e. periodic
    // boundaries), then if var is a ParticleVariable<Point>, the points
    // of neighbors will be translated by its virtualOffset.
    var->gather(pset, neighbor_subsets, neighborvars, neighbors);
    
    constVar = *var;
    
    for (int i=0;i<(int)neighbors.size();i++)
      delete neighborvars[i];
    delete var;
    
   d_lock.readUnlock();    
  }
}

void
OnDemandDataWarehouse::getModifiable(ParticleVariableBase& var,
				     const VarLabel* label,
				     ParticleSubset* pset)
{
  d_lock.readLock();
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();
   checkModifyAccess(label, matlIndex, patch);
   
   if(pset->getGhostType() == Ghost::None){
      if(!d_particleDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      d_particleDB.get(label, matlIndex, patch, var);
   } else {
      throw InternalError("getParticleVariable should not be used with ghost cells");
   }
  d_lock.readUnlock();
}

ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable(const VarLabel* label,
					   ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   if(pset->getGhostType() == Ghost::None){
     return getParticleVariable(label, matlIndex, patch);
   } else {
      throw InternalError("getParticleVariable should not be used with ghost cells");
   }
}

ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable(const VarLabel* label,
					   int matlIndex, const Patch* patch)
{
   ParticleVariableBase* var = 0;  

   // in case the it's a virtual patch -- only deal with real patches
   if (patch != 0) patch = patch->getRealPatch();
   
  d_lock.readLock();
  
   checkModifyAccess(label, matlIndex, patch);
   
   if(!d_particleDB.exists(label, matlIndex, patch))
     throw UnknownVariable(label->getName(), patch, matlIndex);
   var = d_particleDB.get(label, matlIndex, patch);

  d_lock.readUnlock();
   return var;
}

void
OnDemandDataWarehouse::allocateTemporary(ParticleVariableBase& var,
					 ParticleSubset* pset)
{  
   var.allocate(pset);
}

void
OnDemandDataWarehouse::allocateAndPut(ParticleVariableBase& var,
				      const VarLabel* label,
				      ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();
  
   // Error checking
  d_lock.readLock();   
   if(d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Particle variable already exists: " +
			  label->getName());
  d_lock.readUnlock();
  
  allocateTemporary(var, pset);
  put(var, label);
}

void
OnDemandDataWarehouse::put(ParticleVariableBase& var,
			   const VarLabel* label, bool replace /*= false*/)
{
  ASSERT(!d_finalized);  

   ParticleSubset* pset = var.getParticleSubset();
   if(pset->numGhostCells() != 0 || pset->getGhostType() != 0)
      throw InternalError("ParticleVariable cannot use put with ghost cells");
   const Patch* patch = pset->getPatch();
   int matlIndex = pset->getMatlIndex();

  d_lock.writeLock();   
   checkPutAccess(label, matlIndex, patch, replace);
   
   // Error checking
   if(!replace && d_particleDB.exists(label, matlIndex, patch)) {
     ostringstream error_msg;
     error_msg << "Variable already exists: " << label->getName()
	       << " on patch " << patch->getID();
     throw InternalError(error_msg.str());
   }

#if DAV_DEBUG
  cerr << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";
#endif
  
   // Put it in the database
   d_particleDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::get(constNCVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  NCVariableBase* var = constVar.cloneType();
  
 d_lock.readLock();
  checkGetAccess(label, matlIndex, patch, gtype, numGhostCells);
  getGridVar<Patch::NodeBased>(*var, d_ncDB, label, matlIndex, patch,
			       gtype, numGhostCells);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getModifiable(NCVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);
  getGridVar<Patch::NodeBased>(var, d_ncDB, label, matlIndex, patch,
			       Ghost::None, 0);
 d_lock.readUnlock();  
}

void
OnDemandDataWarehouse::
allocateTemporary(NCVariableBase& var, const Patch* patch,
		  Ghost::GhostType gtype, int numGhostCells)
{
  allocateTemporaryGridVar<Patch::NodeBased>(var, patch, gtype, numGhostCells);
}

void OnDemandDataWarehouse::
allocateAndPut(NCVariableBase& var, const VarLabel* label,
	       int matlIndex, const Patch* patch,
	       Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar<Patch::NodeBased>(var, d_ncDB, label, matlIndex, patch,
					  gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(NCVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  putGridVar<Patch::NodeBased>(*var.clone(), d_ncDB, label, matlIndex, patch,
			       replace);
}

void
OnDemandDataWarehouse::get(PerPatchBase& var, const VarLabel* label,
                           int matlIndex, const Patch* patch)
{
  //checkGetAccess(label);
  d_lock.readLock();
  if(!d_perpatchDB.exists(label, matlIndex, patch))
     throw UnknownVariable(label->getName(), patch, matlIndex,
			   "perpatch data");
  d_perpatchDB.get(label, matlIndex, patch, var);
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::put(PerPatchBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  //checkPutAccess(label, replace);
  
  d_lock.writeLock();

   // Error checking
   if(!replace && d_perpatchDB.exists(label, matlIndex, patch))
     throw InternalError("PerPatch variable already exists: "+label->getName());

   // Put it in the database
   d_perpatchDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void OnDemandDataWarehouse::
allocateTemporary(CCVariableBase& var, const Patch* patch,
		  Ghost::GhostType gtype, int numGhostCells)
{
  allocateTemporaryGridVar<Patch::CellBased>(var, patch, gtype, numGhostCells);
}

void OnDemandDataWarehouse::
allocateAndPut(CCVariableBase& var, const VarLabel* label,
	       int matlIndex, const Patch* patch,
	       Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar<Patch::CellBased>(var, d_ccDB, label, matlIndex, 
					  patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::get(constCCVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  CCVariableBase* var = constVar.cloneType();
  
 d_lock.readLock();  
  checkGetAccess(label, matlIndex, patch, gtype, numGhostCells);
  getGridVar<Patch::CellBased>(*var, d_ccDB, label, matlIndex, patch,
			       gtype, numGhostCells);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getModifiable(CCVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::CellBased>(var, d_ccDB, label, matlIndex, patch,
			       Ghost::None, 0);
 d_lock.readUnlock();  
}

void
OnDemandDataWarehouse::put(CCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  putGridVar<Patch::CellBased>(*var.clone(), d_ccDB, label, matlIndex, patch,
			       replace);  
}

void
OnDemandDataWarehouse::get(constSFCXVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  SFCXVariableBase* var = constVar.cloneType();

 d_lock.readLock();  
  checkGetAccess(label, matlIndex, patch, gtype, numGhostCells);
  getGridVar<Patch::XFaceBased>(*var, d_sfcxDB, label, matlIndex, patch,
				gtype, numGhostCells);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getModifiable(SFCXVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::XFaceBased>(var, d_sfcxDB, label, matlIndex, patch,
				Ghost::None, 0);
 d_lock.readUnlock();  
}

void OnDemandDataWarehouse::
allocateTemporary(SFCXVariableBase& var, const Patch* patch,
		  Ghost::GhostType gtype, int numGhostCells)
{
  allocateTemporaryGridVar<Patch::XFaceBased>(var, patch,
					      gtype, numGhostCells);
}

void OnDemandDataWarehouse::
allocateAndPut(SFCXVariableBase& var, const VarLabel* label,
	       int matlIndex, const Patch* patch,
	       Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar<Patch::XFaceBased>(var, d_sfcxDB, label, matlIndex, 
					   patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCXVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  putGridVar<Patch::XFaceBased>(*var.clone(), d_sfcxDB, label, matlIndex,
				patch, replace);
}

void
OnDemandDataWarehouse::get(constSFCYVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  SFCYVariableBase* var = constVar.cloneType();  
  
 d_lock.readLock();  
  checkGetAccess(label, matlIndex, patch, gtype, numGhostCells);
  getGridVar<Patch::YFaceBased>(*var, d_sfcyDB, label, matlIndex, patch,
				gtype, numGhostCells);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getModifiable(SFCYVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::YFaceBased>(var, d_sfcyDB, label, matlIndex, patch,
				Ghost::None, 0);
 d_lock.readUnlock();  
}

void OnDemandDataWarehouse::
allocateTemporary(SFCYVariableBase& var, const Patch* patch,
		  Ghost::GhostType gtype, int numGhostCells)
{
  allocateTemporaryGridVar<Patch::YFaceBased>(var, patch,
					      gtype, numGhostCells);
}

void OnDemandDataWarehouse::
allocateAndPut(SFCYVariableBase& var, const VarLabel* label,
	       int matlIndex, const Patch* patch,
	       Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar<Patch::YFaceBased>(var, d_sfcyDB, label, matlIndex, 
					   patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCYVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  putGridVar<Patch::YFaceBased>(*var.clone(), d_sfcyDB, label, matlIndex,
				patch, replace);  
}

void
OnDemandDataWarehouse::get(constSFCZVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  SFCZVariableBase* var = constVar.cloneType();

 d_lock.readLock();  
  checkGetAccess(label, matlIndex, patch, gtype, numGhostCells);
  getGridVar<Patch::ZFaceBased>(*var, d_sfczDB, label, matlIndex, patch,
				gtype, numGhostCells);
 d_lock.readUnlock();
  
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getModifiable(SFCZVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::ZFaceBased>(var, d_sfczDB, label, matlIndex, patch,
				Ghost::None, 0);
 d_lock.readUnlock();  
}

void OnDemandDataWarehouse::
allocateTemporary(SFCZVariableBase& var, const Patch* patch,
		  Ghost::GhostType gtype, int numGhostCells)
{
  allocateTemporaryGridVar<Patch::ZFaceBased>(var, patch,
					      gtype, numGhostCells);
}

void OnDemandDataWarehouse::
allocateAndPut(SFCZVariableBase& var, const VarLabel* label,
	       int matlIndex, const Patch* patch,
	       Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar<Patch::ZFaceBased>(var, d_sfczDB, label, matlIndex, 
					   patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCZVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  putGridVar<Patch::ZFaceBased>(*var.clone(), d_sfczDB, label, matlIndex,
				patch, replace);
}

bool
OnDemandDataWarehouse::exists(const VarLabel* label, const Patch* patch) const
{
  d_lock.readLock();
   if(!patch){
     if( d_reductionDB.exists(label, NULL) ) {
	d_lock.readUnlock();
	return true;       
     }
   } else {
      if( d_ncDB.exists(label, patch) || 
	  d_ccDB.exists(label, patch) ||
	  d_sfcxDB.exists(label, patch) ||
	  d_sfcyDB.exists(label, patch) ||
	  d_sfczDB.exists(label, patch) ||
	  d_particleDB.exists(label, patch) ){
  d_lock.readUnlock();
	return true;
      }
   }
  d_lock.readUnlock();
   return false;
}


void OnDemandDataWarehouse::emit(OutputContext& oc, const VarLabel* label,
				 int matlIndex, const Patch* patch)
{
  d_lock.readLock();
   checkGetAccess(label, matlIndex, patch);

   Variable* var = NULL;
   if(d_ncDB.exists(label, matlIndex, patch))
      var = d_ncDB.get(label, matlIndex, patch);
   else if(d_particleDB.exists(label, matlIndex, patch))
      var = d_particleDB.get(label, matlIndex, patch);
   else if(d_ccDB.exists(label, matlIndex, patch))
      var = d_ccDB.get(label, matlIndex, patch);
   else if(d_sfcxDB.exists(label, matlIndex, patch))
      var = d_sfcxDB.get(label, matlIndex, patch);
   else if(d_sfcyDB.exists(label, matlIndex, patch))
      var = d_sfcyDB.get(label, matlIndex, patch);
   else if(d_sfczDB.exists(label, matlIndex, patch))
      var = d_sfczDB.get(label, matlIndex, patch);
   else if(d_reductionDB.exists(label, matlIndex, patch))
      var = d_reductionDB.get(label, matlIndex, patch);

   if (var == NULL)
      throw UnknownVariable(label->getName(), patch, matlIndex, "on emit");

   var->emit(oc, label->getCompressionMode());
   
  d_lock.readUnlock();
}

void OnDemandDataWarehouse::print(ostream& intout, const VarLabel* label,
				  int matlIndex /* = -1 */)
{
  d_lock.readLock();

   try {
     checkGetAccess(label, matlIndex, 0); 
     ReductionVariableBase* var = d_reductionDB.get(label, matlIndex, NULL);
     var->print(intout);
   }
   catch (UnknownVariable) {
     throw UnknownVariable(label->getName(), NULL, matlIndex,
			    "on emit reduction");
   }

  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::deleteParticles(ParticleSubset* delset)
{
 d_lock.writeLock();
  int matlIndex = delset->getMatlIndex();
  Patch* patch = (Patch*) delset->getPatch();
  psetDBType::key_type key(matlIndex, patch);
  if(d_delsetDB.find(key) != d_delsetDB.end())
    throw InternalError("deleteParticles called twice for patch");

  d_delsetDB[key]=delset;
  delset->addReference();
 d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::scrub(const VarLabel* var)
{
  d_lock.writeLock();

  switch(var->typeDescription()->getType()){
  case TypeDescription::NCVariable:
    d_ncDB.scrub(var);
    break;
  case TypeDescription::CCVariable:
    d_ccDB.scrub(var);
    break;
  case TypeDescription::SFCXVariable:
    d_sfcxDB.scrub(var);
    break;
  case TypeDescription::SFCYVariable:
    d_sfcyDB.scrub(var);
    break;
  case TypeDescription::SFCZVariable:
    d_sfczDB.scrub(var);
    break;
  case TypeDescription::ParticleVariable:
    d_particleDB.scrub(var);
    break;
  case TypeDescription::PerPatch:
    d_perpatchDB.scrub(var);
    break;
  case TypeDescription::ReductionVariable:
    d_reductionDB.scrub(var);
    break;
  default:
    throw InternalError("Scrubbing variable of unknown type: "+var->getName());
  }
  d_lock.writeUnlock();
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#endif  

template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
void OnDemandDataWarehouse::
getGridVar(VariableBase& var, DWDatabase& db,
	   const VarLabel* label, int matlIndex, const Patch* patch,
	   Ghost::GhostType gtype, int numGhostCells)
{
  ASSERTEQ(basis, Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));  

  if(!db.exists(label, matlIndex, patch))
    throw UnknownVariable(label->getName(), patch, matlIndex);
  db.get(label, matlIndex, patch, var);
  
  IntVector low = patch->getLowIndex(basis);
  IntVector high = patch->getHighIndex(basis);

  // The data should have been put in the database,
  // windowed with this low and high.
  ASSERTEQ(var.getLow(), low);
  ASSERTEQ(var.getHigh(), high);
  
  if (gtype == Ghost::None) {
    if(numGhostCells != 0)
      throw InternalError("Ghost cells specified with task type none!\n");
    // if this assertion fails, then it is having problems getting the
    // correct window of the data.
    bool no_realloc = var.rewindow(low, high);
    ASSERT(no_realloc);
  }
  else {
    IntVector dn = high - low;
    long total = dn.x()*dn.y()*dn.z();
    
    Level::selectType neighbors;
    IntVector lowIndex, highIndex;
    patch->computeVariableExtents(basis, gtype, numGhostCells,
				  neighbors, lowIndex, highIndex);
    if (!var.rewindow(lowIndex, highIndex)) {
      // reallocation needed
      // Ignore this if this is the initialization dw in its old state.
      // The reason for this is that during initialization it doesn't
      // know what ghost cells will be required of it for the next timestep.
      // (This will be an issue whenever the taskgraph changes to require
      // more ghost cells from the old datawarehouse).
      bool ignore = d_isInitializationDW && d_finalized;
      if (!ignore) {
	ostringstream errmsg;
	errmsg << "Reallocation Error: Reallocation needed for " << label->getName();
	if (patch)
	  errmsg << " on patch " << patch->getID();
	errmsg << " for material " << matlIndex;
	throw InternalError(errmsg.str().c_str());
      }
    }
    
    for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      if(neighbor && (neighbor != patch)){
	if(!db.exists(label, matlIndex, neighbor))
	  throw UnknownVariable(label->getName(), neighbor, matlIndex,
				neighbor == patch?"on patch":"on neighbor");
	VariableBase* srcvar = var.cloneType();
	db.get(label, matlIndex, neighbor, *srcvar);
	
	low = Max(lowIndex, neighbor->getLowIndex(basis));
	high= Min(highIndex, neighbor->getHighIndex(basis));
	
	if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	    || ( high.z() < low.z() ) )
	  throw InternalError("Patch doesn't overlap?");
	
	var.copyPatch(srcvar, low, high);
	
	dn = high-low;
	total+=dn.x()*dn.y()*dn.z();
	delete srcvar;
      }
    }
    
    dn = highIndex - lowIndex;
    long wanted = dn.x()*dn.y()*dn.z();
    ASSERTEQ(wanted, total);
    if(wanted!=total){
      // This ASSERT or this warning are invoked even when trying
      // to do pefectly legitimate things.
      cerr << "Warning:  wanted cells/nodes != total cells/nodes " << endl;
    }
  }
}

template <Patch::VariableBasis basis, class VariableBase>
void OnDemandDataWarehouse::
allocateTemporaryGridVar(VariableBase& var, 
			 const Patch* patch,
			 Ghost::GhostType gtype, int numGhostCells)
{
  IntVector lowIndex = patch->getLowIndex(basis);
  IntVector highIndex = patch->getHighIndex(basis);
  if (numGhostCells > 0) {
    IntVector lowOffset, highOffset;
    Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
			   numGhostCells, lowOffset, highOffset);
    patch->computeExtents(basis, lowOffset, highOffset, lowIndex, highIndex);
  }

#ifdef WAYNE_DEBUG
  IntVector diff = highIndex - lowIndex;
  int allocSize = diff.x() * diff.y() * diff.z();
  totalGridAlloc += allocSize;
  cerr << "Allocate temporary: " << lowIndex << " - " << highIndex << " = " << allocSize << endl;
#endif
  var.allocate(lowIndex, highIndex);
}

template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
void OnDemandDataWarehouse::
allocateAndPutGridVar(VariableBase& var, DWDatabase& db,
		      const VarLabel* label, int matlIndex, const Patch* patch,
		      Ghost::GhostType gtype, int numGhostCells)
{
  ASSERT(!d_finalized);

  // Note: almost the entire function is write locked in order to prevent dual
  // allocations in a multi-threaded environment.  Whichever patch in a
  // super patch group gets here first, does the allocating for the entire
  // super patch group.
 d_lock.writeLock();

  if (!hasRunningTask()) {
    throw InternalError("OnDemandDataWarehouse::AllocateAndPutGridVar can only be used when the dw has a running task associated with it.");
  }
 
  checkPutAccess(label, matlIndex, patch, false);  
  bool exists = db.exists(label, matlIndex, patch);

  IntVector lowIndex = patch->getLowIndex(basis);
  IntVector highIndex = patch->getHighIndex(basis);
  if (numGhostCells > 0) {
    // expand lowIndex, highIndex to include ghost cells
    IntVector lowOffset, highOffset;
    Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
			   numGhostCells, lowOffset, highOffset);
    patch->computeExtents(basis, lowOffset, highOffset, lowIndex, highIndex);
  }
  
 if (exists) {
    // it had been allocated and put as part of the superpatch of
    // another patch
    db.get(label, matlIndex, patch, var);
    
    // The var's window should be the size of the patch or smaller than it.
    ASSERTEQ(Min(var.getLow(), lowIndex), lowIndex);
    ASSERTEQ(Max(var.getHigh(), highIndex), highIndex);
    
    if (var.getLow() != patch->getLowIndex(basis) ||
	var.getHigh() != patch->getHighIndex(basis) ||
	var.getBasePointer() == 0 /* place holder for ghost patch */) {
      // It wasn't allocated as part of another patch's superpatch;
      // it existed as ghost patch of another patch.. so we have no
      // choice but to blow it away and replace it.
      db.put(label, matlIndex, patch, 0, true);

      // this is just a tricky way to uninitialize var
      VariableBase* tmpVar = var.cloneType();
      var.copyPointer(*tmpVar);
    }
    else {
      // It was allocated and put as part of the superpatch of another patch
      var.rewindow(lowIndex, highIndex);
     d_lock.writeUnlock();      
      return; // got it -- done
    }
  }

  IntVector superLowIndex, superHighIndex;
  // requiredSuper[Low/High]'s don't take numGhostCells into consideration
  // -- just includes ghosts that will be required by later tasks.
  IntVector requiredSuperLow, requiredSuperHigh;  

  const vector<const Patch*>* superPatchGroup =
    d_scheduler->getSuperPatchExtents(label, matlIndex, patch,
				      gtype, numGhostCells,
				      requiredSuperLow, requiredSuperHigh,
				      superLowIndex, superHighIndex);
 
  ASSERT(superPatchGroup != 0);

#ifdef WAYNE_DEBUG
  IntVector diff = superHighIndex - superLowIndex;
  int allocSize = diff.x() * diff.y() * diff.z();
  totalGridAlloc += allocSize;
  cerr << "Allocate " << label->getName() << ", matl " << matlIndex << ": " << superLowIndex << " - " << superHighIndex << " = " << allocSize << endl;  
#endif
  
  var.allocate(superLowIndex, superHighIndex);
  
  Level::selectType encompassedPatches;
  if (requiredSuperLow == lowIndex && requiredSuperHigh == highIndex) {
    // only encompassing the patch currently being allocated
    encompassedPatches.push_back(patch);
  }
  else {
    // Use requiredSuperLow/High instead of superLowIndex/superHighIndex
    // so we don't put the var for patches in the datawarehouse that won't be
    // required (this is important for scrubbing).
    patch->getLevel()->selectPatches(requiredSuperLow, requiredSuperHigh,
				     encompassedPatches);
  }
  
  // Make a set of the non ghost patches that
  // has quicker lookup than the vector.
  set<const Patch*> nonGhostPatches;
  for (unsigned int i = 0; i < superPatchGroup->size(); ++i) {
    nonGhostPatches.insert((*superPatchGroup)[i]);
  }
  
  Level::selectType::iterator iter = encompassedPatches.begin();    
  for (; iter != encompassedPatches.end(); ++iter) {
    const Patch* patchGroupMember = *iter;
    VariableBase* clone = var.clone();
    IntVector groupMemberLowIndex = patchGroupMember->getLowIndex(basis);
    IntVector groupMemberHighIndex = patchGroupMember->getHighIndex(basis);
    IntVector enclosedLowIndex = Max(groupMemberLowIndex, superLowIndex);
    IntVector enclosedHighIndex = Min(groupMemberHighIndex, superHighIndex);
    
    clone->rewindow(enclosedLowIndex, enclosedHighIndex);
    if (patchGroupMember == patch) {
      // this was checked already
      exists = false;
    }
    else {
      exists = db.exists(label, matlIndex, patchGroupMember);
    }
    if (patchGroupMember->isVirtual()) {
      // Virtual patches can only be ghost patches.
      ASSERT(nonGhostPatches.find(patchGroupMember) ==
	     nonGhostPatches.end());
      clone->offsetGrid(IntVector(0,0,0) -
			patchGroupMember->getVirtualOffset());
      enclosedLowIndex = clone->getLow();
      enclosedHighIndex = clone->getHigh();
      patchGroupMember = patchGroupMember->getRealPatch();
      IntVector dummy;
      if (d_scheduler->
	  getSuperPatchExtents(label, matlIndex, patchGroupMember, gtype,
			       numGhostCells, dummy, dummy, dummy, dummy) != 0)
      {
	// The virtual patch refers to a real patch in which the label
	// is computed locally, so don't overwrite the local copy.
	delete clone;
	continue;
      }
    }
    if (exists) {
      // variable section already exists in this patchGroupMember
      // (which is assumed to be a ghost patch)
      // so check if one is enclosed in the other.
      
      // Assumption is that it is a ghost patch -- so assert that.
      ASSERT(nonGhostPatches.find(patchGroupMember)
	     == nonGhostPatches.end());

      VariableBase* existingGhostVar =
	db.get(label, matlIndex, patchGroupMember);
      IntVector existingLow = existingGhostVar->getLow();
      IntVector existingHigh = existingGhostVar->getHigh();
      IntVector minLow = Min(existingLow, enclosedLowIndex);
      IntVector maxHigh = Max(existingHigh, enclosedHighIndex);

      if (existingGhostVar->isForeign()) {
	// data already being received, so don't replace it
	delete clone;
      }
      else if (minLow == enclosedLowIndex && maxHigh == enclosedHighIndex) {
	// this new ghost variable section encloses the old one,
	// so replace the old one
	db.put(label, matlIndex, patchGroupMember, clone, true);
      }
      else {
	// Either the old ghost variable section encloses this new one
	// (so leave it), or neither encloses the other (so just forget
	// about it -- it'll allocate extra space for it when receiving
	// the ghost data in recvMPIGridVar if nothing else).
	delete clone;
      }
    }
    else {
      // it didn't exist before -- add it
      db.put(label, matlIndex, patchGroupMember, clone, false);
    }
  }
 d_lock.writeUnlock();
   
  var.rewindow(lowIndex, highIndex);
}

template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
void OnDemandDataWarehouse::
putGridVar(VariableBase& var, DWDatabase& db,
	   const VarLabel* label, int matlIndex, const Patch* patch,
	   bool replace /* = false */)
{
  ASSERT(!d_finalized);
  ASSERTEQ(basis, Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));    
 d_lock.writeLock();  

  checkPutAccess(label, matlIndex, patch, replace);

#if DAV_DEBUG
  cerr << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";
#endif
   // Error checking
   if(!replace && db.exists(label, matlIndex, patch))
      throw InternalError("put: grid variable already exists: " +
			  label->getName());

   // Put it in the database
   IntVector low = patch->getLowIndex(basis);
   IntVector high = patch->getHighIndex(basis);
   if (Min(var.getLow(), low) != var.getLow() ||
       Max(var.getHigh(), high) != var.getHigh()) {
     ostringstream msg_str;
     msg_str << "put: Variable's window (" << var.getLow() << " - " << var.getHigh() << ") must encompass patches extent (" << low << " - " << high;
     throw InternalError(msg_str.str());
   }
   bool no_realloc = var.rewindow(low, high);
   // error would have been thrown above if the any reallocation would be
   // needed
   ASSERT(no_realloc);
   db.put(label, matlIndex, patch, &var, true);
  d_lock.writeUnlock();
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424 // template parameter not used in declaring arguments
#endif  
  
void OnDemandDataWarehouse::logMemoryUse(ostream& out, unsigned long& total,
					 const std::string& tag)
{
  int dwid=d_generation;
  d_ncDB.logMemoryUse(out, total, tag, dwid);
  d_ccDB.logMemoryUse(out, total, tag, dwid);
  d_sfcxDB.logMemoryUse(out, total, tag, dwid);
  d_sfcyDB.logMemoryUse(out, total, tag, dwid);
  d_sfczDB.logMemoryUse(out, total, tag, dwid);
  d_particleDB.logMemoryUse(out, total, tag, dwid);
  d_reductionDB.logMemoryUse(out, total, tag, dwid);
  d_perpatchDB.logMemoryUse(out, total, tag, dwid);

  // Log the psets.
  for(psetDBType::iterator iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++){
    ParticleSubset* pset = iter->second;
    ostringstream elems;
    elems << pset->numParticles();
    logMemory(out, total, tag, "particles", "ParticleSubset", pset->getPatch(),
	      pset->getMatlIndex(), elems.str(),
	      pset->numParticles()*sizeof(particleIndex),
	      pset->getPointer(), dwid);
  }
}

inline void
OnDemandDataWarehouse::checkGetAccess(const VarLabel* label,
				      int matlIndex, const Patch* patch,
				      Ghost::GhostType gtype,int numGhostCells)
{
#if SCI_ASSERTION_LEVEL >= 1
  list<RunningTaskInfo>* runningTasks = getRunningTasksInfo();
  if (runningTasks != 0) {
    for (list<RunningTaskInfo>::iterator iter = runningTasks->begin();
	 iter != runningTasks->end(); iter++) {
      RunningTaskInfo& runningTaskInfo = *iter;
      const Task* runningTask = runningTaskInfo.d_task;
      if (runningTask == 0) {
	// don't check if done outside of any task (i.e. SimulationController)
	return;
      }
      
      IntVector lowOffset, highOffset;
      Patch::getGhostOffsets(label->typeDescription()->getType(), gtype,
			     numGhostCells, lowOffset, highOffset);
      
      VarAccessMap& runningTaskAccesses = runningTaskInfo.d_accesses;
      
      // If it was accessed by the current task already, then it should
      // have get access (i.e. if you put it in, you should be able to get it
      // right back out).
      map<SpecificVarLabel, AccessInfo>::iterator findIter;
      findIter = runningTaskAccesses.find(SpecificVarLabel(label, matlIndex,
							   patch));
      if (findIter != runningTaskAccesses.end() &&
	  lowOffset == IntVector(0, 0, 0) && highOffset == IntVector(0, 0, 0)){
	// allow non ghost cell get if any access (get, put, or modify) is allowed
	return;
      }
      
      if (!hasGetAccess(runningTask, label, matlIndex, patch, lowOffset,
			highOffset)) {
	if (runningTask == 0 ||
	    (string(runningTask->getName()) != "Relocate::relocateParticles")){
	  string has = (isFinalized() ? "old" : "new");
	  has += " datawarehouse get";
	  if (numGhostCells > 0) {
	    ostringstream ghost_str;
	    ghost_str << " for " << numGhostCells << " layer";
	    if (numGhostCells > 1) ghost_str << "s";
	    ghost_str << " of ghosts around "
		      << Ghost::getGhostTypeName(gtype);
	    has += ghost_str.str();
	  }	
	  string needs = "task requires";
	  throw DependencyException(runningTask, label, matlIndex, patch,
				    has, needs);
	}
      }
      else {
	// access granted
	if (findIter == runningTaskAccesses.end()) {
	  AccessInfo& accessInfo =
	    runningTaskAccesses[SpecificVarLabel(label, matlIndex, patch)];
	  accessInfo.accessType = GetAccess;
	  accessInfo.encompassOffsets(lowOffset, highOffset);
	}
	else {
	  findIter->second.encompassOffsets(lowOffset, highOffset);
	}
      }
    }
  }
#endif
}

inline void
OnDemandDataWarehouse::checkPutAccess(const VarLabel* label, int matlIndex,
				      const Patch* patch, bool replace)
{ 
#if SCI_ASSERTION_LEVEL >= 1
  list<RunningTaskInfo>* runningTasks = getRunningTasksInfo();
  if (runningTasks != 0) {
    for (list<RunningTaskInfo>::iterator iter = runningTasks->begin();
	 iter != runningTasks->end(); iter++) {
      RunningTaskInfo& runningTaskInfo = *iter;
      const Task* runningTask = runningTaskInfo.d_task;
      
      if (runningTask == 0)
	return; // don't check if outside of any task (i.e. SimulationController)
      
      VarAccessMap& runningTaskAccesses = runningTaskInfo.d_accesses;
      
      if (!hasPutAccess(runningTask, label, matlIndex, patch, replace)) {
	if (string(runningTask->getName())
	    != "Relocate::relocateParticles") {
	  string has, needs;
	  if (replace) {
	    has = "datawarehouse modify";
	    needs = "task modifies";
	  }
	  else {
	    has = "datawarehouse put";
	  needs = "task computes";
	  }
	  throw DependencyException(runningTask, label, matlIndex,
				    patch, has, needs);
	}
      }
      else {
	runningTaskAccesses[SpecificVarLabel(label, matlIndex, patch)].accessType = replace ? ModifyAccess : PutAccess;
      }
    }
  }
#endif
}
  
inline void
OnDemandDataWarehouse::checkModifyAccess(const VarLabel* label, int matlIndex,
					 const Patch* patch)
{ checkPutAccess(label, matlIndex, patch, true); }


inline bool
OnDemandDataWarehouse::hasGetAccess(const Task* runningTask,
				    const VarLabel* label, int matlIndex,
				    const Patch* patch, IntVector lowOffset,
				    IntVector highOffset)
{ 
  return
    runningTask->hasRequires(label, matlIndex, patch, lowOffset, highOffset,
			     isFinalized() ? Task::OldDW : Task::NewDW);
}

inline
bool OnDemandDataWarehouse::hasPutAccess(const Task* runningTask,
					 const VarLabel* label, int matlIndex,
					 const Patch* patch, bool replace)
{
  if (replace)
    return runningTask->hasModifies(label, matlIndex, patch);
  else
    return runningTask->hasComputes(label, matlIndex, patch);
}

void OnDemandDataWarehouse::pushRunningTask(const Task* task)
{
  ASSERT(task);
 d_lock.writeLock();    
  d_runningTasks[Thread::self()].push_back(RunningTaskInfo(task));
 d_lock.writeUnlock();
}

void OnDemandDataWarehouse::popRunningTask()
{
 d_lock.writeLock();
  list<RunningTaskInfo>& runningTasks = d_runningTasks[Thread::self()];
  runningTasks.pop_back();
  if (runningTasks.size() == 0) {
    d_runningTasks.erase(Thread::self());
  }
 d_lock.writeUnlock();
}

inline list<OnDemandDataWarehouse::RunningTaskInfo>*
OnDemandDataWarehouse::getRunningTasksInfo()
{
  map<Thread*, list<RunningTaskInfo> >::iterator findIt =
    d_runningTasks.find(Thread::self());
  return (findIt != d_runningTasks.end()) ? &findIt->second : 0;
}

inline bool OnDemandDataWarehouse::hasRunningTask()
{
  list<OnDemandDataWarehouse::RunningTaskInfo>* runningTasks =
    getRunningTasksInfo();
  return runningTasks ? !runningTasks->empty() : false;
}

inline OnDemandDataWarehouse::RunningTaskInfo*
OnDemandDataWarehouse::getCurrentTaskInfo()
{
  list<RunningTaskInfo>* taskInfoList = getRunningTasksInfo();
  return (taskInfoList && !taskInfoList->empty()) ? &taskInfoList->back() : 0;
}

void OnDemandDataWarehouse::checkTasksAccesses(const PatchSubset* patches,
					       const MaterialSubset* matls)
{
#if SCI_ASSERTION_LEVEL >= 1

  d_lock.readLock();
  
  RunningTaskInfo* currentTaskInfo = getCurrentTaskInfo();
  ASSERT(currentTaskInfo != 0);
  const Task* currentTask = currentTaskInfo->d_task;
  ASSERT(currentTask != 0);  
  
  if (isFinalized()) {
    checkAccesses(currentTaskInfo, currentTask->getRequires(), GetAccess,
		  patches, matls);
  }
  else {
    checkAccesses(currentTaskInfo, currentTask->getRequires(), GetAccess,
		  patches, matls);
    checkAccesses(currentTaskInfo, currentTask->getComputes(), PutAccess,
		  patches, matls);
    checkAccesses(currentTaskInfo, currentTask->getModifies(), ModifyAccess,
		  patches, matls);
  }

  d_lock.readUnlock();
  
#endif
}

void
OnDemandDataWarehouse::checkAccesses(RunningTaskInfo* currentTaskInfo,
				     const Task::Dependency* dep,
				     AccessType accessType,
				     const PatchSubset* domainPatches,
				     const MaterialSubset* domainMatls)
{
  ASSERT(currentTaskInfo != 0);
  const Task* currentTask = currentTaskInfo->d_task;
  if (currentTask->isReductionTask())
    return; // no need to check reduction tasks.

  VarAccessMap& currentTaskAccesses = currentTaskInfo->d_accesses;
  
  Handle<PatchSubset> default_patches = scinew PatchSubset();
  Handle<MaterialSubset> default_matls = scinew MaterialSubset();
  default_patches->add(0);
  default_matls->add(-1);
  
  for (; dep != 0; dep = dep->next) {
    if ((isFinalized() && dep->dw == Task::NewDW) ||
	(!isFinalized() && dep->dw == Task::OldDW))
      continue;
    
    const VarLabel* label = dep->var;
    IntVector lowOffset, highOffset;
    Patch::getGhostOffsets(label->typeDescription()->getType(), dep->gtype,
			   dep->numGhostCells, lowOffset, highOffset);    

    constHandle<PatchSubset> patches =
      dep->getPatchesUnderDomain(domainPatches);
    constHandle<MaterialSubset> matls =
      dep->getMaterialsUnderDomain(domainMatls);
    if (label->typeDescription() &&
	label->typeDescription()->isReductionVariable()) {
      patches = default_patches.get_rep();
    }
    else if (patches == 0) {
      patches = default_patches.get_rep();
    }
    if (matls == 0) {
      matls = default_matls.get_rep();
    }
 
    if (string(currentTask->getName())
	== "Relocate::relocateParticles") {
      continue;
    }
    
    for (int m = 0; m < matls->size(); m++) {
      int matl = matls->get(m);
      
      for (int p = 0; p < patches->size(); p++) {
	const Patch* patch = patches->get(p);
	
	SpecificVarLabel key(label, matl, patch);
	map<SpecificVarLabel, AccessInfo>::iterator find_iter;
	find_iter = currentTaskAccesses.find(key);
	if (find_iter == currentTaskAccesses.end() ||
	    (*find_iter).second.accessType != accessType) {
	  if ((*find_iter).second.accessType == ModifyAccess && accessType == GetAccess) {
	    // If you require with ghost cells and modify, it can get into this situation.
	    continue;
	  }
	  // Makes request that is never followed through.
	  string has, needs;
	  if (accessType == GetAccess) {
	    has = "task requires";
	    if (isFinalized())
	      needs = "get from the old datawarehouse";
	    else
	      needs = "get from the new datawarehouse";
	  }
	  else if (accessType == PutAccess) {
	    has = "task computes";
	    needs = "datawarehouse put";
	  }
	  else {
	    has = "task modifies";
	    needs = "datawarehouse modify";
	  }

	  throw DependencyException(currentTask, label, matl, patch,
				    has, needs);	  
	}
	else if (((*find_iter).second.lowOffset != lowOffset ||
		  (*find_iter).second.highOffset != highOffset) &&
		 accessType != ModifyAccess /* Can == ModifyAccess when you require with
					       ghost cells and modify */ ) {
	  // Makes request for ghost cells that are never gotten.
	  AccessInfo accessInfo = (*find_iter).second;	
	  ASSERT(accessType == GetAccess);

	  // Assert that the request was greater than what was asked for
	  // because the other cases (where it asked for more than the request)
	  // should have been caught in checkGetAccess().
	  ASSERT(Max((*find_iter).second.lowOffset, lowOffset) == lowOffset);
	  ASSERT(Max((*find_iter).second.highOffset,highOffset) == highOffset);

	  string has, needs;
	  has = "task requires";
	  ostringstream ghost_str;
	  ghost_str << " requesting " << dep->numGhostCells << " layer";
	  if (dep->numGhostCells > 1) ghost_str << "s";
	  ghost_str << " of ghosts around " <<
	    Ghost::getGhostTypeName(dep->gtype);
	  has += ghost_str.str();
	  
	  if (isFinalized())
	    needs = "get from the old datawarehouse";
	  else
	    needs = "get from the new datawarehouse";
	  needs += " that includes these ghosts";

	  throw DependencyException(currentTask, label, matl, patch,
				    has, needs);
	}
      }
    }
  }
}


bool OnDemandDataWarehouse::SpecificVarLabel::
operator<(const SpecificVarLabel& other) const
{
  if (label_->equals(other.label_)) {
    if (matlIndex_ == other.matlIndex_)
      return patch_ < other.patch_;
    else
      return matlIndex_ < other.matlIndex_;
  }
  else {
    VarLabel::Compare comp;
    return comp(label_, other.label_);
  }
}
