
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>

#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DeniedAccess.h>
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

using std::cerr;
using std::string;
using std::vector;

using namespace SCIRun;

using namespace Uintah;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern Mutex cerrLock;
extern DebugStream mixedDebug;

Mutex getMPIBuffLock( "getMPIBuffLock" );
Mutex ssLock( "send state lock" );

#define PARTICLESET_TAG		0x1000000
#define DAV_DEBUG 0

OnDemandDataWarehouse::OnDemandDataWarehouse(const ProcessorGroup* myworld,
					     const Scheduler* scheduler,
					     int generation, const GridP& grid)
   : DataWarehouse(myworld, scheduler, generation),
     d_lock("DataWarehouse lock"),
     d_finalized( false ),
     d_grid(grid)
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

void
OnDemandDataWarehouse::get(ReductionVariableBase& var,
			   const VarLabel* label, int matlIndex /*= -1*/)
{
  checkGetAccess(label, matlIndex, 0);
  d_lock.readLock();

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
	getMPIBuffLock.lock(); // Dd: ??
	 var->getMPIBuffer(buffer, sendset);
	getMPIBuffLock.unlock(); // Dd: ??
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
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::CCVariable:
    {
      if(!d_ccDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      CCVariableBase* var = d_ccDB.get(label, matlIndex, patch);
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCXVariable:
    {
      if(!d_sfcxDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      SFCXVariableBase* var = d_sfcxDB.get(label, matlIndex, patch);
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      if(!d_sfcyDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      SFCYVariableBase* var = d_sfcyDB.get(label, matlIndex, patch);
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      if(!d_sfczDB.exists(label, matlIndex, patch))
	throw UnknownVariable(label->getName(), patch, matlIndex,
			      "in sendMPI");
      SFCZVariableBase* var = d_sfczDB.get(label, matlIndex, patch);
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
	getMPIBuffLock.unlock(); // Dd: ??
    }
    break;
  default:
    throw InternalError("sendMPI not implemented for "+label->getFullName(matlIndex, patch));
  } // end switch( label->getType() );
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

      getMPIBuffLock.lock(); // Dd: ??

      Variable* v = label->typeDescription()->createInstance();
      ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(pset);
      var->setForeign();
      if(pset->numParticles() > 0){
	var->getMPIBuffer(buffer, pset);
      }

      getMPIBuffLock.unlock(); // Dd: ??

      d_lock.writeLock();
      d_particleDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  case TypeDescription::NCVariable:
    {
      /* Can modify variable now
      if(d_ncDB.exists(label, matlIndex, patch))
	throw InternalError("Variable already exists before MPI recv: " +
			    label->getFullName(matlIndex, patch));
      */
      Variable* v = label->typeDescription()->createInstance();
      NCVariableBase* var = dynamic_cast<NCVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(dep->low, dep->high);
      var->setForeign();
      
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      d_lock.writeLock();
      d_ncDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  case TypeDescription::CCVariable:
    {
      /* Can modify variable now
      if(d_ccDB.exists(label, matlIndex, patch))
	throw InternalError("Variable already exists before MPI recv: "+label->getFullName(matlIndex, patch));
      */
      Variable* v = label->typeDescription()->createInstance();
      CCVariableBase* var = dynamic_cast<CCVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(dep->low, dep->high);
      var->setForeign();
      
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      d_lock.writeLock();
      d_ccDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  case TypeDescription::SFCXVariable:
    {
      /* Can modify variable now
      if(d_sfcxDB.exists(label, matlIndex, patch))
	throw InternalError("Variable already exists before MPI recv: "+label->getFullName(matlIndex, patch));
      */
      Variable* v = label->typeDescription()->createInstance();
      SFCXVariableBase* var = dynamic_cast<SFCXVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(dep->low, dep->high);
      var->setForeign();

	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      d_lock.writeLock();
      d_sfcxDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      /* Can modify variable now
      if(d_sfcyDB.exists(label, matlIndex, patch))
	throw InternalError("Variable already exists before MPI recv: "+label->getFullName(matlIndex, patch));
      */
      Variable* v = label->typeDescription()->createInstance();
      SFCYVariableBase* var = dynamic_cast<SFCYVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(dep->low, dep->high);
      var->setForeign();
      
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      d_lock.writeLock();
      d_sfcyDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      /* Can modify variable now
      if(d_sfczDB.exists(label, matlIndex, patch))
	throw InternalError("Variable already exists before MPI recv: "+label->getFullName(matlIndex, patch));
      */
      Variable* v = label->typeDescription()->createInstance();
      SFCZVariableBase* var = dynamic_cast<SFCZVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(dep->low, dep->high);
      var->setForeign();
      
	getMPIBuffLock.lock(); // Dd: ??
      var->getMPIBuffer(buffer, dep->low, dep->high);
	getMPIBuffLock.unlock(); // Dd: ??
      d_lock.writeLock();
      d_sfczDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  default:
    throw InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch));
  } // end switch( label->getType() );
} // end recvMPI()

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
	getMPIBuffLock.lock(); // Dd: ??
  var->getMPIBuffer(sendbuf, sendcount, senddatatype, sendop);
	getMPIBuffLock.unlock(); // Dd: ??
  ReductionVariableBase* tmp = var->clone();
  void* recvbuf;
  int recvcount;
  MPI_Datatype recvdatatype;
  MPI_Op recvop;
	getMPIBuffLock.lock(); // Dd: ??
  tmp->getMPIBuffer(recvbuf, recvcount, recvdatatype, recvop);
	getMPIBuffLock.unlock(); // Dd: ??
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
OnDemandDataWarehouse::allocate(ReductionVariableBase& var,
				const VarLabel* label, int)
{
   cerr << "OnDemand DataWarehouse::allocate(ReductionVariable) "
	<< "not finished\n";
   var.setAllocationLabel(label);
}

void
OnDemandDataWarehouse::put(const ReductionVariableBase& var,
			   const VarLabel* label, int matlIndex /* = -1 */)
{
  ASSERT(!d_finalized);  
  checkPutAccess(label, matlIndex, 0,
		 false /* it actually may be replaced, but it doesn't need
			  to explicitly modify with multiple reduces in the
			  task graph */);
  d_lock.writeLock();

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
  checkPutAccess(label, matlIndex, 0, true);
  
  d_lock.writeLock();

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
   psetDBType::key_type key(matlIndex, patch);
   psetDBType::iterator iter = d_psetDB.find(key);
   if(iter == d_psetDB.end()){
  d_lock.readUnlock();
      throw UnknownVariable("ParticleSet", patch, matlIndex,
			    "Cannot find particle set on patch");
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
	 ParticleSubset* pset = getParticleSubset(matlIndex, neighbor);
	 constParticleVariable<Point> pos;
	 get(pos, pos_var, pset);

	 ParticleSubset* subset = 
	    scinew ParticleSubset(pset->getParticleSet(), false, -1, 0);
	 for(ParticleSubset::iterator iter = pset->begin();
	     iter != pset->end(); iter++){
	    particleIndex idx = *iter;
	    if(box.contains(pos[idx]))
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
			   const VarLabel* label,
			   ParticleSubset* pset)
{
  ParticleVariableBase& var = *constVar.cloneType();
  getModifiable(var, label, pset);
  constVar = var;
}

void
OnDemandDataWarehouse::getModifiable(ParticleVariableBase& var,
				     const VarLabel* label,
				     ParticleSubset* pset)
{
  d_lock.readLock();
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();
   checkGetAccess(label, matlIndex, patch);
   
   if(pset->getGhostType() == Ghost::None){
      if(!d_particleDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      d_particleDB.get(label, matlIndex, patch, var);
   } else {
      const vector<const Patch*>& neighbors = pset->getNeighbors();
      const vector<ParticleSubset*>& neighbor_subsets = pset->getNeighborSubsets();
      vector<ParticleVariableBase*> neighborvars(neighbors.size());
      for(int i=0;i<(int)neighbors.size();i++){
	 const Patch* neighbor=neighbors[i];
	 if(!d_particleDB.exists(label, matlIndex, neighbors[i]))
	    throw UnknownVariable(label->getName(), neighbor, matlIndex,
				  neighbor == patch?"on patch":"on neighbor");
	 neighborvars[i] = d_particleDB.get(label, matlIndex, neighbors[i]);
      }
      var.gather(pset, neighbor_subsets, neighborvars);
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
      if(!d_particleDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      return d_particleDB.get(label, matlIndex, patch);
   } else {
      throw InternalError("getParticleVariable should not be used with ghost cells");
   }
}

void
OnDemandDataWarehouse::allocate(ParticleVariableBase& var,
				const VarLabel* label,
				ParticleSubset* pset)
{
  d_lock.writeLock();
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   // Error checking
   if(d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Particle variable already exists: " +
			  label->getName());

   var.allocate(pset);
   var.setAllocationLabel(label);
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(ParticleVariableBase& var,
			   const VarLabel* label, bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  checkAllocation(var, label);

   ParticleSubset* pset = var.getParticleSubset();
   if(pset->numGhostCells() != 0 || pset->getGhostType() != 0)
      throw InternalError("ParticleVariable cannot use put with ghost cells");
   const Patch* patch = pset->getPatch();
   int matlIndex = pset->getMatlIndex();

   checkPutAccess(label, matlIndex, patch, replace);
   
   // Error checking
   if(!replace && d_particleDB.exists(label, matlIndex, patch))
      throw InternalError("Variable already exists: "+label->getName());

#if DAV_DEBUG
  cerr << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";
#endif
   // Put it in the database
  d_lock.writeLock();
   d_particleDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::get(constNCVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  checkGetAccess(label, matlIndex, patch);
  NCVariableBase& var = *constVar.cloneType();
  getGridVar<Patch::NodeBased>(var, d_ncDB, label, matlIndex, patch,
			       gtype, numGhostCells);
  constVar = var;
}

void
OnDemandDataWarehouse::getModifiable(NCVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
  checkModifyAccess(label, matlIndex, patch);
  getGridVar<Patch::NodeBased>(var, d_ncDB, label, matlIndex, patch,
			       Ghost::None, 0);
}

void
OnDemandDataWarehouse::allocate(NCVariableBase& var, const VarLabel* label,
				int matlIndex, const Patch* patch,
				Ghost::GhostType gtype, int numGhostCells)
{
#if DAV_DEBUG
  cerr << "alloc: NC var: " << *label << *patch 
       << " MI: " << matlIndex << "\n";
#endif
  
  allocateGridVar<Patch::NodeBased>(var, d_ncDB, label, matlIndex, patch,
				    gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(NCVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  checkPutAccess(label, matlIndex, patch, replace);
  checkAllocation(var, label);
  
  d_lock.writeLock();

#if DAV_DEBUG
  cerr << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";
#endif
   // Error checking
   if(!replace && d_ncDB.exists(label, matlIndex, patch))
      throw InternalError("put: NC variable already exists: " +
			  label->getName());

   // Put it in the database
   d_ncDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
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

void
OnDemandDataWarehouse::allocate(CCVariableBase& var, const VarLabel* label,
				int matlIndex, const Patch* patch,
				Ghost::GhostType gtype, int numGhostCells)
{
  allocateGridVar<Patch::CellBased>(var, d_ccDB, label, matlIndex, patch,
				    gtype, numGhostCells);
}

void
OnDemandDataWarehouse::get(constCCVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  checkGetAccess(label, matlIndex, patch);
  CCVariableBase& var = *constVar.cloneType();  
  getGridVar<Patch::CellBased>(var, d_ccDB, label, matlIndex, patch,
			       gtype, numGhostCells);
  constVar = var;
}

void
OnDemandDataWarehouse::getModifiable(CCVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::CellBased>(var, d_ccDB, label, matlIndex, patch,
			       Ghost::None, 0);
}

void
OnDemandDataWarehouse::put(CCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  checkPutAccess(label, matlIndex, patch, replace);
  checkAllocation(var, label);
  
  d_lock.writeLock();
   // Error checking
   if(!replace && d_ccDB.exists(label, matlIndex, patch))
      throw InternalError("CC variable already exists: "+label->getName());

   IntVector low, high, size;
   var.getSizes(low, high, size);

   if(low != patch->getCellLowIndex() || high != patch->getCellHighIndex()){
#if 0
      cerr << "Warning, rewindowing array: " << label->getName() << " on patch " << patch->getID() << '\n';
#endif
      CCVariableBase* newvar = var.clone();
      newvar->rewindow(patch->getCellLowIndex(), patch->getCellHighIndex());
      d_ccDB.put(label, matlIndex, patch, newvar, true);
   } else {
      // Put it in the database
      d_ccDB.put(label, matlIndex, patch, var.clone(), true);
   }
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::get(constSFCXVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  checkGetAccess(label, matlIndex, patch);
  SFCXVariableBase& var = *constVar.cloneType();
  getGridVar<Patch::XFaceBased>(var, d_sfcxDB, label, matlIndex, patch,
				gtype, numGhostCells);
  constVar = var;
}

void
OnDemandDataWarehouse::getModifiable(SFCXVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::XFaceBased>(var, d_sfcxDB, label, matlIndex, patch,
				Ghost::None, 0);
}

void
OnDemandDataWarehouse::allocate(SFCXVariableBase& var, const VarLabel* label,
				int matlIndex, const Patch* patch,
				Ghost::GhostType gtype, int numGhostCells)
{
  allocateGridVar<Patch::XFaceBased>(var, d_sfcxDB, label, matlIndex, patch,
				     gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCXVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  checkPutAccess(label, matlIndex, patch, replace);
  checkAllocation(var, label);
  
  d_lock.writeLock();

   // Error checking
   if(!replace && d_sfcxDB.exists(label, matlIndex, patch))
      throw InternalError("SFCX variable already exists: "+label->getName());

   IntVector low, high, size;
   var.getSizes(low, high, size);
   if(low != patch->getSFCXLowIndex() || high != patch->getSFCXHighIndex()) {
#if 0
      cerr << "Warning, rewindowing array: " << label->getName() << " on patch " << patch->getID() << '\n';
#endif
      SFCXVariableBase* newvar = var.clone();
      newvar->rewindow(patch->getSFCXLowIndex(), patch->getSFCXHighIndex());
      d_sfcxDB.put(label, matlIndex, patch, newvar, true);
   } else {

      // Put it in the database
      d_sfcxDB.put(label, matlIndex, patch, var.clone(), true);
   }
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::get(constSFCYVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  checkGetAccess(label, matlIndex, patch);
  SFCYVariableBase& var = *constVar.cloneType();
  getGridVar<Patch::YFaceBased>(var, d_sfcyDB, label, matlIndex, patch,
				gtype, numGhostCells);
  constVar = var;
}

void
OnDemandDataWarehouse::getModifiable(SFCYVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::YFaceBased>(var, d_sfcyDB, label, matlIndex, patch,
				Ghost::None, 0);
}

void
OnDemandDataWarehouse::allocate(SFCYVariableBase& var, const VarLabel* label,
				int matlIndex, const Patch* patch,
				Ghost::GhostType gtype, int numGhostCells)
{
  allocateGridVar<Patch::YFaceBased>(var, d_sfcyDB, label, matlIndex, patch,
				     gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCYVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  checkPutAccess(label, matlIndex, patch, replace);
  checkAllocation(var, label);
  
  d_lock.writeLock();

   // Error checking
   if(!replace && d_sfcyDB.exists(label, matlIndex, patch))
      throw InternalError("SFCY variable already exists: "+label->getName());

   IntVector low, high, size;
   var.getSizes(low, high, size);
   if(low != patch->getSFCYLowIndex() || high != patch->getSFCYHighIndex()) {
#if 0
      cerr << "Warning, rewindowing array: " << label->getName() << " on patch " << patch->getID() << '\n';
#endif
      SFCYVariableBase* newvar = var.clone();
      newvar->rewindow(patch->getSFCYLowIndex(), patch->getSFCYHighIndex());
      d_sfcyDB.put(label, matlIndex, patch, newvar, true);
   } else {

      // Put it in the database
      d_sfcyDB.put(label, matlIndex, patch, var.clone(), true);
   }
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::get(constSFCZVariableBase& constVar,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype, int numGhostCells)
{
  checkGetAccess(label, matlIndex, patch);
  SFCZVariableBase& var = *constVar.cloneType();
  getGridVar<Patch::ZFaceBased>(var, d_sfczDB, label, matlIndex, patch,
				gtype, numGhostCells);
  constVar = var;
}

void
OnDemandDataWarehouse::getModifiable(SFCZVariableBase& var,
				     const VarLabel* label,
				     int matlIndex, const Patch* patch)
{
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar<Patch::ZFaceBased>(var, d_sfczDB, label, matlIndex, patch,
				Ghost::None, 0);
}

void
OnDemandDataWarehouse::allocate(SFCZVariableBase& var, const VarLabel* label,
				int matlIndex, const Patch* patch,
				Ghost::GhostType gtype, int numGhostCells)
{
  allocateGridVar<Patch::ZFaceBased>(var, d_sfczDB, label, matlIndex, patch,
				     gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCZVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  checkPutAccess(label, matlIndex, patch, replace);
  checkAllocation(var, label);
  
  d_lock.writeLock();

   // Error checking
   if(!replace && d_sfczDB.exists(label, matlIndex, patch))
      throw InternalError("SFCZ variable already exists: "+label->getName());

   IntVector low, high, size;
   var.getSizes(low, high, size);
   if(low != patch->getSFCZLowIndex() || high != patch->getSFCZHighIndex()) {
#if 0
      cerr << "Warning, rewindowing array: " << label->getName() << " on patch " << patch->getID() << '\n';
#endif
      SFCZVariableBase* newvar = var.clone();
      newvar->rewindow(patch->getSFCZLowIndex(), patch->getSFCZHighIndex());
      d_sfczDB.put(label, matlIndex, patch, newvar, true);
   } else {

      // Put it in the database
      d_sfczDB.put(label, matlIndex, patch, var.clone(), true);
   }
  d_lock.writeUnlock();
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
OnDemandDataWarehouse::deleteParticles(ParticleSubset* /*delset*/)
{
   // Not implemented
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

#ifdef __sgi
#pragma set woff 1424
#endif  

template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
void OnDemandDataWarehouse::
getGridVar(VariableBase& var, DWDatabase& db,
	   const VarLabel* label, int matlIndex, const Patch* patch,
	   Ghost::GhostType gtype, int numGhostCells)
{
  d_lock.readLock();
  ASSERT(basis == Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));
   if(!db.exists(label, matlIndex, patch))
     throw UnknownVariable(label->getName(), patch, matlIndex);
   db.get(label, matlIndex, patch, var);
   
   if (gtype == Ghost::None) {
     if(numGhostCells != 0)
       throw InternalError("Ghost cells specified with task type none!\n");
   }
   else {
     IntVector low = patch->getLowIndex(basis);
     IntVector high = patch->getHighIndex(basis);
     IntVector dn = high - low;
     long total = dn.x()*dn.y()*dn.z();

     Level::selectType neighbors;
     IntVector lowIndex, highIndex;
     patch->computeVariableExtents(basis, gtype, numGhostCells,
				   neighbors, lowIndex, highIndex);
     if (!var.rewindow(lowIndex, highIndex)) {
       // reallocation needed
       if (show_warnings) {
	 cerr << "Reallocation Warning: Reallocation needed for " << label->getName();
	 if (patch)
	   cerr << " on patch " << patch;
	 cerr << " for material " << matlIndex;
	 const Task* currentTask = getCurrentTask();
	 if (currentTask != 0) {
	   cerr << " in " << currentTask->getName() << ". " <<endl;
	 }
	 else
	   cerr << ".\n";
       }
     }
     
     for(int i=0;i<(int)neighbors.size();i++){
       const Patch* neighbor = neighbors[i];
       if(neighbor && (neighbor != patch)){
	 if(!db.exists(label, matlIndex, neighbor))
	   throw UnknownVariable(label->getName(), neighbor, matlIndex,
				 neighbor == patch?"on patch":"on neighbor");
	 VariableBase* srcvar = db.get(label, matlIndex, neighbor);
	 
	 low = Max(lowIndex, neighbor->getLowIndex(basis));
	 high= Min(highIndex, neighbor->getHighIndex(basis));

	 if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	     || ( high.z() < low.z() ) )
	   throw InternalError("Patch doesn't overlap?");
	    
	 var.copyPatch(srcvar, low, high);

	 dn = high-low;
	 total+=dn.x()*dn.y()*dn.z();
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

  d_lock.readUnlock();
}

template <Patch::VariableBasis basis, class VariableBase, class DWDatabase>
void OnDemandDataWarehouse::
allocateGridVar(VariableBase& var, DWDatabase& db,
		const VarLabel* label, int matlIndex, const Patch* patch,
		Ghost::GhostType gtype, int numGhostCells)
{
  d_lock.writeLock();

   // Error checking
   if(db.exists(label, matlIndex, patch)){
     /*
       cerrLock.lock();
       cerr << string("allocate: variable already exists!\n";
       cerrLock.unlock();
     */
     throw InternalError( "allocate: variable already exists: " +
			  label->getName() + patch->toString() );
   }

   // Allocate the variable
   IntVector expLowIndex, expHighIndex;
   d_scheduler->getExpectedExtents(label, patch, expLowIndex, expHighIndex);
   IntVector lowOffset, highOffset;
   IntVector lowIndex = patch->getLowIndex(basis);
   IntVector highIndex = patch->getHighIndex(basis);
   if (numGhostCells > 0) {
     Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
			    numGhostCells, lowOffset, highOffset);
     patch->computeExtents(basis, lowOffset, highOffset, lowIndex, highIndex);
     lowIndex = Min(expLowIndex, lowIndex);
     highIndex = Max(expHighIndex, highIndex);
   }
#if 1
   // turn this off while testing
   var.allocate(Min(expLowIndex, lowIndex), Max(expHighIndex, highIndex));
#else
   var.allocate(lowIndex, highIndex);
#endif
   var.rewindow(lowIndex, highIndex);
   var.setAllocationLabel(label);
   
  d_lock.writeUnlock();
}

#ifdef __sgi
#pragma reset woff 1424
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
OnDemandDataWarehouse::checkAllocation(const Variable& var,
				       const VarLabel* label)
{
#if SCI_ASSERTION_LEVEL >= 1
  if (var.getAllocationLabel() != label) {
    const Task* currentTask = getCurrentTask();    
    if (currentTask == 0 ||
	(string(currentTask->getName()) != "Relocate::relocateParticles")) {
      IncorrectAllocation errorObj(label, var.getAllocationLabel());

      if (show_warnings) {
	cerr << errorObj.message();
	if (currentTask)
	  cerr << " in " << currentTask->getName() << ".\n";
	else
	  cerr << ".\n";
      }
    }
    //throw IncorrectAllocation(label, var.getAllocationLabel());
  }
#endif
}

inline void OnDemandDataWarehouse::checkGetAccess(const VarLabel* label,
						  int matlIndex,
						  const Patch* patch)
{
#if SCI_ASSERTION_LEVEL >= 1
  // If it was accessed by the current task already, then it should
  // have get access (i.e. if you put it in, you should be able to get it
  // right back out).
  map<SpecificVarLabel, AccessType>::iterator findIter;
  findIter = d_currentTaskAccesses.find(SpecificVarLabel(label, matlIndex,
							 patch));
  if (findIter == d_currentTaskAccesses.end()) {
    // it hasn't been accessed by this task previous, so check that it can.
    if (!hasGetAccess(label, matlIndex, patch)) {
      const Task* currentTask = getCurrentTask();
      if (currentTask == 0 ||
	  (string(currentTask->getName()) != "Relocate::relocateParticles")) {
	DeniedAccess errorObj(label, currentTask, matlIndex, patch, "requires", isFinalized() ? "get from oldDW" : "get from newDW");
	if (show_warnings) {
	  cerr << errorObj.message() << endl;
	}
      }
    //throw DeniedAccess(label, getCurrentTask(), replace ? "modify" : "put");
    }
    else {
      d_currentTaskAccesses[SpecificVarLabel(label, matlIndex, patch)]
	= GetAccess;
    }
  }
#endif
}

inline void
OnDemandDataWarehouse::checkPutAccess(const VarLabel* label, int matlIndex,
				      const Patch* patch, bool replace)
{ 
#if SCI_ASSERTION_LEVEL >= 1
  if (!hasPutAccess(label, matlIndex, patch, replace)) {
    const Task* currentTask = getCurrentTask();
    if (currentTask == 0 ||
	(string(currentTask->getName()) != "Relocate::relocateParticles")) {
      DeniedAccess errorObj(label, currentTask, matlIndex, patch,replace ? "modifies" : "computes", replace ? "modify into the datawarehouse" : "put into the datawarehouse");
      if (show_warnings) {
	cerr << errorObj.message() << endl;
      }
      //throw DeniedAccess(label, getCurrentTask(), replace ? "modify" : "put");
    }
  }
  else {
    d_currentTaskAccesses[SpecificVarLabel(label, matlIndex, patch)] =
      replace ? ModifyAccess : PutAccess;
  }
#endif
}

inline void
OnDemandDataWarehouse::checkModifyAccess(const VarLabel* label, int matlIndex,
					 const Patch* patch)
{ checkPutAccess(label, matlIndex, patch, true); }


inline bool
OnDemandDataWarehouse::hasGetAccess(const VarLabel* label, int matlIndex,
				    const Patch* patch)
{ 
  const Task* currentTask = getCurrentTask();
  if (currentTask) {
    return
      currentTask->hasRequires(label, matlIndex, patch,
			       isFinalized() ? Task::OldDW : Task::NewDW);
  }
  else
    return true; // may just be the simulation controller calling this
}

inline
bool OnDemandDataWarehouse::hasPutAccess(const VarLabel* label, int matlIndex,
					 const Patch* patch, bool replace)
{
  const Task* currentTask = getCurrentTask();
  if (currentTask) {
    if (replace)
      return currentTask->hasModifies(label, matlIndex, patch);
    else
      return currentTask->hasComputes(label, matlIndex, patch);
  }
  else
    return true; // may just be the simulation controller calling this
}

void OnDemandDataWarehouse::setCurrentTask(const Task* task)
{
  if (task)
    d_runningTasks[Thread::self()] = task;
  else
    d_runningTasks.erase(Thread::self());
  d_currentTaskAccesses.clear();
}

inline const Task* OnDemandDataWarehouse::getCurrentTask()
{
  map<Thread*, const Task*>::iterator findIt =
    d_runningTasks.find(Thread::self());
  if (findIt == d_runningTasks.end())
    return 0;
  else
    return (*findIt).second;
}

void OnDemandDataWarehouse::checkTasksAccesses(const PatchSubset* patches,
					       const MaterialSubset* matls)
{
#if SCI_ASSERTION_LEVEL >= 1
  const Task* currentTask = getCurrentTask();
  ASSERT(currentTask != 0);
  
  if (isFinalized()) {
    checkAccesses(currentTask, currentTask->getRequires(), GetAccess,
		  patches, matls);
  }
  else {
    checkAccesses(currentTask, currentTask->getRequires(), GetAccess,
		  patches, matls);
    checkAccesses(currentTask, currentTask->getComputes(), PutAccess,
		  patches, matls);
    checkAccesses(currentTask, currentTask->getModifies(), ModifyAccess,
		  patches, matls);
  }
#endif
}

void
OnDemandDataWarehouse::checkAccesses(const Task* currentTask,
				     const Task::Dependency* dep,
				     AccessType accessType,
				     const PatchSubset* avail_patches,
				     const MaterialSubset* avail_matls)
{
  if (currentTask->isReductionTask())
    return; // no need to check reduction tasks.
  
  PatchSubset default_patches;
  MaterialSubset default_matls;
  default_patches.add(0);
  default_matls.add(-1);
  if (avail_patches == 0)
    avail_patches = &default_patches;
  if (avail_matls == 0)
    avail_matls = &default_matls;
  
  for (; dep != 0; dep = dep->next) {
    if ((isFinalized() && dep->dw == Task::NewDW) ||
	(!isFinalized() && dep->dw == Task::OldDW))
      continue;
    
    const VarLabel* label = dep->var;
    const PatchSubset* patches = dep->patches;
    if (label->typeDescription() &&
	label->typeDescription()->isReductionVariable()) {
      patches = &default_patches;
    }     
    else if (patches == 0) {
      if (currentTask->getPatchSet() != 0)
	patches = currentTask->getPatchSet()->getUnion();
      else
	patches = &default_patches;
    }
    const MaterialSubset* matls = dep->matls;
    if (matls == 0) {
      if (currentTask->getMaterialSet() != 0)
	matls = currentTask->getMaterialSet()->getUnion();
      else
	matls = &default_matls;
    }
    
    if (string(currentTask->getName()) == "Relocate::relocateParticles")
      continue;
    
    for (int m = 0; m < matls->size(); m++) {
      int matl = matls->get(m);
      if (!avail_matls->contains(matl))
	continue;  // matl not handled by the running Detailed Task
      
      for (int p = 0; p < patches->size(); p++) {
	const Patch* patch = patches->get(p);
	if (!avail_patches->contains(patch))
	  continue;  // patch not handled by the running Detailed Task
	
	SpecificVarLabel key(label, matl, patch);
	map<SpecificVarLabel, AccessType>::iterator find_iter;
	find_iter = d_currentTaskAccesses.find(key);
	if (find_iter == d_currentTaskAccesses.end() ||
	    (*find_iter).second != accessType) {
	  if (show_warnings) {
	    cerr << "Task Dependency Warning: " << currentTask->getName() << " was supposed to ";
	    if (accessType == PutAccess) {
	      cerr << "put " << label->getName();
	    }
	    else if (accessType == GetAccess) {
	      cerr << "get " << label->getName();
	      if (isFinalized())
		cerr << " from the old datawarehouse";
	    }
	    else {
	      cerr << "modify " << label->getName();
	    }
	    if (patch)
	      cerr << " on patch " << patch->getID();
	    cerr << " for material " << matl;
	    cerr << " but didn't.\n";
	  }
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
