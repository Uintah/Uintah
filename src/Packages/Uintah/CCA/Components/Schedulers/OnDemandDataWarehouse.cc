
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/DebugStream.h>

#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
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

OnDemandDataWarehouse::OnDemandDataWarehouse( const ProcessorGroup* myworld,
					      int generation, const GridP& grid )
   : DataWarehouse( myworld, generation),
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
OnDemandDataWarehouse::put(const Variable* var, const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   union {
      const ReductionVariableBase* reduction;
      const ParticleVariableBase* particle;
      const NCVariableBase* nc;
      const CCVariableBase* cc;
      const SFCXVariableBase* sfcx;
      const SFCYVariableBase* sfcy;
      const SFCZVariableBase* sfcz;
   } castVar;

   if ((castVar.reduction = dynamic_cast<const ReductionVariableBase*>(var))
       != NULL)
      put(*castVar.reduction, label, matlIndex);
   else if ((castVar.particle = dynamic_cast<const ParticleVariableBase*>(var))
	    != NULL)
      put(*castVar.particle, label);
   else if ((castVar.nc = dynamic_cast<const NCVariableBase*>(var)) != NULL)
      put(*castVar.nc, label, matlIndex, patch);
   else if ((castVar.cc = dynamic_cast<const CCVariableBase*>(var)) != NULL)
      put(*castVar.cc, label, matlIndex, patch);
   else if ((castVar.sfcx=dynamic_cast<const SFCXVariableBase*>(var)) != NULL)
      put(*castVar.sfcx, label, matlIndex, patch);
   else if ((castVar.sfcy=dynamic_cast<const SFCYVariableBase*>(var)) != NULL)
      put(*castVar.sfcy, label, matlIndex, patch);
   else if ((castVar.sfcz=dynamic_cast<const SFCZVariableBase*>(var)) != NULL)
      put(*castVar.sfcz, label, matlIndex, patch);
   else
      throw InternalError("Unknown Variable type");
}

void
OnDemandDataWarehouse::get(ReductionVariableBase& var,
			   const VarLabel* label, int matlIndex /*= -1*/)
{
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
	ParticleVariable<Point> pos;
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
OnDemandDataWarehouse::allocate(ReductionVariableBase&,
				const VarLabel*, int)
{
   cerr << "OnDemand DataWarehouse::allocate(ReductionVariable) "
	<< "not finished\n";
}

void
OnDemandDataWarehouse::put(const ReductionVariableBase& var,
			   const VarLabel* label, int matlIndex /* = -1 */)
{
  d_lock.writeLock();
   ASSERT(!d_finalized);

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
	 ParticleVariable<Point> pos;
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
OnDemandDataWarehouse::get(ParticleVariableBase& var,
			   const VarLabel* label,
			   ParticleSubset* pset)
{
  d_lock.readLock();
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

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
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(const ParticleVariableBase& var,
			   const VarLabel* label, bool replace /*= false*/)
{
   ASSERT(!d_finalized);

   ParticleSubset* pset = var.getParticleSubset();
   if(pset->numGhostCells() != 0 || pset->getGhostType() != 0)
      throw InternalError("ParticleVariable cannot use put with ghost cells");
   const Patch* patch = pset->getPatch();
   int matlIndex = pset->getMatlIndex();

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
OnDemandDataWarehouse::get(NCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
  d_lock.readLock();
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      if(!d_ncDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      d_ncDB.get(label, matlIndex, patch, var);
   } else {
      Level::selectType neighbors;
      IntVector lowIndex, highIndex;
      patch->computeVariableExtents(Patch::NodeBased, gtype, numGhostCells,
				    neighbors, lowIndex, highIndex);
      var.allocate(lowIndex, highIndex);
      long totalNodes=0;
      for(int i=0;i<(int)neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_ncDB.exists(label, matlIndex, neighbor))
	       throw UnknownVariable(label->getName(), neighbor, matlIndex,
				     neighbor == patch?"on patch":"on neighbor");
	    NCVariableBase* srcvar = 
	       d_ncDB.get(label, matlIndex, neighbor);

	    IntVector low = Max(lowIndex, neighbor->getNodeLowIndex());
	    IntVector high= Min(highIndex, neighbor->getNodeHighIndex());

	    if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
		|| ( high.z() < low.z() ) )
	       throw InternalError("Patch doesn't overlap?");
	    
	    var.copyPatch(srcvar, low, high);
	    IntVector dnodes = high-low;
	    totalNodes+=dnodes.x()*dnodes.y()*dnodes.z();
	 }
      }
      //IntVector dn = highIndex-lowIndex;
      //      long wantnodes = dn.x()*dn.y()*dn.z();
      //      ASSERTEQ(wantnodes, totalNodes);
//      if(wantnodes!=totalNodes){
	// This ASSERT or this warning are invoked even when trying
	// to do pefectly legitimate things.
	//  cerr << "Warning:  wantnodes != totalNodes " << endl;
//      }
   }
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::allocate(NCVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch, const IntVector gc)
{
  d_lock.writeLock();

#if DAV_DEBUG
  cerr << "alloc: NC var: " << *label << *patch 
       << " MI: " << matlIndex << "\n";
#endif

  // Error checking
  if(d_ncDB.exists(label, matlIndex, patch)){
    cerrLock.lock();
    cerr << "allocate: NC var already exists!\n";
    cerrLock.unlock();
    throw InternalError( "allocate: NC variable already exists: " +
			 label->getName() + patch->toString() );
  }

  // Allocate the variable
  var.allocate(patch->getNodeLowIndex()-gc, patch->getNodeHighIndex()+gc);
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(const NCVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  d_lock.writeLock();
   ASSERT(!d_finalized);

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
  d_lock.readLock();
  if(!d_perpatchDB.exists(label, matlIndex, patch))
     throw UnknownVariable(label->getName(), patch, matlIndex,
			   "perpatch data");
  d_perpatchDB.get(label, matlIndex, patch, var);
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::put(const PerPatchBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  d_lock.writeLock();
   ASSERT(!d_finalized);

   // Error checking
   if(!replace && d_perpatchDB.exists(label, matlIndex, patch))
     throw InternalError("PerPatch variable already exists: "+label->getName());

   // Put it in the database
   d_perpatchDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::allocate(CCVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch, const IntVector gc)
{
  d_lock.writeLock();
   // Error checking
   if(d_ccDB.exists(label, matlIndex, patch))
      throw InternalError("CC variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getCellLowIndex()-gc, patch->getCellHighIndex()+gc);
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::get(CCVariableBase& var, const VarLabel* label,
			   int matlIndex,
			   const Patch* patch, Ghost::GhostType gtype,
			   int numGhostCells)
{
  d_lock.readLock();
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      if(!d_ccDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      d_ccDB.get(label, matlIndex, patch, var);
   } else {
      Level::selectType neighbors;
#if 1
      IntVector lowIndex, highIndex;
      patch->computeVariableExtents(Patch::CellFaceBased, gtype, numGhostCells,
				    neighbors, lowIndex, highIndex);
      var.allocate(lowIndex, highIndex);
#else
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	throw InternalError("Around Nodes not defined for CCVariable");
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // All 6 faces
	 lowIndex = patch->getGhostCellLowIndex(numGhostCells);
	 highIndex = patch->getGhostCellHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      const Level* level = patch->getLevel();
      IntVector low = lowIndex;
      IntVector high = highIndex;
      level->selectPatches(low, high, neighbors);
#endif
      long totalCells=0;
      for(int i=0;i<(int)neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_ccDB.exists(label, matlIndex, neighbor))
	       throw UnknownVariable(label->getName(), neighbor,
				     matlIndex, "CCVariable");
	    CCVariableBase* srcvar = 
	       d_ccDB.get(label, matlIndex, neighbor);

	    IntVector low = Max(lowIndex, neighbor->getCellLowIndex());
	    IntVector high= Min(highIndex, neighbor->getCellHighIndex());
#if 0
	    cerr << "neighbor: " << neighbor->getID() << ", low=" << neighbor->getCellLowIndex() << ", high=" << neighbor->getCellHighIndex() << '\n';
	    cerr << "low=" << low << ", high=" << high << '\n';
#endif

	    if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
		|| ( high.z() < low.z() ) )
	       throw InternalError("Patch doesn't overlap?");

	    var.copyPatch(srcvar, low, high);
	    IntVector dcells = high-low;
	    totalCells+=dcells.x()*dcells.y()*dcells.z();
	 }
      }
      IntVector dn = highIndex-lowIndex;
      long wantcells = dn.x()*dn.y()*dn.z();
      ASSERTEQ(wantcells, totalCells);
   }
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::put(const CCVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  d_lock.writeLock();
   ASSERT(!d_finalized);
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
OnDemandDataWarehouse::get(SFCXVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
  d_lock.readLock();
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      if(!d_sfcxDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      d_sfcxDB.get(label, matlIndex, patch, var);
   } else {
     Level::selectType neighbors;
#if 1

      IntVector lowIndex, highIndex;
      patch->computeVariableExtents(Patch::XFaceBased, gtype, numGhostCells,
				    neighbors, lowIndex, highIndex);
      var.allocate(lowIndex, highIndex);
#else
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	throw InternalError("Ghost::AroundNodes: illegal ghost type for SFCX Variable");
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Upper neighbors
	 lowIndex = patch->getGhostSFCXLowIndex(numGhostCells);
         highIndex = patch->getGhostSFCXHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      IntVector low = patch->getGhostCellLowIndex(numGhostCells);
      IntVector high = patch->getGhostCellHighIndex(numGhostCells);
      const Level* level = patch->getLevel();
      level->selectPatches(low, high, neighbors);

#endif
      long totalCells=0;
      // modify it to only ignore corner nodes
      for(int i=0;i<(int)neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_sfcxDB.exists(label, matlIndex, neighbor))
	       throw UnknownVariable(label->getName(), neighbor,
				     matlIndex, "SFCXVariable");
	    SFCXVariableBase* srcvar = 
	       d_sfcxDB.get(label, matlIndex, neighbor);

	    IntVector low = Max(lowIndex, neighbor->getSFCXLowIndex());
	    IntVector high= Min(highIndex, neighbor->getSFCXHighIndex());

	    if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
		|| ( high.z() < low.z() ) )
	       throw InternalError("Patch doesn't overlap?");
	    
	    var.copyPatch(srcvar, low, high);
	    IntVector dcells = high-low;
	    totalCells+=dcells.x()*dcells.y()*dcells.z();
	 }
      }
      IntVector dn = highIndex-lowIndex;
      long wantcells = dn.x()*dn.y()*dn.z();
      ASSERTEQ(wantcells, totalCells);
   }
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::allocate(SFCXVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
  d_lock.writeLock();
   // Error checking
  if(d_sfcxDB.exists(label, matlIndex, patch))
    throw InternalError("SFCX variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getSFCXLowIndex(), patch->getSFCXHighIndex());
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(const SFCXVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  d_lock.writeLock();
   ASSERT(!d_finalized);

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
OnDemandDataWarehouse::get(SFCYVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
  d_lock.readLock();
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      if(!d_sfcyDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      d_sfcyDB.get(label, matlIndex, patch, var);
   } else {
      Level::selectType neighbors;
#if 1
      IntVector lowIndex, highIndex;
      patch->computeVariableExtents(Patch::YFaceBased, gtype, numGhostCells,
				    neighbors, lowIndex, highIndex);
      var.allocate(lowIndex, highIndex);
#else
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	throw InternalError("Ghost::AroundNodes: illegal ghost type for SFCY Variable");
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Upper neighbors
	 lowIndex = patch->getGhostSFCYLowIndex(numGhostCells);
         highIndex = patch->getGhostSFCYHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      IntVector low = patch->getGhostCellLowIndex(numGhostCells);
      IntVector high = patch->getGhostCellHighIndex(numGhostCells);
      const Level* level = patch->getLevel();
      level->selectPatches(low, high, neighbors);
#endif
      long totalCells=0;
      for(int i=0;i<(int)neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_sfcyDB.exists(label, matlIndex, neighbor))
	       throw UnknownVariable(label->getName(), neighbor,
				     matlIndex, "SFCYVariable");
	    SFCYVariableBase* srcvar = 
	       d_sfcyDB.get(label, matlIndex, neighbor);

	    IntVector low = Max(lowIndex, neighbor->getSFCYLowIndex());
	    IntVector high= Min(highIndex, neighbor->getSFCYHighIndex());

	    if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
		|| ( high.z() < low.z() ) )
	       throw InternalError("Patch doesn't overlap?");
	    
	    var.copyPatch(srcvar, low, high);
	    IntVector dcells = high-low;
	    totalCells+=dcells.x()*dcells.y()*dcells.z();
	 }
      }
      IntVector dn = highIndex-lowIndex;
      long wantcells = dn.x()*dn.y()*dn.z();
      ASSERTEQ(wantcells, totalCells);
   }
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::allocate(SFCYVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
  d_lock.writeLock();
   // Error checking
   if(d_sfcyDB.exists(label, matlIndex, patch))
      throw InternalError("SFCY variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getSFCYLowIndex(), patch->getSFCYHighIndex());
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(const SFCYVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  d_lock.writeLock();
   ASSERT(!d_finalized);

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
OnDemandDataWarehouse::get(SFCZVariableBase& var, const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   Ghost::GhostType gtype,
			   int numGhostCells)
{
  d_lock.readLock();
   if(gtype == Ghost::None) {
      if(numGhostCells != 0)
	 throw InternalError("Ghost cells specified with task type none!\n");
      if(!d_sfczDB.exists(label, matlIndex, patch))
	 throw UnknownVariable(label->getName(), patch, matlIndex);
      d_sfczDB.get(label, matlIndex, patch, var);
   } else {
     Level::selectType neighbors;
#if 1
      IntVector lowIndex, highIndex;
      patch->computeVariableExtents(Patch::ZFaceBased, gtype, numGhostCells,
				    neighbors, lowIndex, highIndex);
      var.allocate(lowIndex, highIndex);
#else
      int l,h;
      IntVector gc(numGhostCells, numGhostCells, numGhostCells);
      IntVector lowIndex;
      IntVector highIndex;
      switch(gtype){
      case Ghost::AroundNodes:
	throw InternalError("Ghost::AroundNodes: illegal ghost type for SFCZ Variable");
      case Ghost::AroundCells:
	 if(numGhostCells == 0)
	    throw InternalError("No ghost cells specified with Task::AroundCells");
	 // Upper neighbors
	 lowIndex = patch->getGhostSFCZLowIndex(numGhostCells);
         highIndex = patch->getGhostSFCZHighIndex(numGhostCells);
	 break;
      default:
	 throw InternalError("Illegal ghost type");
      }
      var.allocate(lowIndex, highIndex);
      IntVector low = patch->getGhostCellLowIndex(numGhostCells);
      IntVector high=patch->getGhostCellHighIndex(numGhostCells);
      const Level* level = patch->getLevel();
      level->selectPatches(low, high, neighbors);
#endif
      long totalCells=0;
      for(int i=0;i<(int)neighbors.size();i++){
	 const Patch* neighbor = neighbors[i];
	 if(neighbor){
	    if(!d_sfczDB.exists(label, matlIndex, neighbor))
	       throw UnknownVariable(label->getName(), neighbor,
				     matlIndex, "SFCZVariable");
	    SFCZVariableBase* srcvar = 
	       d_sfczDB.get(label, matlIndex, neighbor);

	    IntVector low = Max(lowIndex, neighbor->getSFCZLowIndex());
	    IntVector high= Min(highIndex, neighbor->getSFCZHighIndex());

	    if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
		|| ( high.z() < low.z() ) )
	       throw InternalError("Patch doesn't overlap?");
	    
	    var.copyPatch(srcvar, low, high);
	    IntVector dcells = high-low;
	    totalCells+=dcells.x()*dcells.y()*dcells.z();
	 }
      }
      IntVector dn = highIndex-lowIndex;
      long wantcells = dn.x()*dn.y()*dn.z();
      ASSERTEQ(wantcells, totalCells);
   }
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::allocate(SFCZVariableBase& var,
				const VarLabel* label,
				int matlIndex,
				const Patch* patch)
{
  d_lock.writeLock();
   // Error checking
   if(d_sfczDB.exists(label, matlIndex, patch))
      throw InternalError("SFCZ variable already exists: "+label->getName());

   // Allocate the variable
   var.allocate(patch->getSFCZLowIndex(), patch->getSFCZHighIndex());
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(const SFCZVariableBase& var,
			   const VarLabel* label,
			   int matlIndex, const Patch* patch,
			   bool replace /*= false*/)
{
  d_lock.writeLock();
   ASSERT(!d_finalized);

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
				 int matlIndex, const Patch* patch) const
{
  d_lock.readLock();

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
				 int matlIndex /* = -1 */) const
{
  d_lock.readLock();

   try {
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

void OnDemandDataWarehouse::logMemoryUse(ostream& out, const std::string& tag)
{
  int dwid=d_generation;
  d_ncDB.logMemoryUse(out, tag, dwid);
  d_ccDB.logMemoryUse(out, tag, dwid);
  d_sfcxDB.logMemoryUse(out, tag, dwid);
  d_sfcyDB.logMemoryUse(out, tag, dwid);
  d_sfczDB.logMemoryUse(out, tag, dwid);
  d_particleDB.logMemoryUse(out, tag, dwid);
  d_reductionDB.logMemoryUse(out, tag, dwid);
  d_perpatchDB.logMemoryUse(out, tag, dwid);

  // Log the psets.
  for(psetDBType::iterator iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++){
    ParticleSubset* pset = iter->second;
    out << dwid << "\t" << tag << "\t" << "ParticleSubset" << "\t-\t" << pset->getPatch()->getID() << "\t" << pset->getMatlIndex() << "\t" << pset->numParticles() << "\t" << pset->numParticles()*sizeof(particleIndex) << "\t" << pset->getPointer() << '\n';
  }
}

