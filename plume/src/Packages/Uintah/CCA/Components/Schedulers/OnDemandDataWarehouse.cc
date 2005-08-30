#include <TauProfilerForSCIRun.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/ProgressiveWarning.h>
#include <Core/Util/FancyAssert.h>

#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SendState.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DependencyException.h>
#include <Packages/Uintah/CCA/Components/Schedulers/IncorrectAllocation.h>
#include <Packages/Uintah/Core/Exceptions/TypeMismatchException.h>
#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Packages/Uintah/Core/Parallel/BufferInfo.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>

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

static DebugStream dbg( "OnDemandDataWarehouse", false );
static DebugStream warn( "OnDemandDataWarehouse_warn", true );
extern DebugStream mpidbg;

static Mutex ssLock( "send state lock" );

#define PARTICLESET_TAG		0x4000
#define DAV_DEBUG 0

OnDemandDataWarehouse::OnDemandDataWarehouse(const ProcessorGroup* myworld,
					     Scheduler* scheduler,
					     int generation, const GridP& grid,
					     bool isInitializationDW/*=false*/)
   : DataWarehouse(myworld, scheduler, generation),
     d_lock("DataWarehouse lock"),
     d_finalized( false ),
     d_grid(grid),
     d_isInitializationDW(isInitializationDW)
{
  restart = false;
  aborted = false;
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

  for (psetAddDBType::const_iterator iter = d_addsetDB.begin();
       iter != d_addsetDB.end(); iter++) {
    map<const VarLabel*, ParticleVariableBase*>::const_iterator pvar_itr;
    for (pvar_itr = iter->second->begin(); pvar_itr != iter->second->end();
	 pvar_itr++)
      delete pvar_itr->second;
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
   cerr << "Total Grid alloc: " << totalGridAlloc << "\n";
   totalGridAlloc = 0;
#endif
   
  d_lock.writeUnlock();
}

void OnDemandDataWarehouse::unfinalize()
{
  // this is for processes that need to make small modifications to the DW
  // after it has been finalized.
  d_finalized=false;
}

void OnDemandDataWarehouse::refinalize()
{
  d_finalized=true;
}

void
OnDemandDataWarehouse::put(Variable* var, const VarLabel* label,
			   int matlIndex, const Patch* patch)
{
   union {
      ReductionVariableBase* reduction;
      SoleVariableBase* sole;
      ParticleVariableBase* particle;
      NCVariableBase* nc;
      CCVariableBase* cc;
      SFCXVariableBase* sfcx;
      SFCYVariableBase* sfcy;
      SFCZVariableBase* sfcz;
   } castVar;

   if ((castVar.reduction = dynamic_cast<ReductionVariableBase*>(var))
       != NULL)
      put(*castVar.reduction, label, patch?patch->getLevel():0, matlIndex);
   else if ((castVar.sole = dynamic_cast<SoleVariableBase*>(var))
	    != NULL)
      put(*castVar.sole, label,patch?patch->getLevel():0,matlIndex);
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
     SCI_THROW(InternalError("Unknown Variable type", __FILE__, __LINE__));
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
     SCI_THROW(InternalError("OnDemandDataWarehouse::allocateAndPutGridVar: Not a grid variable type", __FILE__, __LINE__));
}

void
OnDemandDataWarehouse::get(ReductionVariableBase& var,
			   const VarLabel* label, const Level* level,
			   int matlIndex /*= -1*/)
{
  d_lock.readLock();
  
  checkGetAccess(label, matlIndex, 0);

  if(!d_reductionDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			      "on reduction", __FILE__, __LINE__));
  }
  d_reductionDB.get(label, matlIndex, level, var);

  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::get(SoleVariableBase& var,
			   const VarLabel* label, const Level* level,
			   int matlIndex /*= -1*/)
{
  d_lock.readLock();
  
  checkGetAccess(label, matlIndex, 0);

  if(!d_soleDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			      "on sole", __FILE__, __LINE__));
  }
  d_soleDB.get(label, matlIndex, level, var);

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
       d_sfczDB.exists(label,matlIndex,patch) ||
       d_soleDB.exists(label, matlIndex, patch->getLevel()) ||
       d_reductionDB.exists(label, matlIndex, patch->getLevel()) ) {

     d_lock.readUnlock();
     return true;
   } else {
     d_lock.readUnlock();
     return false;
   }
}

void
OnDemandDataWarehouse::sendMPI(SendState& ss, SendState& rs, DependencyBatch* batch,
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
      IntVector range = dep->high - dep->low;
      IntVector low = dep->low;
      IntVector high = dep->high;

      if(!d_particleDB.exists(label, matlIndex, patch))
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI", __FILE__, __LINE__));
      ParticleVariableBase* var = d_particleDB.get(label, matlIndex, patch);

      int dest = batch->toTasks.front()->getAssignedResourceIndex();
      ASSERTRANGE(dest, 0, d_myworld->size());

      ssLock.lock();  // Dd: ??
      // in a case where there is a sendset with a different ghost configuration
      // than we want, there can be problems with dynamic load balancing, when patch
      // used to be on this processor and now we only want ghost data from patch.  So
      // check if dest previously sent us this entire patch, if so, just use that sendset
      ParticleSubset* sendset = rs.find_sendset(dest, patch, matlIndex, patch->getLowIndex(), patch->getHighIndex(),
                                                old_dw->d_generation);
      if (sendset) {
        fflush(stdout);
        sendset = old_dw->getParticleSubset(matlIndex, patch, patch->getLowIndex(), patch->getHighIndex());
        low = patch->getLowIndex();
        high = patch->getHighIndex();
      }
      else
        sendset = ss.find_sendset(dest, patch, matlIndex, low, high, old_dw->d_generation);
      ssLock.unlock();  // Dd: ??

      if(!sendset){

        mixedDebug << "sendset is NULL\n";

        ParticleSubset* pset = var->getParticleSubset();
        ssLock.lock();  // Dd: ??
        sendset = scinew ParticleSubset(pset->getParticleSet(),
                                        false, matlIndex, patch, low, high, 0);
        ssLock.unlock();  // Dd: ??
        constParticleVariable<Point> pos;
        old_dw->get(pos, pos_var, pset);
        Box box=pset->getPatch()->getLevel()->getBox(low, high);
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
#if SCI_ASSERTION_LEVEL >= 1
	int* maxtag, found;
	MPI_Attr_get(d_myworld->getComm(), MPI_TAG_UB, &maxtag, &found);
	ASSERT(found);
	ASSERT((PARTICLESET_TAG|batch->messageTag) <= (*maxtag));
#endif
        ASSERT(batch->messageTag >= 0);
        
        mpidbg << d_myworld->myrank() << " Sending PARTICLE message " << (PARTICLESET_TAG|batch->messageTag) << ", to " << dest << ", patch " << patch->getID() << ", matl " << matlIndex << ", length: " << 1 << "(" << numParticles << ")\n"; cerrLock.unlock();

        MPI_Bsend(&numParticles, 1, MPI_INT, dest,
                  PARTICLESET_TAG|batch->messageTag, d_myworld->getComm());
        mpidbg << d_myworld->myrank() << " Done Sending PARTICLE message " << (PARTICLESET_TAG|batch->messageTag) << ", to " << dest << ", patch " << patch->getID() << ", matl " << matlIndex << ", length: " << 1 << "(" << numParticles << ")\n"; cerrLock.unlock();
        ssLock.lock();  // Dd: ??       
        ss.add_sendset(sendset, dest, patch, matlIndex, low, high, old_dw->d_generation);
        ssLock.unlock();  // Dd: ??
      }
        
      ssLock.lock();  // Dd: ?? 
      int numParticles = sendset->numParticles();
      ssLock.unlock(); // Dd: ??

      // dbg << d_myworld->myrank() << " sendset has " << numParticles << " particles - patch " << patch->getID() << ' ' << "M: " << matlIndex << " GT: (" << gt << ',' << ngc << "), on dest: " << dest << "\n";

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
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI", __FILE__, __LINE__));
      NCVariableBase* var = d_ncDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::CCVariable:
    {
      if(!d_ccDB.exists(label, matlIndex, patch))
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI", __FILE__, __LINE__));
      CCVariableBase* var = d_ccDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCXVariable:
    {
      if(!d_sfcxDB.exists(label, matlIndex, patch))
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI", __FILE__, __LINE__));
      SFCXVariableBase* var = d_sfcxDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCYVariable:
    {
      if(!d_sfcyDB.exists(label, matlIndex, patch))
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI", __FILE__, __LINE__));
      SFCYVariableBase* var = d_sfcyDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  case TypeDescription::SFCZVariable:
    {
      if(!d_sfczDB.exists(label, matlIndex, patch))
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI", __FILE__, __LINE__));
      SFCZVariableBase* var = d_sfczDB.get(label, matlIndex, patch);
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  default:
    SCI_THROW(InternalError("sendMPI not implemented for "+label->getFullName(matlIndex, patch), __FILE__, __LINE__));
  } // end switch( label->getType() );
 d_lock.readUnlock();  
}

void
OnDemandDataWarehouse::recvMPI(SendState& rs, BufferInfo& buffer,
                               DependencyBatch* batch,
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
      IntVector low = dep->low;
      IntVector high = dep->high;

      // First, get the particle set.  We should already have it
      //      if(!old_dw->haveParticleSubset(matlIndex, patch, gt, ngc)){
      int from=batch->fromTask->getAssignedResourceIndex();

      // if we already have a subset for the entire patch, there's little point 
      // in getting another one (and if we did, it would cause problems - see
      // comment in sendMPI)
      ParticleSubset* recvset;
      if (old_dw->haveParticleSubset(matlIndex, patch, patch->getLowIndex(), patch->getHighIndex())) {
        recvset = old_dw->getParticleSubset(matlIndex, patch, patch->getLowIndex(), patch->getHighIndex());
        low = patch->getLowIndex();
        high = patch->getHighIndex();
      }
      else
        recvset = rs.find_sendset(from, patch, matlIndex, low, high, old_dw->d_generation);
      
      if(!recvset){
        int numParticles;
        MPI_Status status;
        ASSERT(batch->messageTag >= 0);
	ASSERTRANGE(from, 0, d_myworld->size());
        mpidbg << d_myworld->myrank() << " Posting PARTICLES receive for message " << (PARTICLESET_TAG|batch->messageTag) << " from " << from << ", patch " << patch->getID() << ", matl " << matlIndex << ", length=" << 1 << "\n";      
        MPI_Recv(&numParticles, 1, MPI_INT, from,
                 PARTICLESET_TAG|batch->messageTag, d_myworld->getComm(),
                 &status);
        mpidbg << d_myworld->myrank() << "   recved " << numParticles << "particles\n";
        
        // sometime we have to force a receive to match a send.
        // in these cases just ignore this new subset
        ParticleSubset* psubset;
        if (!old_dw->haveParticleSubset(matlIndex, patch, low, high)) {
          psubset = old_dw->createParticleSubset(numParticles, matlIndex, patch, low, high);
        }
        else {
          //old_dw->printParticleSubsets();
          psubset = old_dw->getParticleSubset(matlIndex,patch,low,high);
          ASSERTEQ(numParticles, psubset->numParticles());
        }
        ParticleSubset* recvset = new ParticleSubset(psubset->getParticleSet(),
                                                     true, matlIndex, patch, 
                                                     low, high, 0);
        rs.add_sendset(recvset, from, patch, matlIndex, low, high, old_dw->d_generation);
      }
      ParticleSubset* pset = old_dw->getParticleSubset(matlIndex,patch,low, high);


      //brydbg << d_myworld->myrank() << " RECVset has" << pset->numParticles() << " particles - patch " << patch << ' ' << "M: " << matlIndex << "GT: (" << gt << ',' << ngc << ")\n";
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
    SCI_THROW(InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch), __FILE__, __LINE__));
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
				 const Level* level,
                                 const MaterialSubset* inmatls)
{
  const MaterialSubset* matls;
  if(!inmatls){
    MaterialSubset* tmpmatls = scinew MaterialSubset();
    tmpmatls->add(-1);
    matls = tmpmatls;
  } else {
    matls = inmatls;
  }

  // Count the number of data elements in the reduction array
  int nmatls = matls->size();
  int count=0;
  MPI_Op op;
  MPI_Datatype datatype;

  d_lock.readLock();
  for(int m=0;m<nmatls;m++){
    int matlIndex = matls->get(m);

    ReductionVariableBase* var;
    if (d_reductionDB.exists(label, matlIndex, level))
      var = d_reductionDB.get(label, matlIndex, level);
    else {

      // create a new var with a harmless value.  This will make 
      // "inactive" processors work with reduction vars
      //cout << "NEWRV1\n";
      var = dynamic_cast<ReductionVariableBase*>(label->typeDescription()->createInstance());
      //cout << "var=" << var << '\n';
      //cout << "NEWRV2\n";
      //cout << "VL = " << label->getName() << endl;
      var->setBenignValue();
      //cout << d_myworld->myrank() << " BENIGN VAL ";
      //var->print(cout);

      // put it in the db so the next get won't fail and so we won't 
      // have to delete it manually
      d_reductionDB.put(label, matlIndex, level, var, true);
      //cout << "NEWRV3\n";
      //cout << endl;
      //SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
      //				"on reduceMPI", __FILE__, __LINE__));
    }
    int sendcount;
    MPI_Datatype senddatatype = NULL;
    MPI_Op sendop = MPI_OP_NULL;
    var->getMPIInfo(sendcount, senddatatype, sendop);
    if(m==0){
      op=sendop;
      datatype=senddatatype;
    } else {
      ASSERTEQ(op, sendop);
      ASSERTEQ(datatype, senddatatype);
    }
    count += sendcount;
  }
  int packsize;
  MPI_Pack_size(count, datatype, d_myworld->getComm(), &packsize);
  vector<char> sendbuf(packsize);

  int packindex=0;
  for(int m=0;m<nmatls;m++){
    int matlIndex = matls->get(m);

    ReductionVariableBase* var;
    try {
      var = d_reductionDB.get(label, matlIndex, level);
    } catch (UnknownVariable) {
      SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
                                "on reduceMPI(pass 2)", __FILE__, __LINE__));
    }
    var->getMPIData(sendbuf, packindex);
  }
  d_lock.readUnlock();

  vector<char> recvbuf(packsize);

  if( mixedDebug.active() ) {
    cerrLock.lock(); mixedDebug << "calling MPI_Allreduce\n";
    cerrLock.unlock();
  }

  mpidbg << d_myworld->myrank() << " allreduce, name " << label->getName() << " level " << (level?level->getID():-1) << endl;
  int error = MPI_Allreduce(&sendbuf[0], &recvbuf[0], count, datatype, op,
			    d_myworld->getComm());

  mpidbg << d_myworld->myrank() << " allreduce, name " << label->getName() << " level " << (level?level->getID():-1) << endl;
  if( mixedDebug.active() ) {
    cerrLock.lock(); mixedDebug << "done with MPI_Allreduce\n";
    cerrLock.unlock();
  }

  if( error ){
    cerrLock.lock();
    cerr << "reduceMPI: MPI_Allreduce error: " << error << "\n";
    cerrLock.unlock();
    SCI_THROW(InternalError("reduceMPI: MPI error", __FILE__, __LINE__));     
  }

  d_lock.writeLock();
  int unpackindex=0;
  for(int m=0;m<nmatls;m++){
    int matlIndex = matls->get(m);

    ReductionVariableBase* var;
    try {
      var = d_reductionDB.get(label, matlIndex, level);
    } catch (UnknownVariable) {
      SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
				"on reduceMPI(pass 2)", __FILE__, __LINE__));
    }
    var->putMPIData(recvbuf, unpackindex);
  }
  d_lock.writeUnlock();
  if(matls != inmatls)
    delete matls;
}

void
OnDemandDataWarehouse::put(const ReductionVariableBase& var,
			   const VarLabel* label, const Level* level,
			   int matlIndex /* = -1 */)
{
  ASSERT(!d_finalized);
  d_lock.writeLock();

  checkPutAccess(label, matlIndex, 0,
		 false /* it actually may be replaced, but it doesn't need
			  to explicitly modify with multiple reduces in the
			  task graph */);
  // Put it in the database
  if (!d_reductionDB.exists(label, matlIndex, level))
    d_reductionDB.put(label, matlIndex, level, var.clone(), false);
  else {
    ReductionVariableBase* foundVar
      = d_reductionDB.get(label, matlIndex, level);
    foundVar->reduce(var);
  }
   
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::override(const ReductionVariableBase& var,
				const VarLabel* label, const Level* level,
				int matlIndex /*=-1*/)
{
  d_lock.writeLock();  

  checkPutAccess(label, matlIndex, 0, true);

  // Put it in the database, replace whatever may already be there
  d_reductionDB.put(label, matlIndex, level, var.clone(), true);
   
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::put(const SoleVariableBase& var,
			   const VarLabel* label, const Level* level,
			   int matlIndex /* = -1 */)
{
  ASSERT(!d_finalized);
  d_lock.writeLock();

  checkPutAccess(label, matlIndex, 0,
		 false /* it actually may be replaced, but it doesn't need
			  to explicitly modify with multiple soles in the
			  task graph */);
  // Put it in the database
  if (!d_soleDB.exists(label, matlIndex, level))
    d_soleDB.put(label, matlIndex, level, var.clone(), false);
  
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::override(const SoleVariableBase& var,
				const VarLabel* label, const Level* level,
				int matlIndex /*=-1*/)
{
  d_lock.writeLock();  

  checkPutAccess(label, matlIndex, 0, true);

  // Put it in the database, replace whatever may already be there
  d_soleDB.put(label, matlIndex, level, var.clone(), true);
   
  d_lock.writeUnlock();
}


ParticleSubset*
OnDemandDataWarehouse::createParticleSubset(particleIndex numParticles,
                                            int matlIndex, const Patch* patch,
                                            IntVector low /* = (0,0,0) */,
                                            IntVector high /* = (0,0,0) */)
{
  d_lock.writeLock();

  if (low == high && high == IntVector(0,0,0)) {
    low = patch->getLowIndex();
    high = patch->getHighIndex();
  }

  dbg << d_myworld->myrank() << " DW ID " << getID() << " createParticleSubset: MI: " << matlIndex << " P: " << patch->getID() << " (" << low << ", " << high << ")\n";

  ASSERT(!patch->isVirtual());

  ParticleSet* pset = scinew ParticleSet(numParticles);
  ParticleSubset* psubset = 
    scinew ParticleSubset(pset, true, matlIndex, patch, low, high, 0);
  
  psetDBType::key_type key(patch, matlIndex, low, high, getID());
  if(d_psetDB.find(key) != d_psetDB.end())
    SCI_THROW(InternalError("createParticleSubset called twice for patch", __FILE__, __LINE__));
  
  d_psetDB[key]=psubset;
  psubset->addReference();
  d_lock.writeUnlock();
  return psubset;
}

void
OnDemandDataWarehouse::saveParticleSubset(ParticleSubset* psubset, 
                                          int matlIndex, const Patch* patch,
                                          IntVector low /* = (0,0,0) */,
                                          IntVector high /* = (0,0,0) */)
{
  ASSERTEQ(psubset->getPatch(), patch);
  ASSERTEQ(psubset->getMatlIndex(), matlIndex);
  ASSERT(!patch->isVirtual());  
  d_lock.writeLock();

  if (low == high && high == IntVector(0,0,0)) {
    low = patch->getLowIndex();
    high = patch->getHighIndex();
  }

  psetDBType::key_type key(patch, matlIndex, low, high, getID());
  if(d_psetDB.find(key) != d_psetDB.end())
    SCI_THROW(InternalError("saveParticleSubset called twice for patch", __FILE__, __LINE__));

  d_psetDB[key]=psubset;
  psubset->addReference();

  d_lock.writeUnlock();
}

void OnDemandDataWarehouse::printParticleSubsets()
{
  psetDBType::iterator iter;
  cout << d_myworld->myrank() << " Available psets on DW " << d_generation << ":\n";
  for (iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++) {
    cout << d_myworld->myrank() << " " <<*(iter->second) << endl;
  }
  
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch)
{
  return getParticleSubset(matlIndex, patch, patch->getLowIndex(), patch->getHighIndex());
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch,
                                         IntVector low, IntVector high)
{
  d_lock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;

  psetDBType::key_type key(realPatch, matlIndex, low, high, getID());
  psetDBType::iterator iter = d_psetDB.find(key);
  if(iter == d_psetDB.end()){
    // if not found, look for an encompassing particle subset
    for (iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++) {
      const PSPatchMatlGhost& pmg = iter->first;
      if (pmg.patch_ == realPatch && pmg.matl_ == matlIndex &&
          pmg.dwid_ == getID() && 
          low.x() >= pmg.low_.x() && low.y() >= pmg.low_.y() && low.z() >= pmg.low_.z() &&
          high.x() <= pmg.high_.x() && high.y() <= pmg.high_.y() && high.z() <= pmg.high_.z())
        break;
    }
    
    if (iter == d_psetDB.end()){
      printParticleSubsets();
      d_lock.readUnlock();
      ostringstream s;
      s << "ParticleSet, (low: " << low << ", high: " << high <<  " DWID " << getID() << ')';
      SCI_THROW(UnknownVariable(s.str().c_str(), getID(), realPatch, matlIndex,
                                "Cannot find particle set on patch", __FILE__, __LINE__));
    }
  }
  d_lock.readUnlock();
  return iter->second;
}

ParticleSubset*
OnDemandDataWarehouse::getDeleteSubset(int matlIndex, const Patch* patch)
{
  d_lock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
   psetDBType::key_type key(realPatch, matlIndex, realPatch->getLowIndex(), realPatch->getHighIndex(), getID());
   psetDBType::iterator iter = d_delsetDB.find(key);
   if(iter == d_delsetDB.end()){
     d_lock.readUnlock();
     SCI_THROW(UnknownVariable("DeleteSet", getID(), realPatch, matlIndex,
			   "Cannot find delete set on patch", __FILE__, __LINE__));
   }
  d_lock.readUnlock();
   return iter->second;
}

map<const VarLabel*, ParticleVariableBase*>* 
OnDemandDataWarehouse::getNewParticleState(int matlIndex, const Patch* patch)
{
  d_lock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
  psetAddDBType::key_type key(matlIndex, realPatch);
  psetAddDBType::iterator iter = d_addsetDB.find(key);
  if(iter == d_addsetDB.end()){
    d_lock.readUnlock();
    return 0;
  }
  d_lock.readUnlock();
  return iter->second;
}



bool
OnDemandDataWarehouse::haveParticleSubset(int matlIndex, const Patch* patch,
                                          IntVector low /* = (0,0,0) */,
                                          IntVector high /* = (0,0,0) */)
{
  d_lock.readLock();

  if (low == high && high == IntVector(0,0,0)) {
    low = patch->getLowIndex();
    high = patch->getHighIndex();
  }

   psetDBType::key_type key(patch->getRealPatch(), matlIndex, low, high, getID());
   psetDBType::iterator iter = d_psetDB.find(key);
   if (iter != d_psetDB.end()) {
     d_lock.readUnlock();
     return true;
   }
   
   // if not found, look for an encompassing particle subset
   for (iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++) {
     const PSPatchMatlGhost& pmg = iter->first;
     if (pmg.patch_ == patch && pmg.matl_ == matlIndex &&
         pmg.dwid_ == getID() && 
         low.x() >= pmg.low_.x() && low.y() >= pmg.low_.y() && low.z() >= pmg.low_.z() &&
         high.x() <= pmg.high_.x() && high.y() <= pmg.high_.y() && high.z() <= pmg.high_.z())
       break;
   }
   
  d_lock.readUnlock();
  return iter != d_psetDB.end();
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch,
                                         Ghost::GhostType gtype,
                                         int numGhostCells,
                                         const VarLabel* pos_var)
{
  IntVector lowIndex, highIndex;
  patch->computeVariableExtents(Patch::CellBased, pos_var->getBoundaryLayer(),
				gtype, numGhostCells, lowIndex, highIndex);
  if(gtype == Ghost::None || (lowIndex == patch->getLowIndex() && highIndex == patch->getHighIndex())) {
    return getParticleSubset(matlIndex, patch);
  }

  Patch::selectType neighbors;
  patch->getLevel()->selectPatches(lowIndex, highIndex, neighbors);
  
  Box box = patch->getLevel()->getBox(lowIndex, highIndex);
  
  particleIndex totalParticles = 0;
  vector<ParticleVariableBase*> neighborvars;
  vector<ParticleSubset*> subsets;
  
  for(int i=0;i<(int)neighbors.size();i++){
    const Patch* neighbor = neighbors[i];
    const Patch* realNeighbor = neighbor->getRealPatch();
    if(neighbor){
      IntVector newLow = Max(lowIndex, neighbor->getLowIndex());
      IntVector newHigh = Min(highIndex, neighbor->getHighIndex());

      Box adjustedBox = box;
      if (neighbor->isVirtual()) {
        // rather than offsetting each point of pos_var's data,
        // just adjust the box to compare it with.
        Vector offset = neighbor->getVirtualOffsetVector();
        IntVector cellOffset = neighbor->getVirtualOffset();
        adjustedBox = Box(box.lower() - offset,
                          box.upper() - offset);
        newLow -= cellOffset;
        newHigh -= cellOffset;
      }
      ParticleSubset* pset;

      pset = getParticleSubset(matlIndex, neighbor, newLow, newHigh);
      constParticleVariable<Point> pos;

      get(pos, pos_var, pset);

      particleIndex sizeHint = realNeighbor == patch? pset->numParticles():0;
      ParticleSubset* subset = 
        scinew ParticleSubset(pset->getParticleSet(), false, -1, 0, sizeHint);
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if(adjustedBox.contains(pos[idx])) {
          subset->addParticle(idx);
        }
      }

      //      cout << d_myworld->myrank() << " Adding " << subset->numParticles() << " particles from patch " << neighbor->getID() << endl;
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
                                                    lowIndex, highIndex,
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
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
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

  //cout << d_myworld->myrank() << " get: " << *pset <<endl;
  if((pset->getLow() == patch->getLowIndex() && pset->getHigh() == patch->getHighIndex()) ||
     pset->getNeighbors().size() == 0){
    get(constVar, label, matlIndex, patch);
  }
  else {
   d_lock.readLock();
    checkGetAccess(label, matlIndex, patch);
    ParticleVariableBase* var = constVar.cloneType();

    const vector<const Patch*>& neighbors = pset->getNeighbors();
    const vector<ParticleSubset*>& neighbor_subsets = pset->getNeighborSubsets();

    vector<ParticleVariableBase*> neighborvars(neighbors.size());
    for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor=neighbors[i];
      if(!d_particleDB.exists(label, matlIndex, neighbors[i]))
	SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor, matlIndex,
			      neighbor == patch?"on patch":"on neighbor", __FILE__, __LINE__));
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
   
   if(pset->getLow() == patch->getLowIndex() && pset->getHigh() == patch->getHighIndex()){
     if(!d_particleDB.exists(label, matlIndex, patch))
       SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
     d_particleDB.get(label, matlIndex, patch, var);
   } else {
     SCI_THROW(InternalError("getParticleVariable should not be used with ghost cells", __FILE__, __LINE__));
   }
  d_lock.readUnlock();
}

ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable(const VarLabel* label,
                                           ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   if(pset->getLow() == patch->getLowIndex() && pset->getHigh() == patch->getHighIndex()){
     return getParticleVariable(label, matlIndex, patch);
   } else {
     SCI_THROW(InternalError("getParticleVariable should not be used with ghost cells", __FILE__, __LINE__));
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
     SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
   var = d_particleDB.get(label, matlIndex, patch);

  d_lock.readUnlock();
   return var;
}

void
OnDemandDataWarehouse::allocateTemporary(ParticleVariableBase& var,
                                         ParticleSubset* pset)
{  
  //TAU_PROFILE("allocateTemporary()", "OnDemand.cc", TAU_USER);
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
    SCI_THROW(InternalError("Particle variable already exists: " +
			    label->getName(), __FILE__, __LINE__));
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
   
   const Patch* patch = pset->getPatch();
   if(pset->getLow() != patch->getLowIndex() || pset->getHigh() != patch->getHighIndex())
     SCI_THROW(InternalError("ParticleVariable cannot use put with ghost cells", __FILE__, __LINE__));
   int matlIndex = pset->getMatlIndex();

  dbg << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";

  d_lock.writeLock();   
  checkPutAccess(label, matlIndex, patch, replace);
   
  // Error checking
  if(!replace && d_particleDB.exists(label, matlIndex, patch)) {
    ostringstream error_msg;
    error_msg << "Variable already exists: " << label->getName()
	      << " on patch " << patch->getID();
    SCI_THROW(InternalError(error_msg.str(), __FILE__, __LINE__));
  }

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
 //checkModifyAccess(label, matlIndex, patch);
  getGridVar<Patch::NodeBased>(var, d_ncDB, label, matlIndex, patch,
                               Ghost::None, 0);
 d_lock.readUnlock();  
}

void
OnDemandDataWarehouse::
allocateTemporary(NCVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar<Patch::NodeBased>(var, patch, gtype, numGhostCells,
					     boundaryLayer);
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
  putGridVar<Patch::NodeBased>(*dynamic_cast<NCVariableBase*>(var.clone()), d_ncDB, label, matlIndex, patch,
                               replace);
}

void
OnDemandDataWarehouse::get(PerPatchBase& var, const VarLabel* label,
                           int matlIndex, const Patch* patch)
{
  //checkGetAccess(label);
  d_lock.readLock();
  if(!d_perpatchDB.exists(label, matlIndex, patch))
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			   "perpatch data", __FILE__, __LINE__));
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
     SCI_THROW(InternalError("PerPatch variable already exists: "+label->getName(), __FILE__, __LINE__));

   // Put it in the database
   d_perpatchDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void OnDemandDataWarehouse::
allocateTemporary(CCVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar<Patch::CellBased>(var, patch, gtype, numGhostCells,
					     boundaryLayer);
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
OnDemandDataWarehouse::getRegion(constNCVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  NCVariableBase* var = constVar.cloneType();
  var->allocate(low, high);

  Patch::selectType patches;
  level->selectPatches(low, high, patches);
  
  d_lock.readLock();
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    if(!d_ncDB.exists(label, matlIndex, patch->getRealPatch()))
      SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
    NCVariableBase* tmpVar = constVar.cloneType();
    d_ncDB.get(label, matlIndex, patch, *tmpVar);
    IntVector l(Max(patch->getLowIndex(Patch::NodeBased, label->getBoundaryLayer()), low));
    IntVector h(Min(patch->getHighIndex(Patch::NodeBased, label->getBoundaryLayer()), high));

    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    
    var->copyPatch(tmpVar, l, h);
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }
  IntVector diff(high-low);
  ASSERTEQ(diff.x()*diff.y()*diff.z(), totalCells);
  d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constCCVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  CCVariableBase* var = constVar.cloneType();
  var->allocate(low, high);

  Patch::selectType patches;
  level->selectPatches(low, high, patches);
  
  d_lock.readLock();
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    if(!d_ccDB.exists(label, matlIndex, patch->getRealPatch()))
      SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
    CCVariableBase* tmpVar = constVar.cloneType();
    d_ccDB.get(label, matlIndex, patch, *tmpVar);
    IntVector l(Max(patch->getLowIndex(Patch::CellBased, label->getBoundaryLayer()), low));
    IntVector h(Min(patch->getHighIndex(Patch::CellBased, label->getBoundaryLayer()), high));
    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    var->copyPatch(tmpVar, l, h);
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }
  IntVector diff(high-low);
  ASSERTEQ(diff.x()*diff.y()*diff.z(), totalCells);
  d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constSFCXVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  SFCXVariableBase* var = constVar.cloneType();
  var->allocate(low, high);

  Patch::selectType patches;
  level->selectPatches(low - IntVector(1,0,0), high + IntVector(1,0,0), patches);
  
  d_lock.readLock();
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    if(!d_sfcxDB.exists(label, matlIndex, patch->getRealPatch()))
      continue;
    SFCXVariableBase* tmpVar = constVar.cloneType();
    d_sfcxDB.get(label, matlIndex, patch, *tmpVar);
    IntVector l(Max(patch->getLowIndex(Patch::XFaceBased, label->getBoundaryLayer()), low));
    IntVector h(Min(patch->getHighIndex(Patch::XFaceBased, label->getBoundaryLayer()), high));
    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    var->copyPatch(tmpVar, l, h);
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }
  IntVector diff(high-low);
  ASSERTEQ(diff.x()*diff.y()*diff.z(), totalCells);
  d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constSFCYVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  SFCYVariableBase* var = constVar.cloneType();
  var->allocate(low, high);

  Patch::selectType patches;
  level->selectPatches(low - IntVector(0,1,0), high + IntVector(0,1,0), patches);
  
  d_lock.readLock();
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    if(!d_sfcyDB.exists(label, matlIndex, patch->getRealPatch()))
      continue;
    SFCYVariableBase* tmpVar = constVar.cloneType();
    d_sfcyDB.get(label, matlIndex, patch, *tmpVar);
    IntVector l(Max(patch->getLowIndex(Patch::YFaceBased, label->getBoundaryLayer()), low));
    IntVector h(Min(patch->getHighIndex(Patch::YFaceBased, label->getBoundaryLayer()), high));
    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    var->copyPatch(tmpVar, l, h);
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }
  IntVector diff(high-low);
  ASSERTEQ(diff.x()*diff.y()*diff.z(), totalCells);
  d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constSFCZVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  SFCZVariableBase* var = constVar.cloneType();
  var->allocate(low, high);

  Patch::selectType patches;
  level->selectPatches(low - IntVector(0,0,1), high + IntVector(0,0,1), patches);
  
  d_lock.readLock();
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    if(!d_sfczDB.exists(label, matlIndex, patch->getRealPatch()))
      continue;
    SFCZVariableBase* tmpVar = constVar.cloneType();
    d_sfczDB.get(label, matlIndex, patch, *tmpVar);
    IntVector l(Max(patch->getLowIndex(Patch::YFaceBased, label->getBoundaryLayer()), low));
    IntVector h(Min(patch->getHighIndex(Patch::YFaceBased, label->getBoundaryLayer()), high));
    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    var->copyPatch(tmpVar, l, h);
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }
  IntVector diff(high-low);
  ASSERTEQ(diff.x()*diff.y()*diff.z(), totalCells);
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
  putGridVar<Patch::CellBased>(*dynamic_cast<CCVariableBase*>(var.clone()), d_ccDB, label, matlIndex, patch,
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
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar<Patch::XFaceBased>(var, patch,
                                              gtype, numGhostCells,
					      boundaryLayer);
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
  putGridVar<Patch::XFaceBased>(*dynamic_cast<SFCXVariableBase*>(var.clone()), d_sfcxDB, label, matlIndex,
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
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar<Patch::YFaceBased>(var, patch,
                                              gtype, numGhostCells,
					      boundaryLayer);
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
  putGridVar<Patch::YFaceBased>(*dynamic_cast<SFCYVariableBase*>(var.clone()), d_sfcyDB, label, matlIndex,
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
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar<Patch::ZFaceBased>(var, patch,
                                              gtype, numGhostCells,
					      boundaryLayer);
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
  putGridVar<Patch::ZFaceBased>(*dynamic_cast<SFCZVariableBase*>(var.clone()), d_sfczDB, label, matlIndex,
                                patch, replace);
}

bool
OnDemandDataWarehouse::exists(const VarLabel* label, const Patch* patch) const
{
  d_lock.readLock();
  ASSERT(patch != 0);
  if( d_ncDB.exists(label, patch) || 
      d_ccDB.exists(label, patch) ||
      d_sfcxDB.exists(label, patch) ||
      d_sfcyDB.exists(label, patch) ||
      d_sfczDB.exists(label, patch) ||
      d_particleDB.exists(label, patch) ){
    d_lock.readUnlock();
    return true;
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
   else {
     const Level* level = patch?patch->getLevel():0;
     if(d_reductionDB.exists(label, matlIndex, level))
       var = d_reductionDB.get(label, matlIndex, level);
     if(d_soleDB.exists(label, matlIndex, level))
       var = d_soleDB.get(label, matlIndex, level);
   }
   IntVector l, h;
   if(patch)
     // save with the boundary layer, otherwise restarting from the DataArchive
     // won't work.
     patch->computeVariableExtents(label->typeDescription()->getType(),
				   label->getBoundaryLayer(), Ghost::None, 0,
				   l, h);
   else
     l=h=IntVector(-1,-1,-1);
   if (var == NULL) {
     print();
     SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "on emit", __FILE__, __LINE__));
   }
   var->emit(oc, l, h, label->getCompressionMode());
   
  d_lock.readUnlock();
}

void OnDemandDataWarehouse::print(ostream& intout, const VarLabel* label,
				  const Level* level, int matlIndex /* = -1 */)
{
  d_lock.readLock();

  try {
    checkGetAccess(label, matlIndex, 0); 
    ReductionVariableBase* var = d_reductionDB.get(label, matlIndex, level);
    var->print(intout);
  } catch (UnknownVariable) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			  "on emit reduction", __FILE__, __LINE__));
  }
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::deleteParticles(ParticleSubset* delset)
{
 d_lock.writeLock();
  int matlIndex = delset->getMatlIndex();
  Patch* patch = (Patch*) delset->getPatch();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;

  psetDBType::key_type key(patch, matlIndex, realPatch->getLowIndex(), realPatch->getHighIndex(), getID());
  psetDBType::iterator iter = d_delsetDB.find(key);
  ParticleSubset* currentDelset;
  if(iter != d_delsetDB.end()) {
    //    SCI_THROW(InternalError("deleteParticles called twice for patch", __FILE__, __LINE__));
    // Concatenate the delsets into the delset that already exists in the DB.
    currentDelset = iter->second;
    for (ParticleSubset::iterator d=delset->begin(); d != delset->end(); d++)
      currentDelset->addParticle(*d);
    d_delsetDB[key]=currentDelset;
    //currentDelset->addReference();
    delete delset;
  } else {
    d_delsetDB[key]=delset;
    delset->addReference();
  }
  d_lock.writeUnlock();
}


void
OnDemandDataWarehouse::addParticles(const Patch* patch, int matlIndex,
				    map<const VarLabel*, ParticleVariableBase*>* addedState)
{
 d_lock.writeLock();
  psetAddDBType::key_type key(matlIndex, patch);
  psetAddDBType::iterator iter = d_addsetDB.find(key);
  if(iter  != d_addsetDB.end()) 
    // SCI_THROW(InternalError("addParticles called twice for patch", __FILE__, __LINE__));
    cerr << "addParticles called twice for patch" << endl;
  
  else
    d_addsetDB[key]=addedState;
  
 d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::decrementScrubCount(const VarLabel* var, int matlIndex,
					   const Patch* patch)
{
  d_lock.writeLock();

  switch(var->typeDescription()->getType()){
  case TypeDescription::NCVariable:
    d_ncDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::CCVariable:
    d_ccDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::SFCXVariable:
    d_sfcxDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::SFCYVariable:
    d_sfcyDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::SFCZVariable:
    d_sfczDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::ParticleVariable:
    d_particleDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::PerPatch:
    d_perpatchDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::SoleVariable:
    d_soleDB.decrementScrubCount(var,matlIndex,patch->getLevel());
    break;
  case TypeDescription::ReductionVariable:
    SCI_THROW(InternalError("decrementScrubCount called for reduction variable: "+var->getName(), __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("decrementScrubCount for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
  }
  d_lock.writeUnlock();
}

DataWarehouse::ScrubMode
OnDemandDataWarehouse::setScrubbing(ScrubMode scrubMode)
{
  ScrubMode oldmode = d_scrubMode;
  d_scrubMode = scrubMode;
  return oldmode;
}

void
OnDemandDataWarehouse::setScrubCount(const VarLabel* var, int matlIndex,
				     const Patch* patch, int count)
{
  d_lock.writeLock();
  switch(var->typeDescription()->getType()){
  case TypeDescription::NCVariable:
    d_ncDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::CCVariable:
    d_ccDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::SFCXVariable:
    d_sfcxDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::SFCYVariable:
    d_sfcyDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::SFCZVariable:
    d_sfczDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::ParticleVariable:
    d_particleDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::PerPatch:
    d_perpatchDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::SoleVariable:
    d_soleDB.setScrubCount(var, matlIndex, patch->getLevel(), count);
    break;
  case TypeDescription::ReductionVariable:
    // Reductions are not scrubbed
    SCI_THROW(InternalError("setScrubCount called for reduction variable: "+var->getName(), __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("setScrubCount for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
  }
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::scrub(const VarLabel* var, int matlIndex,
			     const Patch* patch)
{
  d_lock.writeLock();
  switch(var->typeDescription()->getType()){
  case TypeDescription::NCVariable:
    d_ncDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::CCVariable:
    d_ccDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::SFCXVariable:
    d_sfcxDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::SFCYVariable:
    d_sfcyDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::SFCZVariable:
    d_sfczDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::ParticleVariable:
    d_particleDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::PerPatch:
    d_perpatchDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::SoleVariable:
    d_soleDB.scrub(var, matlIndex, patch->getLevel());
    break;
  case TypeDescription::ReductionVariable:
    // Reductions are not scrubbed
    SCI_THROW(InternalError("scrub called for reduction variable: "+var->getName(), __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("scrub for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
  }
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::initializeScrubs(int dwid, 
	const map<VarLabelMatlDW<Patch>, int>& scrubcounts)
{
  d_lock.writeLock();
  d_ncDB.initializeScrubs(dwid, scrubcounts);
  d_ccDB.initializeScrubs(dwid, scrubcounts);
  d_sfcxDB.initializeScrubs(dwid, scrubcounts);
  d_sfcyDB.initializeScrubs(dwid, scrubcounts);
  d_sfczDB.initializeScrubs(dwid, scrubcounts);
  d_particleDB.initializeScrubs(dwid, scrubcounts);
  d_perpatchDB.initializeScrubs(dwid, scrubcounts);
  d_lock.writeUnlock();
}

void
OnDemandDataWarehouse::initializeScrubs(int dwid, 
      const map<VarLabelMatlDW<Level>, int>& scrubcounts)
{
  d_lock.writeLock();
  d_soleDB.initializeScrubs(dwid, scrubcounts);
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
  ASSERTEQ(basis,Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));  

  if(!db.exists(label, matlIndex, patch))
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
  if(patch->isVirtual()){
    db.get(label, matlIndex, patch->getRealPatch(), var);
    var.offsetGrid(patch->getVirtualOffset());
  } else {
    db.get(label, matlIndex, patch, var);
  }
  
  IntVector low = patch->getLowIndex(basis, label->getBoundaryLayer());
  IntVector high = patch->getHighIndex(basis, label->getBoundaryLayer());

  // The data should have been put in the database,
  // windowed with this low and high.
  ASSERTEQ(var.getLow(), low);
  ASSERTEQ(var.getHigh(), high);
  
  if (gtype == Ghost::None) {
    if(numGhostCells != 0)
      SCI_THROW(InternalError("Ghost cells specified with type: None!\n", __FILE__, __LINE__));
    // if this assertion fails, then it is having problems getting the
    // correct window of the data.
    USE_IF_ASSERTS_ON(bool no_realloc =) var.rewindow(low, high);
    ASSERT(no_realloc);
  }
  else {
    IntVector dn = high - low;
    long total = dn.x()*dn.y()*dn.z();
    
    Patch::selectType neighbors;
    IntVector lowIndex, highIndex;
    patch->computeVariableExtents(basis, label->getBoundaryLayer(),
				  gtype, numGhostCells,
                                  neighbors, lowIndex, highIndex);
    if (!var.rewindow(lowIndex, highIndex)) {
      // reallocation needed
      // Ignore this if this is the initialization dw in its old state.
      // The reason for this is that during initialization it doesn't
      // know what ghost cells will be required of it for the next timestep.
      // (This will be an issue whenever the taskgraph changes to require
      // more ghost cells from the old datawarehouse).
      static bool warned = false;
      bool ignore = d_isInitializationDW && d_finalized;
      if (!ignore && !warned ) {
	warned = true;
        static ProgressiveWarning rw("Warning: Reallocation needed for ghost region you requested.\nThis means the data you get back will be a copy of what's in the DW", 5, warn);
        if (rw.invoke()) {
          // print out this message if the ProgressiveWarning does
          ostringstream errmsg;
          errmsg << d_myworld->myrank() << " This occurrence for " << label->getName(); 
          if (patch)
            errmsg << " on patch " << patch->getID();
          errmsg << " for material " << matlIndex;

          errmsg << "You may ignore this under normal circumstances";
          warn << errmsg.str() << '\n';
        }
      }
    }
    
    for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      if(neighbor && (neighbor != patch)){
	if(!db.exists(label, matlIndex, neighbor)) {
	  SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor,
				    matlIndex, neighbor == patch?
				    "on patch":"on neighbor", __FILE__, __LINE__));
        }

        VariableBase* srcvar = dynamic_cast<VariableBase*>(var.cloneType());
	db.get(label, matlIndex, neighbor, *srcvar);
	if(neighbor->isVirtual())
	  srcvar->offsetGrid(neighbor->getVirtualOffset());
	
	IntVector low = Max(lowIndex, srcvar->getLow());
	IntVector high= Min(highIndex, srcvar->getHigh());
	
	if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	    || ( high.z() < low.z() ) ) {
	  //SCI_THROW(InternalError("Patch doesn't overlap?", __FILE__, __LINE__));
        }

	var.copyPatch(srcvar, low, high);
	dn = high-low;
	total+=dn.x()*dn.y()*dn.z();
	delete srcvar;
      }
    }
    
    //dn = highIndex - lowIndex;
    //long wanted = dn.x()*dn.y()*dn.z();
    //ASSERTEQ(wanted, total);
  }
}

template <Patch::VariableBasis basis, class VariableBase>
void OnDemandDataWarehouse::
allocateTemporaryGridVar(VariableBase& var, 
                         const Patch* patch,
                         Ghost::GhostType gtype, int numGhostCells,
			 const IntVector& boundaryLayer)
{
  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
			 numGhostCells, lowOffset, highOffset);
  patch->computeExtents(basis, boundaryLayer, lowOffset, highOffset,
			lowIndex, highIndex);

#ifdef WAYNE_DEBUG
  IntVector diff = highIndex - lowIndex;
  int allocSize = diff.x() * diff.y() * diff.z();
  totalGridAlloc += allocSize;
  cerr << "Allocate temporary: " << lowIndex << " - " << highIndex << " = " << allocSize << "\n";
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

#if 0
  if (!hasRunningTask()) {
    SCI_THROW(InternalError("OnDemandDataWarehouse::AllocateAndPutGridVar can only be used when the dw has a running task associated with it.", __FILE__, __LINE__));
  }
#endif

  checkPutAccess(label, matlIndex, patch, false);  
  bool exists = db.exists(label, matlIndex, patch);

  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
			 numGhostCells, lowOffset, highOffset);
  patch->computeExtents(basis, label->getBoundaryLayer(),
			lowOffset, highOffset, lowIndex, highIndex);
  
  if (exists) {
    // it had been allocated and put as part of the superpatch of
    // another patch
    db.get(label, matlIndex, patch, var);
    
    // The var's window should be the size of the patch or smaller than it.
    ASSERTEQ(Min(var.getLow(), lowIndex), lowIndex);
    ASSERTEQ(Max(var.getHigh(), highIndex), highIndex);
    
    if (var.getLow() != patch->getLowIndex(basis, label->getBoundaryLayer()) ||
        var.getHigh() != patch->getHighIndex(basis, label->getBoundaryLayer()) ||
        var.getBasePointer() == 0 /* place holder for ghost patch */) {
      // It wasn't allocated as part of another patch's superpatch;
      // it existed as ghost patch of another patch.. so we have no
      // choice but to blow it away and replace it.
      db.put(label, matlIndex, patch, 0, true);

      // this is just a tricky way to uninitialize var
      VariableBase* tmpVar = dynamic_cast<VariableBase*>(var.cloneType());
      var.copyPointer(*tmpVar);
      delete tmpVar;
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
  //totalGridAlloc += allocSize;
  cerr << d_myworld->myrank() << " Allocate " << label->getName() << ", matl " << matlIndex 
       << ": " << superLowIndex << " - " << superHighIndex << " = " 
       << allocSize << "\n";  
#endif
  
  var.allocate(superLowIndex, superHighIndex);

  Patch::selectType encompassedPatches;
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
  
  Patch::selectType::iterator iter = encompassedPatches.begin();    
  for (; iter != encompassedPatches.end(); ++iter) {
    const Patch* patchGroupMember = *iter;
    VariableBase* clone = dynamic_cast<VariableBase*>(var.clone());
    IntVector groupMemberLowIndex = patchGroupMember->getLowIndex(basis, label->getBoundaryLayer());
    IntVector groupMemberHighIndex = patchGroupMember->getHighIndex(basis, label->getBoundaryLayer());
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


void OnDemandDataWarehouse::transferFrom(DataWarehouse* from,
					 const VarLabel* var,
					 const PatchSubset* patches,
					 const MaterialSubset* matls,
                                         bool replace /*=false*/,
                                         const PatchSubset* newPatches /*=0*/)
{
  OnDemandDataWarehouse* fromDW = dynamic_cast<OnDemandDataWarehouse*>(from);
  ASSERT(fromDW != 0);
  ASSERT(!d_finalized);
  d_lock.writeLock();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    const Patch* copyPatch = (newPatches ? newPatches->get(p) : patch);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);
      checkPutAccess(var, matl, patch, replace);      
      switch(var->typeDescription()->getType()){
      case TypeDescription::NCVariable:
	{
	  if(!fromDW->d_ncDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
	  NCVariableBase* v = dynamic_cast<NCVariableBase*>(fromDW->d_ncDB.get(var, matl, patch)->clone());
	  d_ncDB.put(var, matl, copyPatch, v, replace);
	}
	break;
      case TypeDescription::CCVariable:
	{
	  if(!fromDW->d_ccDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
	  CCVariableBase* v = dynamic_cast<CCVariableBase*>(fromDW->d_ccDB.get(var, matl, patch)->clone());
	  d_ccDB.put(var, matl, copyPatch, v, replace);
	}
	break;
      case TypeDescription::SFCXVariable:
	{
	  if(!fromDW->d_sfcxDB.exists(var, matl, patch)) {
            fromDW->d_sfcxDB.print(cout);
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
          }
	  SFCXVariableBase* v = dynamic_cast<SFCXVariableBase*>(fromDW->d_sfcxDB.get(var, matl, patch)->clone());
	  d_sfcxDB.put(var, matl, copyPatch, v, replace);
	}
	break;
      case TypeDescription::SFCYVariable:
	{
	  if(!fromDW->d_sfcyDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
	  SFCYVariableBase* v = dynamic_cast<SFCYVariableBase*>(fromDW->d_sfcyDB.get(var, matl, patch)->clone());
	  d_sfcyDB.put(var, matl, copyPatch, v, replace);
	}
	break;
      case TypeDescription::SFCZVariable:
	{
	  if(!fromDW->d_sfczDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
	  SFCZVariableBase* v = dynamic_cast<SFCZVariableBase*>(fromDW->d_sfczDB.get(var, matl, patch)->clone());
	  d_sfczDB.put(var, matl, copyPatch, v, replace);
	}
	break;
      case TypeDescription::ParticleVariable:
	{
	  if(!fromDW->d_particleDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));

          // or else the readLock in haveParticleSubset will hang
          d_lock.writeUnlock();
          ParticleSubset* subset;
          if (!haveParticleSubset(matl, copyPatch)) {
            ParticleSubset* oldsubset = fromDW->getParticleSubset(matl, patch);
            subset = createParticleSubset(oldsubset->numParticles(), matl, copyPatch);
          }
          else
            subset = getParticleSubset(matl, copyPatch);
          d_lock.writeLock();
	  ParticleVariableBase* v = fromDW->d_particleDB.get(var, matl, patch);
          if (patch == copyPatch)
            d_particleDB.put(var, matl, copyPatch, v->clone(), replace);
          else {
            ParticleVariableBase* newv = v->cloneType();
            newv->copyPointer(*v);
            newv->setParticleSubset(subset);
            d_particleDB.put(var, matl, copyPatch, newv, replace);
          }
	}
	break;
      case TypeDescription::PerPatch:
	{
	  if(!fromDW->d_perpatchDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
	  PerPatchBase* v = fromDW->d_perpatchDB.get(var, matl, patch);
	  d_perpatchDB.put(var, matl, copyPatch, v->clone(), replace);
	}
	break;
      case TypeDescription::ReductionVariable:
	SCI_THROW(InternalError("transferFrom doesn't work for reduction variable: "+var->getName(), __FILE__, __LINE__));
	break;
      case TypeDescription::SoleVariable:
	SCI_THROW(InternalError("transferFrom doesn't work for sole variable: "+var->getName(), __FILE__, __LINE__));
	break;
      default:
	SCI_THROW(InternalError("Unknown variable type in transferFrom: "+var->getName(), __FILE__, __LINE__));
      }
    }
  }
  d_lock.writeUnlock();
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
     SCI_THROW(InternalError("put: grid variable already exists: " +
			     label->getName(), __FILE__, __LINE__));

   // Put it in the database
   IntVector low = patch->getLowIndex(basis, label->getBoundaryLayer());
   IntVector high = patch->getHighIndex(basis, label->getBoundaryLayer());
   if (Min(var.getLow(), low) != var.getLow() ||
       Max(var.getHigh(), high) != var.getHigh()) {
     ostringstream msg_str;
     msg_str << "put: Variable's window (" << var.getLow() << " - " << var.getHigh() << ") must encompass patches extent (" << low << " - " << high;
     SCI_THROW(InternalError(msg_str.str(), __FILE__, __LINE__));
   }
   USE_IF_ASSERTS_ON(bool no_realloc =) var.rewindow(low, high);
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
OnDemandDataWarehouse::checkGetAccess(const VarLabel* /*label*/,
                                      int /*matlIndex*/, const Patch* /*patch*/,
                                      Ghost::GhostType /*gtype*/,int /*numGhostCells*/)
{
#if 0
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
      
      map<VarLabelMatl<Patch>, AccessInfo>::iterator findIter;
      findIter = runningTaskAccesses.find(VarLabelMatl<Patch>(label, matlIndex,
							      patch));

      if (!hasGetAccess(runningTask, label, matlIndex, patch, lowOffset,
			highOffset)) {

	// If it was accessed by the current task already, then it should
	// have get access (i.e. if you put it in, you should be able to get it
	// right back out).
	if (findIter != runningTaskAccesses.end() &&
	    lowOffset == IntVector(0, 0, 0) && 
	    highOffset == IntVector(0, 0, 0)){
	  // allow non ghost cell get if any access (get, put, or modify) is allowed
	  //cout << "allowing non-ghost cell access\n";
	  return;
        }
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
      } else {
        // access granted
        if (findIter == runningTaskAccesses.end()) {
          AccessInfo& accessInfo =
            runningTaskAccesses[VarLabelMatl<Patch>(label, matlIndex, patch)];
          accessInfo.accessType = GetAccess;
          accessInfo.encompassOffsets(lowOffset, highOffset);

          int ID = 0;
          if( patch ) ID = patch->getID();
          string varname = "noname";
          if( label ) varname = label->getName();

          dbg << "Task running is: " << runningTask->getName() << "\n";
          dbg << "data " << varname << " on patch " << ID
               << " and matl: " << matlIndex << " has been gotten\n";
        } else {
          findIter->second.encompassOffsets(lowOffset, highOffset);
        }
      }
    }
  }
#endif
#endif
}

inline void
OnDemandDataWarehouse::checkPutAccess(const VarLabel* /*label*/, int /*matlIndex*/,
                                      const Patch* /*patch*/, bool /*replace*/)
{
#if 0
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
	  SCI_THROW(DependencyException(runningTask, label, matlIndex,
				    patch, has, needs, __FILE__, __LINE__));
	}
      }
      else {
        runningTaskAccesses[VarLabelMatl<Patch>(label, matlIndex, patch)].accessType = replace ? ModifyAccess : PutAccess;
      }
    }
  }
#endif
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

void OnDemandDataWarehouse::pushRunningTask(const Task* task,
					    vector<OnDemandDataWarehouseP>* dws)
{
  ASSERT(task);
 d_lock.writeLock();    
  d_runningTasks[Thread::self()].push_back(RunningTaskInfo(task, dws));
 d_lock.writeUnlock();
}

void OnDemandDataWarehouse::popRunningTask()
{
 d_lock.writeLock();
  list<RunningTaskInfo>& runningTasks = d_runningTasks[Thread::self()];
  runningTasks.pop_back();
  if (runningTasks.empty()) {
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

DataWarehouse*
OnDemandDataWarehouse::getOtherDataWarehouse(Task::WhichDW dw)
{
  RunningTaskInfo* info = getCurrentTaskInfo();
  int dwindex = info->d_task->mapDataWarehouse(dw);
  DataWarehouse* result = (*info->dws)[dwindex].get_rep();
  ASSERT(result != 0);
  return result;
}

void OnDemandDataWarehouse::checkTasksAccesses(const PatchSubset* /*patches*/,
                                               const MaterialSubset* /*matls*/)
{
#if 0
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
#if 0
    if ((isFinalized() && dep->dw == Task::NewDW) ||
        (!isFinalized() && dep->dw == Task::OldDW))
      continue;
#endif
    
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
 
    if( currentTask->getName() == "Relocate::relocateParticles" ){
      continue;
    }
    
    for (int m = 0; m < matls->size(); m++) {
      int matl = matls->get(m);
      
      for (int p = 0; p < patches->size(); p++) {
        const Patch* patch = patches->get(p);
        
        VarLabelMatl<Patch> key(label, matl, patch);
        map<VarLabelMatl<Patch>, AccessInfo>::iterator find_iter;
        find_iter = currentTaskAccesses.find(key);
        if (find_iter == currentTaskAccesses.end() ||
            (*find_iter).second.accessType != accessType) {
          if ((*find_iter).second.accessType == ModifyAccess && 
              accessType == GetAccess) { // If you require with ghost cells
            continue;                    // and modify, it can get into this
          }                              // situation.

#if 1
// THIS OLD HACK PERHAPS CAN GO AWAY
          if( lowOffset == IntVector(0, 0, 0) && 
              highOffset == IntVector(0, 0, 0)){
            // In checkGetAccess(), this case does not record the fact
            // that the var was accessed, so don't throw exception here.
            continue;
          }
#endif
          if( find_iter == currentTaskAccesses.end() ) {
            cout << "Error: did not find " << label->getName() << "\n";
            cout << "Mtl: " << m << ", Patch: " << *patch << "\n";
          } else {
            cout << "Error: accessType is not GetAccess for " 
                 << label->getName() << "\n";
          }
          cout << "For Task:\n";
          currentTask->displayAll( cout );

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

          SCI_THROW(DependencyException(currentTask, label, matl, patch,
					has, needs, __FILE__, __LINE__));
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

          SCI_THROW(DependencyException(currentTask, label, matl, patch,
					has, needs, __FILE__, __LINE__));
        }
      }
    }
  }
}


// For timestep abort/restart
bool OnDemandDataWarehouse::timestepAborted()
{
  return aborted;
}

bool OnDemandDataWarehouse::timestepRestarted()
{
  return restart;
}

void OnDemandDataWarehouse::abortTimestep()
{
  aborted=true;
}

void OnDemandDataWarehouse::restartTimestep()
{
  restart=true;
}

namespace Uintah {
  int getDB_ID(const Patch* patch) {
    if(!patch)
      return -1;
    //ASSERT(!patch->isVirtual());
    if(patch->isVirtual())
      patch = patch->getRealPatch();
    return patch->getID();
  }
  int getDB_ID(const Level* level) {
    if(!level)
      return -1;
    return level->getIndex();
  }
}

void OnDemandDataWarehouse::getVarLabelMatlLevelTriples( vector<VarLabelMatl<Level> >& vars ) const
{
  d_reductionDB.getVarLabelMatlTriples(vars);
  //  d_soleDB.getVarLabelMatlTriples(vars);
}

void OnDemandDataWarehouse::print()
{
  d_ncDB.print(cout);
  d_ccDB.print(cout);
  d_sfcxDB.print(cout);
  d_sfcyDB.print(cout);
  d_sfczDB.print(cout);
  d_particleDB.print(cout);  
  d_soleDB.print(cout);  
}
