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
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
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
static DebugStream particles("DWParticles", false);
extern DebugStream mpidbg;

static Mutex ssLock( "send state lock" );

// we want a particle message to have a unique tag per patch/matl/batch/dest.
// we only have 32K message tags, so this will have to do.
//   We need this because the possibility exists (particularly with DLB) of 
//   two messages with the same tag being sent from the same processor.  Even
//   if these messages are sent to different processors, they can get crossed in the mail
//   or one can overwrite the other.
#define PARTICLESET_TAG	0x4000|batch->messageTag
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

   d_varDB.cleanForeign();
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

  if(!d_levelDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			      "on reduction", __FILE__, __LINE__));
  }
  d_levelDB.get(label, matlIndex, level, var);

  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::get(SoleVariableBase& var,
			   const VarLabel* label, const Level* level,
			   int matlIndex /*= -1*/)
{
  d_lock.readLock();
  
  checkGetAccess(label, matlIndex, 0);

  if(!d_levelDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			      "on sole", __FILE__, __LINE__));
  }
  d_levelDB.get(label, matlIndex, level, var);

  d_lock.readUnlock();
}

bool
OnDemandDataWarehouse::exists(const VarLabel* label, int matlIndex,
                              const Patch* patch) const
{
  d_lock.readLock();
  
   if( d_varDB.exists(label, matlIndex, patch) ||
       d_levelDB.exists(label, matlIndex, patch->getLevel()) ) {

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

      if(!d_varDB.exists(label, matlIndex, patch)) {
        print();
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI", __FILE__, __LINE__));
      }
      ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>(d_varDB.get(label, matlIndex, patch));

      int dest = batch->toTasks.front()->getAssignedResourceIndex();
      ASSERTRANGE(dest, 0, d_myworld->size());

      ssLock.lock();  // Dd: ??
      // in a case where there is a sendset with a different ghost configuration
      // than we want, there can be problems with dynamic load balancing, when patch
      // used to be on this processor and now we only want ghost data from patch.  So
      // check if dest previously sent us (during this timestep) this entire patch, 
      // if so, just use that sendset
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
	ASSERT((PARTICLESET_TAG) <= (*maxtag));
#endif
        ASSERT(batch->messageTag >= 0);
        

        int tag = PARTICLESET_TAG;
        particles << d_myworld->myrank() << " " << getID() << " Sending PARTICLE message " << tag << ", to " << dest << ", patch " << patch->getID() << ", matl " << matlIndex << ", length: " << 1 << "(" << numParticles << ") " << sendset->getLow() << " " << sendset->getHigh() << " GI " << patch->getGridIndex() << " tag " << batch->messageTag << endl; cerrLock.unlock();
        MPI_Bsend(&numParticles, 1, MPI_INT, dest, tag, d_myworld->getComm());
        mpidbg << d_myworld->myrank() << " Done Sending PARTICLE message " << tag << ", to " << dest << ", patch " << patch->getID() << ", matl " << matlIndex << ", length: " << 1 << "(" << numParticles << ")\n"; cerrLock.unlock();
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
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
    {
      if(!d_varDB.exists(label, matlIndex, patch)) {
        cout << d_myworld->myrank() << "  Needed by " << *dep << " on task " << *dep->toTasks.front() << endl;
        SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
          "in sendMPI", __FILE__, __LINE__));
      }
      GridVariable* var;
      var = dynamic_cast<GridVariable*>(d_varDB.get(label, matlIndex, patch));
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
        int tag = PARTICLESET_TAG;

        particles << d_myworld->myrank() << " " << getID() << " Posting PARTICLES receive for message " << tag << " from " << from << ", patch " << patch->getID() << ", matl " << matlIndex << ", length=" << 1 << " " << low << " " << high << " GI " << patch->getGridIndex() << " tag " << batch->messageTag << "\n";      
        MPI_Recv(&numParticles, 1, MPI_INT, from, tag, d_myworld->getComm(), &status);
        particles << d_myworld->myrank() << "   recved " << numParticles << " particles " << endl;
        
        // sometime we have to force a receive to match a send.
        // in these cases just ignore this new subset
        ParticleSubset* psubset;
        if (!old_dw->haveParticleSubset(matlIndex, patch, low, high)) {
          psubset = old_dw->createParticleSubset(numParticles, matlIndex, patch, low, high);
        }
        else {
          psubset = old_dw->getParticleSubset(matlIndex,patch,low,high);
          if (numParticles != psubset->numParticles()) {
            cout << d_myworld->myrank() << " BAD: pset " << psubset->getLow() << " " << psubset->getHigh() << " " << psubset->numParticles() << " particles, src: " << from << " range: " << low << " " << high << " " << numParticles << " particles " << " patch " << patch->getLowIndex() << " " << patch->getHighIndex() << " " << matlIndex << endl;
            //old_dw->printParticleSubsets();
            ASSERTEQ(numParticles, psubset->numParticles());
          }
        }
        ParticleSubset* recvset = new ParticleSubset(psubset->getParticleSet(),
                                                     true, matlIndex, patch, 
                                                     low, high, 0);
        rs.add_sendset(recvset, from, patch, matlIndex, low, high, old_dw->d_generation);
      }
      ParticleSubset* pset = old_dw->getParticleSubset(matlIndex,patch,low, high);

      Variable* v = label->typeDescription()->createInstance();
      ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>(v);
      ASSERT(var != 0);
      var->allocate(pset);
      var->setForeign();
      if(pset->numParticles() > 0){
        var->getMPIBuffer(buffer, pset);
      }

      d_lock.writeLock();
      d_varDB.put(label, matlIndex, patch, var, true);
      d_lock.writeUnlock();
    }
    break;
  case TypeDescription::NCVariable:
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
    recvMPIGridVar(buffer, dep, label, matlIndex, patch);
    break;
  default:
    SCI_THROW(InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch), __FILE__, __LINE__));
  } // end switch( label->getType() );
} // end recvMPI()

void
OnDemandDataWarehouse::recvMPIGridVar(BufferInfo& buffer,
                                       const DetailedDep* dep,
                                       const VarLabel* label, int matlIndex,
                                       const Patch* patch)
{
  d_lock.readLock();
  GridVariable* var = 0;
  if (d_varDB.exists(label, matlIndex, patch)) {
    var = dynamic_cast<GridVariable*>(d_varDB.get(label, matlIndex, patch));
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
    GridVariable* v = dynamic_cast<GridVariable*>(label->typeDescription()->createInstance());
    v->allocate(dep->low, dep->high);
    var = dynamic_cast<GridVariable*>(v);
    var->setForeign();
    d_lock.writeLock();
    d_varDB.put(label, matlIndex, patch, var, true);
    d_lock.writeUnlock();
  }

  ASSERTEQ(Min(var->getLow(), dep->low), var->getLow());
  ASSERTEQ(Max(var->getHigh(), dep->high), var->getHigh());

  var->getMPIBuffer(buffer, dep->low, dep->high);
}

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
    if (d_levelDB.exists(label, matlIndex, level))
      var = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
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
      d_levelDB.put(label, matlIndex, level, var, true);
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
      var = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
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

  mpidbg << d_myworld->myrank() << " allreduce, done " << label->getName() << " level " << (level?level->getID():-1) << endl;
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
      var = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
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
  if (!d_levelDB.exists(label, matlIndex, level))
    d_levelDB.put(label, matlIndex, level, var.clone(), false);
  else {
    ReductionVariableBase* foundVar
      = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
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
  d_levelDB.put(label, matlIndex, level, var.clone(), true);
   
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
  if (!d_levelDB.exists(label, matlIndex, level))
    d_levelDB.put(label, matlIndex, level, var.clone(), false);
  
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
  d_levelDB.put(label, matlIndex, level, var.clone(), true);
   
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
  const Patch* realPatch = patch->getRealPatch();

   psetDBType::key_type key(realPatch, matlIndex, low, high, getID());
   psetDBType::iterator iter = d_psetDB.find(key);
   if (iter != d_psetDB.end()) {
     d_lock.readUnlock();
     return true;
   }
   
   // if not found, look for an encompassing particle subset
   for (iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++) {
     const PSPatchMatlGhost& pmg = iter->first;
     if (pmg.patch_ == realPatch && pmg.matl_ == matlIndex &&
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

  return getParticleSubset(matlIndex, lowIndex, highIndex, patch->getLevel(), patch, pos_var);
}

ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, IntVector lowIndex, IntVector highIndex, 
                                         const Level* level, const Patch* relPatch, const VarLabel* pos_var)
{
  // relPatch can be NULL if trying to get a particle subset for an arbitrary spot on the level
  Patch::selectType neighbors;
  level->selectPatches(lowIndex, highIndex, neighbors);
  Box box = level->getBox(lowIndex, highIndex);
  
  particleIndex totalParticles = 0;
  vector<ParticleVariableBase*> neighborvars;
  vector<ParticleSubset*> subsets;
  vector<const Patch*> vneighbors;
  
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

      if (relPatch && relPatch != neighbor) {
        relPatch->cullIntersection(Patch::CellBased, IntVector(0,0,0), neighbor, newLow, newHigh);
        if (newLow == newHigh) {
          continue;
        }
      }
      pset = getParticleSubset(matlIndex, neighbor, newLow, newHigh);

      constParticleVariable<Point> pos;

      get(pos, pos_var, pset);

      particleIndex sizeHint = (relPatch && realNeighbor == relPatch) ? pset->numParticles():0;
      ParticleSubset* subset = 
        scinew ParticleSubset(pset->getParticleSet(), false, -1, 0, sizeHint);
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        if(adjustedBox.contains(pos[idx])) {
          subset->addParticle(idx);
        }
      }

      totalParticles+=subset->numParticles();
      subsets.push_back(subset);
      vneighbors.push_back(neighbors[i]);
    }
  }
  ParticleSet* newset = scinew ParticleSet(totalParticles);
  ParticleSubset* newsubset = scinew ParticleSubset(newset, true,
                                                    matlIndex, relPatch,
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

  if(!d_varDB.exists(label, matlIndex, patch)) {
    print();
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
  }
  constVar = *dynamic_cast<ParticleVariableBase*>(d_varDB.get(label, matlIndex, patch));
   
  d_lock.readUnlock();
}

void
OnDemandDataWarehouse::get(constParticleVariableBase& constVar,
                           const VarLabel* label,
                           ParticleSubset* pset)
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();

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
      if(!d_varDB.exists(label, matlIndex, neighbors[i]))
	SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor, matlIndex,
			      neighbor == patch?"on patch":"on neighbor", __FILE__, __LINE__));
      neighborvars[i] = var->cloneType();

      d_varDB.get(label, matlIndex, neighbors[i], *neighborvars[i]);
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
     if(!d_varDB.exists(label, matlIndex, patch))
       SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
     d_varDB.get(label, matlIndex, patch, var);
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
   
   if(!d_varDB.exists(label, matlIndex, patch))
     SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
   var = dynamic_cast<ParticleVariableBase*>(d_varDB.get(label, matlIndex, patch));

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
  if(d_varDB.exists(label, matlIndex, patch))
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
  if(!replace && d_varDB.exists(label, matlIndex, patch)) {
    ostringstream error_msg;
    error_msg << "Variable already exists: " << label->getName()
	      << " on patch " << patch->getID();
    SCI_THROW(InternalError(error_msg.str(), __FILE__, __LINE__));
  }

  // Put it in the database
  d_varDB.put(label, matlIndex, patch, var.clone(), replace);
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
  getGridVar(*var, label, matlIndex, patch, gtype, numGhostCells);
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
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void
OnDemandDataWarehouse::
allocateTemporary(NCVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void OnDemandDataWarehouse::
allocateAndPut(NCVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(NCVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);
}

void
OnDemandDataWarehouse::get(PerPatchBase& var, const VarLabel* label,
                           int matlIndex, const Patch* patch)
{
  //checkGetAccess(label);
  d_lock.readLock();
  if(!d_varDB.exists(label, matlIndex, patch))
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			   "perpatch data", __FILE__, __LINE__));
  d_varDB.get(label, matlIndex, patch, var);
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
   if(!replace && d_varDB.exists(label, matlIndex, patch))
     SCI_THROW(InternalError("PerPatch variable already exists: "+label->getName(), __FILE__, __LINE__));

   // Put it in the database
   d_varDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void OnDemandDataWarehouse::
allocateTemporary(CCVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void OnDemandDataWarehouse::
allocateAndPut(CCVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
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
  getGridVar(*var, label, matlIndex, patch, gtype, numGhostCells);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
OnDemandDataWarehouse::getRegionGridVar(GridVariable& var,
                                        const VarLabel* label,
                                        int matlIndex, const Level* level,
                                        const IntVector& low, const IntVector& high,
                                        bool useBoundaryCells /*=true*/)
{
  var.allocate(low, high);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  IntVector adjustment = IntVector(1,1,1);
  if (basis == Patch::XFaceBased) adjustment = IntVector(1,0,0);
  else if (basis == Patch::YFaceBased) adjustment = IntVector(0,1,0);
  else if (basis == Patch::ZFaceBased) adjustment = IntVector(0,0,1);

  Patch::selectType patches;
  
  // if in AMR and one node intersects from another patch and that patch is missing
  // ignore the error instead of throwing an exception
  // (should only be for node-based vars)
  vector<const Patch*> missing_patches;

  // make sure we grab all the patches, sometimes we might call only with an extra cell region, which
  // selectPatches doesn't detect
  IntVector tmpLow(low-adjustment);
  IntVector tmpHigh(high+adjustment);
  level->selectPatches(tmpLow, tmpHigh, patches);
  
  d_lock.readLock();
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    IntVector l, h;

    // the caller should determine whether or not he wants extra cells.
    // It will matter in AMR cases with corner-aligned patches
    if (useBoundaryCells) {
      l = Max(patch->getLowIndex(basis, label->getBoundaryLayer()), low);
      h = Min(patch->getHighIndex(basis, label->getBoundaryLayer()), high);
    }
    else {
      l = Max(patch->getInteriorLowIndex(basis), low);
      h = Min(patch->getInteriorHighIndex(basis), high);
    }
    if (l.x() >= h.x() || l.y() >= h.y() || l.z() >= h.z())
      continue;
    if(!d_varDB.exists(label, matlIndex, patch->getRealPatch())) {
      missing_patches.push_back(patch->getRealPatch());
      continue;
    }
    GridVariable* tmpVar = var.cloneType();
    d_varDB.get(label, matlIndex, patch, *tmpVar);

    if (Max(l, tmpVar->getLow()) != l || Min(h, tmpVar->getHigh()) != h) {
      // just like a "missing patch": got data on this patch, but it either corresponds to a different
      // region or is incomplete"
      missing_patches.push_back(patch->getRealPatch());
      continue;
    }
    
    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    try {
      var.copyPatch(tmpVar, l, h);
    } catch (InternalError& e) {
      cout << " Bad range: " << low << " " << high << ", patch intersection: " << l << " " << h 
           << " actual patch " << patch->getInteriorLowIndex(basis) << " " << patch->getInteriorHighIndex(basis) 
           << " var range: "  << tmpVar->getLow() << " " << tmpVar->getHigh() << endl;
      throw e;
    }
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }
  IntVector diff(high-low);

  if (diff.x()*diff.y()*diff.z() > totalCells && missing_patches.size() > 0) {
    cout << d_myworld->myrank() << "  Unknown Variable " << *label << " matl " << matlIndex << " for patch(es): ";
    for (unsigned i = 0; i < missing_patches.size(); i++) 
      cout << missing_patches[i]->getID() << " ";
    cout << endl;
    throw InternalError("Missing patches in getRegion", __FILE__, __LINE__);
  }

  ASSERT(diff.x()*diff.y()*diff.z() <= totalCells);
  if (diff.x()*diff.y()*diff.z() != totalCells) {
    static ProgressiveWarning warn("GetRegion Warning", 100);
    warn.invoke();
  }

  d_lock.readUnlock();
 
}

void
OnDemandDataWarehouse::getRegion(constNCVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high,
                                 bool useBoundaryCells /*=true*/)
{
  GridVariable* var = constVar.cloneType();
  getRegionGridVar(*var, label, matlIndex, level, low, high, useBoundaryCells);
  constVar = *dynamic_cast<NCVariableBase*>(var);
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constCCVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high,
                                 bool useBoundaryCells /*=true*/)
{
  GridVariable* var = constVar.cloneType();
  getRegionGridVar(*var, label, matlIndex, level, low, high, useBoundaryCells);
  constVar = *dynamic_cast<CCVariableBase*>(var);
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constSFCXVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high,
                                 bool useBoundaryCells /*=true*/)
{
  GridVariable* var = constVar.cloneType();
  getRegionGridVar(*var, label, matlIndex, level, low, high, useBoundaryCells);
  constVar = *dynamic_cast<SFCXVariableBase*>(var);
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constSFCYVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high,
                                 bool useBoundaryCells /*=true*/)
{
  GridVariable* var = constVar.cloneType();
  getRegionGridVar(*var, label, matlIndex, level, low, high, useBoundaryCells);
  constVar = *dynamic_cast<SFCYVariableBase*>(var);
  delete var;
}

void
OnDemandDataWarehouse::getRegion(constSFCZVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high,
                                 bool useBoundaryCells /*=true*/)
{
  GridVariable* var = constVar.cloneType();
  getRegionGridVar(*var, label, matlIndex, level, low, high, useBoundaryCells);
  constVar = *dynamic_cast<SFCZVariableBase*>(var);
  delete var;
}

void
OnDemandDataWarehouse::getModifiable(CCVariableBase& var,
                                     const VarLabel* label,
                                     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void
OnDemandDataWarehouse::put(CCVariableBase& var, const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);  
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
  getGridVar(*var, label, matlIndex, patch, gtype, numGhostCells);
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
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void OnDemandDataWarehouse::
allocateTemporary(SFCXVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void OnDemandDataWarehouse::
allocateAndPut(SFCXVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCXVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);
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
  getGridVar(*var, label, matlIndex, patch, gtype, numGhostCells);
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
  getGridVar(var, label, matlIndex, patch,
                                Ghost::None, 0);
 d_lock.readUnlock();  
}

void OnDemandDataWarehouse::
allocateTemporary(SFCYVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void OnDemandDataWarehouse::
allocateAndPut(SFCYVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCYVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);  
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
  getGridVar(*var, label, matlIndex, patch, gtype, numGhostCells);
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
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void OnDemandDataWarehouse::
allocateTemporary(SFCZVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void OnDemandDataWarehouse::
allocateAndPut(SFCZVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
OnDemandDataWarehouse::put(SFCZVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);
}

void OnDemandDataWarehouse::emit(OutputContext& oc, const VarLabel* label,
                                 int matlIndex, const Patch* patch)
{
  d_lock.readLock();
   checkGetAccess(label, matlIndex, patch);

   Variable* var = NULL;
   if(patch && d_varDB.exists(label, matlIndex, patch))
      var = d_varDB.get(label, matlIndex, patch);
   else {
     const Level* level = patch?patch->getLevel():0;
     if(d_levelDB.exists(label, matlIndex, level))
       var = d_levelDB.get(label, matlIndex, level);
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
    ReductionVariableBase* var = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
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
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
  case TypeDescription::ParticleVariable:
  case TypeDescription::PerPatch:
    d_varDB.decrementScrubCount(var, matlIndex, patch);
    break;
  case TypeDescription::SoleVariable:
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
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
  case TypeDescription::ParticleVariable:
  case TypeDescription::PerPatch:
    d_varDB.setScrubCount(var, matlIndex, patch, count);
    break;
  case TypeDescription::SoleVariable:
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
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
  case TypeDescription::ParticleVariable:
  case TypeDescription::PerPatch:
    d_varDB.scrub(var, matlIndex, patch);
    break;
  case TypeDescription::SoleVariable:
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
	const FastHashTable<ScrubItem>* scrubcounts)
{
  d_lock.writeLock();
  d_varDB.initializeScrubs(dwid, scrubcounts);
  d_lock.writeUnlock();
}

void OnDemandDataWarehouse::
getGridVar(GridVariable& var, const VarLabel* label, int matlIndex, const Patch* patch,
           Ghost::GhostType gtype, int numGhostCells)
{
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);
  ASSERTEQ(basis,Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));  

  if(!d_varDB.exists(label, matlIndex, patch)) {
    print();
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
  }
  if(patch->isVirtual()){
    d_varDB.get(label, matlIndex, patch->getRealPatch(), var);
    var.offsetGrid(patch->getVirtualOffset());
  } else {
    d_varDB.get(label, matlIndex, patch, var);
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
        static ProgressiveWarning rw("Warning: Reallocation needed for ghost region you requested.\nThis means the data you get back will be a copy of what's in the DW", 5);
        if (rw.invoke()) {
          // print out this message if the ProgressiveWarning does
          ostringstream errmsg;
          /*          errmsg << d_myworld->myrank() << " This occurrence for " << label->getName(); 
          if (patch)
            errmsg << " on patch " << patch->getID();
          errmsg << " for material " << matlIndex;

          errmsg << "You may ignore this under normal circumstances";
          warn << errmsg.str() << '\n';
          */
        }
      }
    }

    for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      if(neighbor && (neighbor != patch)){
        IntVector low = Max(neighbor->getLowIndex(basis, label->getBoundaryLayer()), lowIndex);
        IntVector high = Min(neighbor->getHighIndex(basis, label->getBoundaryLayer()), highIndex);

        if (patch->getLevel()->getIndex() > 0 && patch != neighbor) {
          patch->cullIntersection(basis, label->getBoundaryLayer(), neighbor, low, high);
        }
        if (low == high) {
          continue;
        }

	if(!d_varDB.exists(label, matlIndex, neighbor)) {
	  SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor,
				    matlIndex, neighbor == patch?
				    "on patch":"on neighbor", __FILE__, __LINE__));
        }

        GridVariable* srcvar = var.cloneType();
	d_varDB.get(label, matlIndex, neighbor, *srcvar);
	if(neighbor->isVirtual())
	  srcvar->offsetGrid(neighbor->getVirtualOffset());
	
	if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	    || ( high.z() < low.z() ) ) {
	  //SCI_THROW(InternalError("Patch doesn't overlap?", __FILE__, __LINE__));
        }

        try {
          var.copyPatch(srcvar, low, high);
        } catch (InternalError& e) {
          cout << "  Can't copy patch " << neighbor->getID() << " for var " << *label << " " << low << " " << high << endl;
          cout << e.message() << endl;
          throw;
        }
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

void OnDemandDataWarehouse::
allocateTemporaryGridVar(GridVariable& var, const Patch* patch,
                         Ghost::GhostType gtype, int numGhostCells,
			 const IntVector& boundaryLayer)
{
  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), false);
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

void OnDemandDataWarehouse::
allocateAndPutGridVar(GridVariable& var, const VarLabel* label, int matlIndex, const Patch* patch,
                      Ghost::GhostType gtype, int numGhostCells)
{
  if (d_finalized)
    cerr << "  DW " << getID() << " finalized!\n";
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
  bool exists = d_varDB.exists(label, matlIndex, patch);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
			 numGhostCells, lowOffset, highOffset);
  patch->computeExtents(basis, label->getBoundaryLayer(),
			lowOffset, highOffset, lowIndex, highIndex);
  
  if (exists) {
    // it had been allocated and put as part of the superpatch of
    // another patch
    d_varDB.get(label, matlIndex, patch, var);
    
    // The var's window should be the size of the patch or smaller than it.
    ASSERTEQ(Min(var.getLow(), lowIndex), lowIndex);
    ASSERTEQ(Max(var.getHigh(), highIndex), highIndex);
    
    if (var.getLow() != patch->getLowIndex(basis, label->getBoundaryLayer()) ||
        var.getHigh() != patch->getHighIndex(basis, label->getBoundaryLayer()) ||
        var.getBasePointer() == 0 /* place holder for ghost patch */) {
      // It wasn't allocated as part of another patch's superpatch;
      // it existed as ghost patch of another patch.. so we have no
      // choice but to blow it away and replace it.
      d_varDB.put(label, matlIndex, patch, 0, true);

      // this is just a tricky way to uninitialize var
      Variable* tmpVar = dynamic_cast<Variable*>(var.cloneType());
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

#if SCI_ASSERTION_LEVEL >= 3


  // check for dead portions of a variable (variable space that isn't covered by any patch).  
  // This will happen with L-shaped patch configs and ngc > extra cells.  
  // find all dead space and mark it with a bogus value.
  IntVector extraCells = patch->getLevel()->getExtraCells();

  // use the max extra cells for now
  int ec = Max(Max(extraCells[0], extraCells[1]), extraCells[2]);
  if (1) { // numGhostCells > ec) { (numGhostCells is 0, query it from the superLowIndex...
    deque<Box> b1, b2, difference;
    b1.push_back(Box(Point(superLowIndex(0), superLowIndex(1), superLowIndex(2)), 
                     Point(superHighIndex(0), superHighIndex(1), superHighIndex(2))));
    for (unsigned i = 0; i < (*superPatchGroup).size(); i++) {
      const Patch* p = (*superPatchGroup)[i];
      IntVector low = p->getLowIndex(basis, label->getBoundaryLayer());
      IntVector high = p->getHighIndex(basis, label->getBoundaryLayer());
      b2.push_back(Box(Point(low(0), low(1), low(2)), Point(high(0), high(1), high(2))));
    }
    difference = Box::difference(b1, b2);

#if 0
    if (difference.size() > 0) {
      cout << "Box difference: " << superLowIndex << " " << superHighIndex << " with patches " << endl;
      for (unsigned i = 0; i < (*superPatchGroup).size(); i++) {
        const Patch* p = (*superPatchGroup)[i];
        cout << p->getLowIndex(basis, label->getBoundaryLayer()) << " " << p->getHighIndex(basis, label->getBoundaryLayer()) << endl;
      }

      for (unsigned i = 0; i < difference.size(); i++) {
        cout << difference[i].lower() << " " << difference[i].upper() << endl;
      }
    }
#endif
    // get more efficient way of doing this...
    for (unsigned i = 0; i < difference.size(); i++) {
      Box b = difference[i];
      IntVector low(b.lower()(0), b.lower()(1), b.lower()(2));
      IntVector high(b.upper()(0), b.upper()(1), b.upper()(2));
      if (NCVariable<double>* typedVar = dynamic_cast<NCVariable<double>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (NCVariable<Vector>* typedVar = dynamic_cast<NCVariable<Vector>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (CCVariable<double>* typedVar = dynamic_cast<CCVariable<double>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (CCVariable<Vector>* typedVar = dynamic_cast<CCVariable<Vector>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (SFCXVariable<double>* typedVar = dynamic_cast<SFCXVariable<double>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (SFCXVariable<Vector>* typedVar = dynamic_cast<SFCXVariable<Vector>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (SFCYVariable<double>* typedVar = dynamic_cast<SFCYVariable<double>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (SFCYVariable<Vector>* typedVar = dynamic_cast<SFCYVariable<Vector>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (SFCZVariable<double>* typedVar = dynamic_cast<SFCZVariable<double>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
      else if (SFCZVariable<Vector>* typedVar = dynamic_cast<SFCZVariable<Vector>*>(&var)) {
        for (CellIterator iter(low, high); !iter.done(); iter++)
          (*typedVar)[*iter] = -5.555555e256;
      }
    }
  }
#endif 

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
    GridVariable* clone = var.clone();
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
      exists = d_varDB.exists(label, matlIndex, patchGroupMember);
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
      
      GridVariable* existingGhostVar =
        dynamic_cast<GridVariable*>(d_varDB.get(label, matlIndex, patchGroupMember));
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
        d_varDB.put(label, matlIndex, patchGroupMember, clone, true);
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
      d_varDB.put(label, matlIndex, patchGroupMember, clone, false);
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
      case TypeDescription::CCVariable:
      case TypeDescription::SFCXVariable:
      case TypeDescription::SFCYVariable:
      case TypeDescription::SFCZVariable:
	{
	  if(!fromDW->d_varDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
	  GridVariable* v = dynamic_cast<GridVariable*>(fromDW->d_varDB.get(var, matl, patch))->clone();
	  d_varDB.put(var, matl, copyPatch, v, replace);
	}
	break;
      case TypeDescription::ParticleVariable:
	{
	  if(!fromDW->d_varDB.exists(var, matl, patch))
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
	  ParticleVariableBase* v = dynamic_cast<ParticleVariableBase*>(fromDW->d_varDB.get(var, matl, patch));
          if (patch == copyPatch)
            d_varDB.put(var, matl, copyPatch, v->clone(), replace);
          else {
            ParticleVariableBase* newv = v->cloneType();
            newv->copyPointer(*v);
            newv->setParticleSubset(subset);
            d_varDB.put(var, matl, copyPatch, newv, replace);
          }
	}
	break;
      case TypeDescription::PerPatch:
	{
	  if(!fromDW->d_varDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom", __FILE__, __LINE__));
	  PerPatchBase* v = dynamic_cast<PerPatchBase*>(fromDW->d_varDB.get(var, matl, patch));
	  d_varDB.put(var, matl, copyPatch, v->clone(), replace);
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

void OnDemandDataWarehouse::
putGridVar(GridVariable& var,  const VarLabel* label, int matlIndex, const Patch* patch,
           bool replace /* = false */)
{
  ASSERT(!d_finalized);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);
  ASSERTEQ(basis, Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));    
 d_lock.writeLock();  

 checkPutAccess(label, matlIndex, patch, replace);

#if DAV_DEBUG
  cerr << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";
#endif
   // Error checking
   if(!replace && d_varDB.exists(label, matlIndex, patch))
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
   d_varDB.put(label, matlIndex, patch, &var, true);
  d_lock.writeUnlock();
}

void OnDemandDataWarehouse::logMemoryUse(ostream& out, unsigned long& total,
                                         const std::string& tag)
{
  int dwid=d_generation;
  d_varDB.logMemoryUse(out, total, tag, dwid);

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

void OnDemandDataWarehouse::getVarLabelMatlLevelTriples( vector<VarLabelMatl<Level> >& vars ) const
{
  d_levelDB.getVarLabelMatlTriples(vars);
}

void OnDemandDataWarehouse::print()
{
  cout << "  VARS in DW " << getID() << endl;
  d_varDB.print(cout);
  d_levelDB.print(cout);  
}
