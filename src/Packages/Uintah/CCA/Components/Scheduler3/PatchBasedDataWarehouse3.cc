
#include <TauProfilerForSCIRun.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>

#include <Packages/Uintah/CCA/Components/Scheduler3/PatchBasedDataWarehouse3.h>
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

static DebugStream dbg( "PatchBasedDataWarehouse3", false );
static DebugStream warn( "PatchBasedDataWarehouse3_warn", true );

static Mutex ssLock( "send state lock" );

#define PARTICLESET_TAG		0x4000
#define DAV_DEBUG 0

PatchBasedDataWarehouse3::PatchBasedDataWarehouse3(const ProcessorGroup* myworld,
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

PatchBasedDataWarehouse3::~PatchBasedDataWarehouse3()
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

bool PatchBasedDataWarehouse3::isFinalized() const
{
   return d_finalized;
}

void PatchBasedDataWarehouse3::finalize()
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

void PatchBasedDataWarehouse3::unfinalize()
{
  // this is for processes that need to make small modifications to the DW
  // after it has been finalized.
  d_finalized=false;
}

void PatchBasedDataWarehouse3::refinalize()
{
  d_finalized=true;
}


void
PatchBasedDataWarehouse3::put(Variable* var, const VarLabel* label,
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
     SCI_THROW(InternalError("Unknown Variable type"));
}

void PatchBasedDataWarehouse3::
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
     SCI_THROW(InternalError("PatchBasedDataWarehouse3::allocateAndPutGridVar: Not a grid variable type"));
}

void
PatchBasedDataWarehouse3::get(ReductionVariableBase& var,
			   const VarLabel* label, const Level* level,
			   int matlIndex /*= -1*/)
{
  d_lock.readLock();
  
  checkGetAccess(label, matlIndex, 0);

  if(!d_levelDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			      "on reduction"));
  }
  d_levelDB.get(label, matlIndex, level, var);

  d_lock.readUnlock();
}

void
PatchBasedDataWarehouse3::get(SoleVariableBase& var,
			   const VarLabel* label, const Level* level,
			   int matlIndex /*= -1*/)
{
  d_lock.readLock();
  
  checkGetAccess(label, matlIndex, 0);

  if(!d_levelDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			      "on sole"));
  }
  d_levelDB.get(label, matlIndex, level, var);

  d_lock.readUnlock();
}

bool
PatchBasedDataWarehouse3::exists(const VarLabel* label, int matlIndex,
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
PatchBasedDataWarehouse3::sendMPI(SendState& ss, SendState& rs, DependencyBatch* batch,
                               const VarLabel* pos_var,
                               BufferInfo& buffer,
                               PatchBasedDataWarehouse3* old_dw,
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
  Ghost::GhostType gt = dep->req->gtype;
  int ngc = dep->req->numGhostCells;

  IntVector range = dep->high - dep->low;
  if (range.x() > ngc && range.y() > ngc && range.z() > ngc) {
    //brydbg << d_myworld->myrank() << " sending entire patch of data\n";
    gt = Ghost::None;
    ngc = 0;
  }

 d_lock.readLock();
  switch(label->typeDescription()->getType()){
  case TypeDescription::ParticleVariable:
    {
      if(!d_varDB.exists(label, matlIndex, patch))
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			      "in sendMPI"));
      ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>(d_varDB.get(label, matlIndex, patch));

      int dest = batch->toTasks.front()->getAssignedResourceIndex();
      ASSERTRANGE(dest, 0, d_myworld->size());

      ssLock.lock();  // Dd: ??
      // in a case where there is a sendset with a different ghost configuration
      // than we want, there can be problems with dynamic load balancing, when patch
      // used to be on this processor and now we only want ghost data from patch.  So
      // check if dest previously sent us this entire patch, if so, just use that sendset
      ParticleSubset* sendset = rs.find_sendset(dest, patch, matlIndex, Ghost::None, 0, old_dw->d_generation);
      if (sendset) {
        fflush(stdout);
        sendset = old_dw->getParticleSubset(matlIndex, patch, Ghost::None, 0);
        gt = Ghost::None;
        ngc = 0;
      }
      else
        sendset = ss.find_sendset(dest, patch, matlIndex, gt, ngc, old_dw->d_generation);
      ssLock.unlock();  // Dd: ??

      if(!sendset){

        dbg << "sendset is NULL\n";

        ParticleSubset* pset = var->getParticleSubset();
        ssLock.lock();  // Dd: ??
        sendset = scinew ParticleSubset(pset->getParticleSet(),
                                        false, matlIndex, patch, gt, ngc, 0);
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
#if SCI_ASSERTION_LEVEL >= 1
	int* maxtag, found;
	MPI_Attr_get(d_myworld->getComm(), MPI_TAG_UB, &maxtag, &found);
	ASSERT(found);
	ASSERT((PARTICLESET_TAG|batch->messageTag) <= (*maxtag));
#endif
        ASSERT(batch->messageTag >= 0);
        
        // dbg << d_myworld->myrank() << " Sending PARTICLE message number " << (PARTICLESET_TAG|batch->messageTag) << ", to " << dest << ", patch " << patch->getID() << ", matl " << matlIndex << ", length: " << 1 << "(" << numParticles << ")\n"; cerrLock.unlock();

        MPI_Bsend(&numParticles, 1, MPI_INT, dest,
                  PARTICLESET_TAG|batch->messageTag, d_myworld->getComm());
        ssLock.lock();  // Dd: ??       
        ss.add_sendset(sendset, dest, patch, matlIndex, gt, ngc, old_dw->d_generation);
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
      if(!d_varDB.exists(label, matlIndex, patch))
        SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
          "in sendMPI"));
      GridVariable* var;
      var = dynamic_cast<GridVariable*>(d_varDB.get(label, matlIndex, patch));
      var->getMPIBuffer(buffer, dep->low, dep->high);
      buffer.addSendlist(var->getRefCounted());
    }
    break;
  default:
    SCI_THROW(InternalError("sendMPI not implemented for "+label->getFullName(matlIndex, patch)));
  } // end switch( label->getType() );
 d_lock.readUnlock();  
}

void
PatchBasedDataWarehouse3::recvMPI(SendState& rs, BufferInfo& buffer,
                               DependencyBatch* batch,
                               PatchBasedDataWarehouse3* old_dw,
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

  Ghost::GhostType gt = dep->req->gtype;
  int ngc = dep->req->numGhostCells;

  // this is a special case, based on the circumstances, we have sent
  // the entire patch instead of just the ghost region.
  // This can happen when dynamic load balancing or when 1 patch needs to send
  // to multiple patchs on 1 processor
  
  IntVector range = dep->high - dep->low;
  if (range.x() > ngc && range.y() > ngc && range.z() > ngc) {
    //brydbg << d_myworld->myrank() << " Recving entire patch of data\n";
    gt = Ghost::None;
    ngc = 0;
  }

  switch(label->typeDescription()->getType()){
  case TypeDescription::ParticleVariable:
    {
      // First, get the particle set.  We should already have it
      //      if(!old_dw->haveParticleSubset(matlIndex, patch, gt, ngc)){
      int from=batch->fromTask->getAssignedResourceIndex();

      // if we already have a subset for the entire patch, there's little point 
      // in getting another one (and if we did, it would cause problems - see
      // comment in sendMPI)
      ParticleSubset* recvset;
      if (old_dw->haveParticleSubset(matlIndex, patch, Ghost::None, 0)) {
        recvset = old_dw->getParticleSubset(matlIndex, patch, Ghost::None, 0);
        gt = Ghost::None;
        ngc = 0;
      }
      else
        recvset = rs.find_sendset(from, patch, matlIndex, gt, ngc, old_dw->d_generation);
      
      if(!recvset){
        int numParticles;
        MPI_Status status;
        ASSERT(batch->messageTag >= 0);
	ASSERTRANGE(from, 0, d_myworld->size());
        // dbg << d_myworld->myrank() << " Posting PARTICLES receive for message number " << (PARTICLESET_TAG|batch->messageTag) << " from " << from << ", patch " << patch->getID() << ", matl " << matlIndex << ", length=" << 1 << "\n";      
        MPI_Recv(&numParticles, 1, MPI_INT, from,
                 PARTICLESET_TAG|batch->messageTag, d_myworld->getComm(),
                 &status);
        // dbg << d_myworld->myrank() << " recving " << numParticles << "particles\n";
        
        // sometime we have to force a receive to match a send.
        // in these cases just ignore this new subset
        ParticleSubset* psubset;
        if (!old_dw->haveParticleSubset(matlIndex, patch, gt, ngc)) {
          psubset = old_dw->createParticleSubset(numParticles, matlIndex, patch, gt, ngc);
        }
        else {
          old_dw->printParticleSubsets();
          psubset = old_dw->getParticleSubset(matlIndex,patch,gt,ngc);
          ASSERTEQ(numParticles, psubset->numParticles());
        }
        ParticleSubset* recvset = new ParticleSubset(psubset->getParticleSet(),
                                                     true, matlIndex, patch, 
                                                     gt, ngc, 0);
        rs.add_sendset(recvset, from, patch, matlIndex, gt, ngc, old_dw->d_generation);
      }
      ParticleSubset* pset = old_dw->getParticleSubset(matlIndex,patch,gt,ngc);


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
    SCI_THROW(InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch)));
  } // end switch( label->getType() );
} // end recvMPI()

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // template parameter not used in declaring arguments
#endif  

void
PatchBasedDataWarehouse3::recvMPIGridVar(BufferInfo& buffer,
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

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif  

void
PatchBasedDataWarehouse3::reduceMPI(const VarLabel* label,
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
      //				"on reduceMPI"));
    }
    int sendcount;
    MPI_Datatype senddatatype;
    MPI_Op sendop;
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
                                "on reduceMPI(pass 2)"));
    }
    var->getMPIData(sendbuf, packindex);
  }
  d_lock.readUnlock();

  vector<char> recvbuf(packsize);

  if( dbg.active() ) {
    cerrLock.lock(); dbg << "calling MPI_Allreduce\n";
    cerrLock.unlock();
  }

  dbg << d_myworld->myrank() << " allreduce, buf=" << &sendbuf[0] << ", count=" << count << ", datatype=" << datatype << ", op=" << op << '\n';
  int error = MPI_Allreduce(&sendbuf[0], &recvbuf[0], count, datatype, op,
			    d_myworld->getComm());

  if( dbg.active() ) {
    cerrLock.lock(); dbg << "done with MPI_Allreduce\n";
    cerrLock.unlock();
  }

  if( error ){
    cerrLock.lock();
    cerr << "reduceMPI: MPI_Allreduce error: " << error << "\n";
    cerrLock.unlock();
    SCI_THROW(InternalError("reduceMPI: MPI error"));     
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
				"on reduceMPI(pass 2)"));
    }
    var->putMPIData(recvbuf, unpackindex);
  }
  d_lock.writeUnlock();
  if(matls != inmatls)
    delete matls;
}

void
PatchBasedDataWarehouse3::put(const ReductionVariableBase& var,
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
PatchBasedDataWarehouse3::override(const ReductionVariableBase& var,
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
PatchBasedDataWarehouse3::put(const SoleVariableBase& var,
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
PatchBasedDataWarehouse3::override(const SoleVariableBase& var,
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
PatchBasedDataWarehouse3::createParticleSubset(particleIndex numParticles,
                                            int matlIndex, const Patch* patch,
                                            Ghost::GhostType gt, int numgc)
{
  d_lock.writeLock();
  dbg << d_myworld->myrank() << " DW ID " << getID() << " createParticleSubset: MI: " << matlIndex << " P: " << patch->getID() << " (" << gt << "," << numgc << ")\n";

  ASSERT(!patch->isVirtual());

   ParticleSet* pset = scinew ParticleSet(numParticles);
   ParticleSubset* psubset = 
     scinew ParticleSubset(pset, true, matlIndex, patch, gt, numgc, 0);

   psetDBType::key_type key(patch, matlIndex, gt, numgc);
   if(d_psetDB.find(key) != d_psetDB.end())
     SCI_THROW(InternalError("createParticleSubset called twice for patch"));

   d_psetDB[key]=psubset;
   psubset->addReference();
  d_lock.writeUnlock();
   return psubset;
}

void
PatchBasedDataWarehouse3::saveParticleSubset(ParticleSubset* psubset, 
                                          int matlIndex, const Patch* patch,
                                          Ghost::GhostType gt, int numgc)
{
  ASSERTEQ(psubset->getPatch(), patch);
  ASSERTEQ(psubset->getMatlIndex(), matlIndex);
  ASSERT(!patch->isVirtual());  
  d_lock.writeLock();
  dbg << d_myworld->myrank() << " DW ID " << getID() << " saveParticleSubset: MI: " << matlIndex << " P: " << patch->getID() << " (" << gt << "," << numgc << ")\n";
  psetDBType::key_type key(patch, matlIndex, gt, numgc);
  if(d_psetDB.find(key) != d_psetDB.end())
    SCI_THROW(InternalError("saveParticleSubset called twice for patch"));

  d_psetDB[key]=psubset;
  psubset->addReference();

  d_lock.writeUnlock();
}

void PatchBasedDataWarehouse3::printParticleSubsets()
{
  psetDBType::iterator iter;
  cout << d_myworld->myrank() << " Available psets on DW " << d_generation << ":\n";
  for (iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++) {
    cout << d_myworld->myrank() << " " <<*(iter->second) << endl;
  }
  
}

ParticleSubset*
PatchBasedDataWarehouse3::getParticleSubset(int matlIndex, const Patch* patch,
                                         Ghost::GhostType gt, int numgc)
{
  d_lock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;

  psetDBType::key_type key(realPatch, matlIndex, gt, numgc);
  psetDBType::iterator iter = d_psetDB.find(key);
  if(iter == d_psetDB.end()){
    printParticleSubsets();
    d_lock.readUnlock();
    ostringstream s;
    s << "ParticleSet, ghost: (" << gt << ',' << numgc << ')';
    SCI_THROW(UnknownVariable(s.str().c_str(), getID(), realPatch, matlIndex,
                              "Cannot find particle set on patch"));
  }
  d_lock.readUnlock();
  return iter->second;
}

ParticleSubset*
PatchBasedDataWarehouse3::getDeleteSubset(int matlIndex, const Patch* patch,
                                       Ghost::GhostType gt, int numgc)
{
  d_lock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
   psetDBType::key_type key(realPatch, matlIndex, gt, numgc);
   psetDBType::iterator iter = d_delsetDB.find(key);
   if(iter == d_delsetDB.end()){
     d_lock.readUnlock();
     SCI_THROW(UnknownVariable("DeleteSet", getID(), realPatch, matlIndex,
			   "Cannot find delete set on patch"));
   }
  d_lock.readUnlock();
   return iter->second;
}

map<const VarLabel*, ParticleVariableBase*>* 
PatchBasedDataWarehouse3::getNewParticleState(int matlIndex, const Patch* patch)
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
PatchBasedDataWarehouse3::haveParticleSubset(int matlIndex, const Patch* patch,
                                          Ghost::GhostType gt, int numgc)
{
  d_lock.readLock();
   psetDBType::key_type key(patch->getRealPatch(), matlIndex, gt, numgc);
   psetDBType::iterator iter = d_psetDB.find(key);
  d_lock.readUnlock();
   return !(iter == d_psetDB.end());
}

ParticleSubset*
PatchBasedDataWarehouse3::getParticleSubset(int matlIndex, const Patch* patch,
                                         Ghost::GhostType gtype,
                                         int numGhostCells,
                                         const VarLabel* pos_var)
{
  if(gtype == Ghost::None){
    if(numGhostCells != 0)
      SCI_THROW(InternalError("Ghost cells specified with task type none!\n"));
    return getParticleSubset(matlIndex, patch, gtype, numGhostCells);
  }
  
  Patch::selectType neighbors;
  IntVector lowIndex, highIndex;
  patch->computeVariableExtents(Patch::CellBased, pos_var->getBoundaryLayer(),
				gtype, numGhostCells,
                                neighbors, lowIndex, highIndex);
  Box box = patch->getLevel()->getBox(lowIndex, highIndex);
  
  particleIndex totalParticles = 0;
  vector<ParticleVariableBase*> neighborvars;
  vector<ParticleSubset*> subsets;
  
  for(int i=0;i<(int)neighbors.size();i++){
    const Patch* neighbor = neighbors[i];
    const Patch* realNeighbor = neighbor->getRealPatch();
    if(neighbor){
      Box adjustedBox = box;
      if (neighbor->isVirtual()) {
        // rather than offsetting each point of pos_var's data,
        // just adjust the box to compare it with.
        Vector offset = neighbor->getVirtualOffsetVector();
        adjustedBox = Box(box.lower() - offset,
                          box.upper() - offset);
      }
      ParticleSubset* pset;
      // if neighbor is stored entirely on this dw - multi-patch per proc, or 
      // this is a load balancing timestep and neighbor used to be on this proc
      LoadBalancer* lb = d_scheduler->getLoadBalancer();

      if (lb->getPatchwiseProcessorAssignment(neighbor) == d_myworld->myrank() || 
          //          (lb->getOldProcessorAssignment(pos_var, neighbor, 0) == d_myworld->myrank() &&
          //           haveParticleSubset(matlIndex, neighbor, Ghost::None, 0)) ||
          !haveParticleSubset(matlIndex, neighbor, gtype, numGhostCells))
        pset = getParticleSubset(matlIndex, neighbor, Ghost::None, 0);
      else
        pset = getParticleSubset(matlIndex, neighbor, gtype, numGhostCells);
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

      //cout << d_myworld->myrank() << " GPS 1 "  << *subset << endl;
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
PatchBasedDataWarehouse3::get(constParticleVariableBase& constVar,
                           const VarLabel* label, int matlIndex,
                           const Patch* patch)
{
  d_lock.readLock();

  checkGetAccess(label, matlIndex, patch);

  if(!d_varDB.exists(label, matlIndex, patch))
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex));
  constVar = *dynamic_cast<ParticleVariableBase*>(d_varDB.get(label, matlIndex, patch));
   
  d_lock.readUnlock();
}

void
PatchBasedDataWarehouse3::get(constParticleVariableBase& constVar,
                           const VarLabel* label,
                           ParticleSubset* pset)
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();

  //cout << d_myworld->myrank() << " get: " << *pset <<endl;
  if(pset->getGhostType() == Ghost::None || pset->getNeighbors().size() == 0){
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
      if(!d_varDB.exists(label, matlIndex, neighbors[i]))
	SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor, matlIndex,
			      neighbor == patch?"on patch":"on neighbor"));
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
PatchBasedDataWarehouse3::getModifiable(ParticleVariableBase& var,
                                     const VarLabel* label,
                                     ParticleSubset* pset)
{
  d_lock.readLock();
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();
   checkModifyAccess(label, matlIndex, patch);
   
   if(pset->getGhostType() == Ghost::None){
      if(!d_varDB.exists(label, matlIndex, patch))
	SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex));
      d_varDB.get(label, matlIndex, patch, var);
   } else {
     SCI_THROW(InternalError("getParticleVariable should not be used with ghost cells"));
   }
  d_lock.readUnlock();
}

ParticleVariableBase*
PatchBasedDataWarehouse3::getParticleVariable(const VarLabel* label,
                                           ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   if(pset->getGhostType() == Ghost::None){
     return getParticleVariable(label, matlIndex, patch);
   } else {
     SCI_THROW(InternalError("getParticleVariable should not be used with ghost cells"));
   }
}

ParticleVariableBase*
PatchBasedDataWarehouse3::getParticleVariable(const VarLabel* label,
                                           int matlIndex, const Patch* patch)
{
   ParticleVariableBase* var = 0;  

   // in case the it's a virtual patch -- only deal with real patches
   if (patch != 0) patch = patch->getRealPatch();
   
  d_lock.readLock();
  
   checkModifyAccess(label, matlIndex, patch);
   
   if(!d_varDB.exists(label, matlIndex, patch))
     SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex));
   var = dynamic_cast<ParticleVariableBase*>(d_varDB.get(label, matlIndex, patch));

  d_lock.readUnlock();
   return var;
}

void
PatchBasedDataWarehouse3::allocateTemporary(ParticleVariableBase& var,
                                         ParticleSubset* pset)
{  
  //TAU_PROFILE("allocateTemporary()", "OnDemand.cc", TAU_USER);
  var.allocate(pset);
}

void
PatchBasedDataWarehouse3::allocateAndPut(ParticleVariableBase& var,
                                      const VarLabel* label,
                                      ParticleSubset* pset)
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();
  
  // Error checking
  d_lock.readLock();   
  if(d_varDB.exists(label, matlIndex, patch))
    SCI_THROW(InternalError("Particle variable already exists: " +
			    label->getName()));
  d_lock.readUnlock();
  
  allocateTemporary(var, pset);
  put(var, label);
}

void
PatchBasedDataWarehouse3::put(ParticleVariableBase& var,
                           const VarLabel* label, bool replace /*= false*/)
{
  ASSERT(!d_finalized);  

   ParticleSubset* pset = var.getParticleSubset();
   if(pset->numGhostCells() != 0 || pset->getGhostType() != 0)
     SCI_THROW(InternalError("ParticleVariable cannot use put with ghost cells"));
   const Patch* patch = pset->getPatch();
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
    SCI_THROW(InternalError(error_msg.str()));
  }

  // Put it in the database
  d_varDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void
PatchBasedDataWarehouse3::get(constNCVariableBase& constVar,
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
PatchBasedDataWarehouse3::getModifiable(NCVariableBase& var,
                                     const VarLabel* label,
                                     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
 //checkModifyAccess(label, matlIndex, patch);
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void
PatchBasedDataWarehouse3::
allocateTemporary(NCVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void PatchBasedDataWarehouse3::
allocateAndPut(NCVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
PatchBasedDataWarehouse3::put(NCVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);
}

void
PatchBasedDataWarehouse3::get(PerPatchBase& var, const VarLabel* label,
                           int matlIndex, const Patch* patch)
{
  //checkGetAccess(label);
  d_lock.readLock();
  if(!d_varDB.exists(label, matlIndex, patch))
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
			   "perpatch data"));
  d_varDB.get(label, matlIndex, patch, var);
  d_lock.readUnlock();
}

void
PatchBasedDataWarehouse3::put(PerPatchBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  ASSERT(!d_finalized);  
  //checkPutAccess(label, replace);
  
  d_lock.writeLock();

   // Error checking
   if(!replace && d_varDB.exists(label, matlIndex, patch))
     SCI_THROW(InternalError("PerPatch variable already exists: "+label->getName()));

   // Put it in the database
   d_varDB.put(label, matlIndex, patch, var.clone(), true);
  d_lock.writeUnlock();
}

void PatchBasedDataWarehouse3::
allocateTemporary(CCVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void PatchBasedDataWarehouse3::
allocateAndPut(CCVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
PatchBasedDataWarehouse3::get(constCCVariableBase& constVar,
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
PatchBasedDataWarehouse3::getRegionGridVar(GridVariable& var,
                                           const VarLabel* label,
                                           int matlIndex, const Level* level,
                                           const IntVector& low, const IntVector& high)
{
  var.allocate(low, high);

  Patch::selectType patches;
  level->selectPatches(low, high, patches);
  
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    if(!d_varDB.exists(label, matlIndex, patch->getRealPatch()))
      SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex));
    GridVariable* tmpVar = var.cloneType();
    d_varDB.get(label, matlIndex, patch, *tmpVar);
    IntVector l(Max(patch->getLowIndex(Patch::NodeBased, label->getBoundaryLayer()), low));
    IntVector h(Min(patch->getHighIndex(Patch::NodeBased, label->getBoundaryLayer()), high));
    
    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    var.copyPatch(tmpVar, l, h);
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }
  IntVector diff(high-low);
  ASSERTEQ(diff.x()*diff.y()*diff.z(), totalCells);
}

void
PatchBasedDataWarehouse3::getRegion(constNCVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  NCVariableBase* var = constVar.cloneType();
  
 d_lock.readLock();
  getRegionGridVar(*var, label, matlIndex, level, low, high);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
PatchBasedDataWarehouse3::getRegion(constCCVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  CCVariableBase* var = constVar.cloneType();
  
 d_lock.readLock();
  getRegionGridVar(*var, label, matlIndex, level, low, high);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
PatchBasedDataWarehouse3::getRegion(constSFCXVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  SFCXVariableBase* var = constVar.cloneType();
  
 d_lock.readLock();
  getRegionGridVar(*var, label, matlIndex, level, low, high);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
PatchBasedDataWarehouse3::getRegion(constSFCYVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  SFCYVariableBase* var = constVar.cloneType();
  
 d_lock.readLock();
  getRegionGridVar(*var, label, matlIndex, level, low, high);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
PatchBasedDataWarehouse3::getRegion(constSFCZVariableBase& constVar,
				 const VarLabel* label,
				 int matlIndex, const Level* level,
				 const IntVector& low, const IntVector& high)
{
  SFCZVariableBase* var = constVar.cloneType();
  
 d_lock.readLock();
  getRegionGridVar(*var, label, matlIndex, level, low, high);
 d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}

void
PatchBasedDataWarehouse3::getModifiable(CCVariableBase& var,
                                     const VarLabel* label,
                                     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void
PatchBasedDataWarehouse3::put(CCVariableBase& var, const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);  
}

void
PatchBasedDataWarehouse3::get(constSFCXVariableBase& constVar,
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
PatchBasedDataWarehouse3::getModifiable(SFCXVariableBase& var,
                                     const VarLabel* label,
                                     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void PatchBasedDataWarehouse3::
allocateTemporary(SFCXVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void PatchBasedDataWarehouse3::
allocateAndPut(SFCXVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
PatchBasedDataWarehouse3::put(SFCXVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);
}

void
PatchBasedDataWarehouse3::get(constSFCYVariableBase& constVar,
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
PatchBasedDataWarehouse3::getModifiable(SFCYVariableBase& var,
                                     const VarLabel* label,
                                     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void PatchBasedDataWarehouse3::
allocateTemporary(SFCYVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void PatchBasedDataWarehouse3::
allocateAndPut(SFCYVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
PatchBasedDataWarehouse3::put(SFCYVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);  
}

void
PatchBasedDataWarehouse3::get(constSFCZVariableBase& constVar,
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
PatchBasedDataWarehouse3::getModifiable(SFCZVariableBase& var,
                                     const VarLabel* label,
                                     int matlIndex, const Patch* patch)
{
 d_lock.readLock();  
  checkModifyAccess(label, matlIndex, patch);  
  getGridVar(var, label, matlIndex, patch, Ghost::None, 0);
 d_lock.readUnlock();  
}

void PatchBasedDataWarehouse3::
allocateTemporary(SFCZVariableBase& var, const Patch* patch,
                  Ghost::GhostType gtype, int numGhostCells,
		  const IntVector& boundaryLayer)
{
  allocateTemporaryGridVar(var, patch, gtype, numGhostCells, boundaryLayer);
}

void PatchBasedDataWarehouse3::
allocateAndPut(SFCZVariableBase& var, const VarLabel* label,
               int matlIndex, const Patch* patch,
               Ghost::GhostType gtype, int numGhostCells)
{
  allocateAndPutGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

void
PatchBasedDataWarehouse3::put(SFCZVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, const Patch* patch,
                           bool replace /*= false*/)
{
  putGridVar(*var.clone(), label, matlIndex, patch, replace);
}

void PatchBasedDataWarehouse3::emit(OutputContext& oc, const VarLabel* label,
                                 int matlIndex, const Patch* patch)
{
  d_lock.readLock();
   checkGetAccess(label, matlIndex, patch);

   Variable* var = NULL;
   if(d_varDB.exists(label, matlIndex, patch))
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
     print();
     SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "on emit"));
   }
   var->emit(oc, l, h, label->getCompressionMode());
   
  d_lock.readUnlock();
}

void PatchBasedDataWarehouse3::print(ostream& intout, const VarLabel* label,
				  const Level* level, int matlIndex /* = -1 */)
{
  d_lock.readLock();

  try {
    checkGetAccess(label, matlIndex, 0); 
    ReductionVariableBase* var = 
      dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
    var->print(intout);
  } catch (UnknownVariable) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
			  "on emit reduction"));
  }
  d_lock.readUnlock();
}

void
PatchBasedDataWarehouse3::deleteParticles(ParticleSubset* delset)
{
 d_lock.writeLock();
  int matlIndex = delset->getMatlIndex();
  Patch* patch = (Patch*) delset->getPatch();
  psetDBType::key_type key(patch, matlIndex);
  psetDBType::iterator iter = d_delsetDB.find(key);
  ParticleSubset* currentDelset;
  if(iter != d_delsetDB.end()) {
    //    SCI_THROW(InternalError("deleteParticles called twice for patch"));
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
PatchBasedDataWarehouse3::addParticles(const Patch* patch, int matlIndex,
				    map<const VarLabel*, ParticleVariableBase*>* addedState)
{
 d_lock.writeLock();
  psetAddDBType::key_type key(matlIndex, patch);
  psetAddDBType::iterator iter = d_addsetDB.find(key);
  if(iter  != d_addsetDB.end()) 
    // SCI_THROW(InternalError("addParticles called twice for patch"));
    cerr << "addParticles called twice for patch" << endl;
  
  else
    d_addsetDB[key]=addedState;
  
 d_lock.writeUnlock();
}

void
PatchBasedDataWarehouse3::decrementScrubCount(const VarLabel* var, int matlIndex,
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
  case TypeDescription::ReductionVariable:
    SCI_THROW(InternalError("decrementScrubCount called for reduction variable: "+var->getName()));
  default:
    SCI_THROW(InternalError("decrementScrubCount for variable of unknown type: "+var->getName()));
  }
  d_lock.writeUnlock();
}

DataWarehouse::ScrubMode
PatchBasedDataWarehouse3::setScrubbing(ScrubMode scrubMode)
{
  ScrubMode oldmode = d_scrubMode;
  d_scrubMode = scrubMode;
  return oldmode;
}

void
PatchBasedDataWarehouse3::setScrubCount(const VarLabel* var, int matlIndex,
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
  case TypeDescription::ReductionVariable:
    // Reductions are not scrubbed
    SCI_THROW(InternalError("setScrubCount called for reduction variable: "+var->getName()));
  default:
    SCI_THROW(InternalError("setScrubCount for variable of unknown type: "+var->getName()));
  }
  d_lock.writeUnlock();
}

void
PatchBasedDataWarehouse3::scrub(const VarLabel* var, int matlIndex,
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
  case TypeDescription::ReductionVariable:
    // Reductions are not scrubbed
    SCI_THROW(InternalError("scrub called for reduction variable: "+var->getName()));
  default:
    SCI_THROW(InternalError("scrub for variable of unknown type: "+var->getName()));
  }
  d_lock.writeUnlock();
}

void
PatchBasedDataWarehouse3::initializeScrubs(int dwid, const map<VarLabelMatlDW<Patch>, int>& scrubcounts)
{
  d_lock.writeLock();
  d_varDB.initializeScrubs(dwid, scrubcounts);
  d_lock.writeUnlock();
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#endif  

void PatchBasedDataWarehouse3::
getGridVar(GridVariable& var, const VarLabel* label, 
           int matlIndex, const Patch* patch, Ghost::GhostType gtype, int numGhostCells)
{
  Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true);

  if(!d_varDB.exists(label, matlIndex, patch))
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex));
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
      SCI_THROW(InternalError("Ghost cells specified with type: None!\n"));
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
	ostringstream errmsg;
	errmsg << d_myworld->myrank() << " Reallocation Error:" <<
	  " Reallocation needed for " << label->getName();
	if (patch)
	  errmsg << " on patch " << patch->getID();
	errmsg << " for material " << matlIndex;
	//SCI_THROW(InternalError(errmsg.str().c_str()));
	//warn << "WARNING: this needs to be fixed:\n" << errmsg.str() << '\n';
       cout << "WARNING: this needs to be fixed:\n" << errmsg.str() << '\n';
      }
    }
    
    for(int i=0;i<(int)neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      if(neighbor && (neighbor != patch)){
	if(!d_varDB.exists(label, matlIndex, neighbor))
	  SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor,
				    matlIndex, neighbor == patch?
				    "on patch":"on neighbor"));
	GridVariable* srcvar = var.cloneType();
	d_varDB.get(label, matlIndex, neighbor, *srcvar);
	if(neighbor->isVirtual())
	  srcvar->offsetGrid(neighbor->getVirtualOffset());
	
	IntVector low = Max(lowIndex,
			    neighbor->getLowIndex(basis,
						  label->getBoundaryLayer()));
	IntVector high= Min(highIndex, 
			    neighbor->getHighIndex(basis,
						   label->getBoundaryLayer()));
	
	if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	    || ( high.z() < low.z() ) )
	  SCI_THROW(InternalError("Patch doesn't overlap?"));
	
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

void PatchBasedDataWarehouse3::
allocateTemporaryGridVar(GridVariable& var, const Patch* patch, Ghost::GhostType gtype, int numGhostCells,
                         const IntVector& boundaryLayer)
{
  Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true);

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

void PatchBasedDataWarehouse3::
allocateAndPutGridVar(GridVariable& var, const VarLabel* label, int matlIndex, const Patch* patch,
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
    SCI_THROW(InternalError("PatchBasedDataWarehouse3::AllocateAndPutGridVar can only be used when the dw has a running task associated with it."));
  }
#endif

  checkPutAccess(label, matlIndex, patch, false);  
  bool exists = d_varDB.exists(label, matlIndex, patch);

  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true);
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
      GridVariable* tmpVar = var.cloneType();
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
    GridVariable* clone = dynamic_cast<GridVariable*>(var.clone());
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


void PatchBasedDataWarehouse3::transferFrom(DataWarehouse* from,
					 const VarLabel* var,
					 const PatchSubset* patches,
					 const MaterialSubset* matls,
                                         bool replace /*=false*/,
                                         const PatchSubset* newPatches /*=0*/)
{
  PatchBasedDataWarehouse3* fromDW = dynamic_cast<PatchBasedDataWarehouse3*>(from);
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
      case TypeDescription::PerPatch:
      {
        if(!fromDW->d_varDB.exists(var, matl, patch))
          SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
            "in transferFrom"));
        GridVariable* v = dynamic_cast<GridVariable*>(fromDW->d_varDB.get(var, matl, patch));
        d_varDB.put(var, matl, copyPatch, v->clone(), replace);
      }
      break;
      case TypeDescription::ParticleVariable:
	{
	  if(!fromDW->d_varDB.exists(var, matl, patch))
	    SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
				      "in transferFrom"));

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
      case TypeDescription::ReductionVariable:
	SCI_THROW(InternalError("transferFrom doesn't work for reduction variable: "+var->getName()));
      default:
	SCI_THROW(InternalError("Unknown variable type in transferFrom: "+var->getName()));
      }
    }
  }
  d_lock.writeUnlock();
}

void PatchBasedDataWarehouse3::
putGridVar(GridVariable& var, const VarLabel* label, int matlIndex, const Patch* patch,
           bool replace /* = false */)
{
  ASSERT(!d_finalized);
 d_lock.writeLock();  

 checkPutAccess(label, matlIndex, patch, replace);

#if DAV_DEBUG
  cerr << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";
#endif
   // Error checking
   if(!replace && d_varDB.exists(label, matlIndex, patch))
     SCI_THROW(InternalError("put: grid variable already exists: " +
			     label->getName()));

   // Put it in the database
   Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true);
   IntVector low = patch->getLowIndex(basis, label->getBoundaryLayer());
   IntVector high = patch->getHighIndex(basis, label->getBoundaryLayer());
   if (Min(var.getLow(), low) != var.getLow() ||
       Max(var.getHigh(), high) != var.getHigh()) {
     ostringstream msg_str;
     msg_str << "put: Variable's window (" << var.getLow() << " - " << var.getHigh() << ") must encompass patches extent (" << low << " - " << high;
     SCI_THROW(InternalError(msg_str.str()));
   }
   USE_IF_ASSERTS_ON(bool no_realloc =) var.rewindow(low, high);
   // error would have been thrown above if the any reallocation would be
   // needed
   ASSERT(no_realloc);
   d_varDB.put(label, matlIndex, patch, &var, true);
  d_lock.writeUnlock();
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424 // template parameter not used in declaring arguments
#endif  
  
void PatchBasedDataWarehouse3::logMemoryUse(ostream& out, unsigned long& total,
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
PatchBasedDataWarehouse3::checkGetAccess(const VarLabel* /*label*/,
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
      
      map<VarLabelMatlPatch, AccessInfo>::iterator findIter;
      findIter = runningTaskAccesses.find(VarLabelMatlPatch(label, matlIndex,
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
            runningTaskAccesses[VarLabelMatlPatch(label, matlIndex, patch)];
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
PatchBasedDataWarehouse3::checkPutAccess(const VarLabel* /*label*/, int /*matlIndex*/,
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
				    patch, has, needs));
	}
      }
      else {
        runningTaskAccesses[VarLabelMatlPatch(label, matlIndex, patch)].accessType = replace ? ModifyAccess : PutAccess;
      }
    }
  }
#endif
#endif
}
  
inline void
PatchBasedDataWarehouse3::checkModifyAccess(const VarLabel* label, int matlIndex,
                                         const Patch* patch)
{ checkPutAccess(label, matlIndex, patch, true); }


inline bool
PatchBasedDataWarehouse3::hasGetAccess(const Task* runningTask,
                                    const VarLabel* label, int matlIndex,
                                    const Patch* patch, IntVector lowOffset,
                                    IntVector highOffset)
{ 
  return
    runningTask->hasRequires(label, matlIndex, patch, lowOffset, highOffset,
                             isFinalized() ? Task::OldDW : Task::NewDW);
}

inline
bool PatchBasedDataWarehouse3::hasPutAccess(const Task* runningTask,
                                         const VarLabel* label, int matlIndex,
                                         const Patch* patch, bool replace)
{
  if (replace)
    return runningTask->hasModifies(label, matlIndex, patch);
  else
    return runningTask->hasComputes(label, matlIndex, patch);
}

void PatchBasedDataWarehouse3::pushRunningTask(const Task* task,
					    vector<PatchBasedDataWarehouse3P>* dws)
{
  ASSERT(task);
 d_lock.writeLock();    
  d_runningTasks[Thread::self()].push_back(RunningTaskInfo(task, dws));
 d_lock.writeUnlock();
}

void PatchBasedDataWarehouse3::popRunningTask()
{
 d_lock.writeLock();
  list<RunningTaskInfo>& runningTasks = d_runningTasks[Thread::self()];
  runningTasks.pop_back();
  if (runningTasks.empty()) {
    d_runningTasks.erase(Thread::self());
  }
 d_lock.writeUnlock();
}

inline list<PatchBasedDataWarehouse3::RunningTaskInfo>*
PatchBasedDataWarehouse3::getRunningTasksInfo()
{
  map<Thread*, list<RunningTaskInfo> >::iterator findIt =
    d_runningTasks.find(Thread::self());
  return (findIt != d_runningTasks.end()) ? &findIt->second : 0;
}

inline bool PatchBasedDataWarehouse3::hasRunningTask()
{
  list<PatchBasedDataWarehouse3::RunningTaskInfo>* runningTasks =
    getRunningTasksInfo();
  return runningTasks ? !runningTasks->empty() : false;
}

inline PatchBasedDataWarehouse3::RunningTaskInfo*
PatchBasedDataWarehouse3::getCurrentTaskInfo()
{
  list<RunningTaskInfo>* taskInfoList = getRunningTasksInfo();
  return (taskInfoList && !taskInfoList->empty()) ? &taskInfoList->back() : 0;
}

DataWarehouse*
PatchBasedDataWarehouse3::getOtherDataWarehouse(Task::WhichDW dw)
{
  RunningTaskInfo* info = getCurrentTaskInfo();
  int dwindex = info->d_task->mapDataWarehouse(dw);
  DataWarehouse* result = (*info->dws)[dwindex].get_rep();
  ASSERT(result != 0);
  return result;
}

void PatchBasedDataWarehouse3::checkTasksAccesses(const PatchSubset* /*patches*/,
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
PatchBasedDataWarehouse3::checkAccesses(RunningTaskInfo* currentTaskInfo,
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
					has, needs));
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
					has, needs));
        }
      }
    }
  }
}


// For timestep abort/restart
bool PatchBasedDataWarehouse3::timestepAborted()
{
  return aborted;
}

bool PatchBasedDataWarehouse3::timestepRestarted()
{
  return restart;
}

void PatchBasedDataWarehouse3::abortTimestep()
{
  aborted=true;
}

void PatchBasedDataWarehouse3::restartTimestep()
{
  restart=true;
}

void PatchBasedDataWarehouse3::getVarLabelMatlLevelTriples( vector<VarLabelMatl<Level> >& vars ) const
{
  d_levelDB.getVarLabelMatlTriples(vars);
}

void PatchBasedDataWarehouse3::print()
{
  d_varDB.print(cout);  
  d_levelDB.print(cout);
}
