/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/DependencyException.h>
#include <CCA/Components/Schedulers/IncorrectAllocation.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Core/Parallel/BufferInfo.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstring>


using namespace std;
using namespace SCIRun;
using namespace Uintah;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)  From sus.cc:
extern SCIRun::Mutex       cerrLock;
extern DebugStream mixedDebug;

static DebugStream dbg( "OnDemandDataWarehouse", false );
static DebugStream warn( "OnDemandDataWarehouse_warn", true );
static DebugStream particles("DWParticles", false);
static DebugStream particles2("DWParticles2", false);
extern DebugStream mpidbg;

struct ParticleSend : public RefCounted {
  int numParticles;
};

// we want a particle message to have a unique tag per patch/matl/batch/dest.
// we only have 32K message tags, so this will have to do.
//   We need this because the possibility exists (particularly with DLB) of 
//   two messages with the same tag being sent from the same processor.  Even
//   if these messages are sent to different processors, they can get crossed in the mail
//   or one can overwrite the other.
#define PARTICLESET_TAG 0x4000|batch->messageTag
#define DAV_DEBUG 0

bool OnDemandDataWarehouse::d_combineMemory=true;

OnDemandDataWarehouse::OnDemandDataWarehouse(const ProcessorGroup* myworld,
                                             Scheduler* scheduler,
                                             int generation, 
                                             const GridP& grid,
                                             bool isInitializationDW/*=false*/)
   : DataWarehouse(myworld, scheduler, generation),
     d_lock("DataWarehouse lock"),
     d_lvlock("DataWarehouse level lock"),
     d_plock("DataWarehouse particle lock"),
     d_pvlock("DataWarehouse particle var lock"),
     d_pslock("DataWarehouse particle subset&state lock"),
     d_finalized( false ),
     d_grid(grid),
     d_isInitializationDW(isInitializationDW),
     d_scrubMode(DataWarehouse::ScrubNone)
{
  restart = false;
  hasRestarted_ = false;
  aborted = false;
}
//______________________________________________________________________
//
OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
  clear();
}
//______________________________________________________________________
//
void OnDemandDataWarehouse::clear()
{
  d_plock.writeLock();
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
  d_plock.writeUnlock();
  d_lock.writeLock();
  for (dataLocationDBtype::const_iterator iter = d_dataLocation.begin();
       iter != d_dataLocation.end(); iter++) {
    for (size_t i = 0; i<iter->second->size(); i++ )
      delete &(iter->second[i]);
    delete iter->second;
  }
  d_varDB.clear();
  d_lock.writeUnlock();
  d_pvlock.writeLock();
  d_pvarDB.clear();
  d_pvlock.writeUnlock();
  d_lvlock.writeLock();
  d_levelDB.clear();
  d_lvlock.writeUnlock();
}
//__________________________________
//
bool OnDemandDataWarehouse::isFinalized() const
{
   return d_finalized;
}
//__________________________________
//
void OnDemandDataWarehouse::finalize()
{
   d_lock.writeLock();
   d_varDB.cleanForeign();
   d_lock.writeUnlock();
   d_pvlock.writeLock();
   d_pvarDB.cleanForeign();
   d_pvlock.writeUnlock();
   d_finalized=true;   
}
//__________________________________
//
void OnDemandDataWarehouse::unfinalize()
{
  // this is for processes that need to make small modifications to the DW
  // after it has been finalized.
  d_finalized=false;
}
//__________________________________
//
void OnDemandDataWarehouse::refinalize()
{
  d_finalized=true;
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(Variable* var, 
                           const VarLabel* label,
                           int matlIndex, 
                           const Patch* patch)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::put(variable):" + label->getName());
   union {
      ReductionVariableBase* reduction;
      SoleVariableBase* sole;
      ParticleVariableBase* particle;
      GridVariableBase* gv;
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
   else if ((castVar.gv = dynamic_cast<GridVariableBase*>(var)) != NULL)
      put(*castVar.gv, label, matlIndex, patch);
   else
     SCI_THROW(InternalError("Unknown Variable type", __FILE__, __LINE__));
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(ReductionVariableBase& var,
                           const VarLabel* label, 
                           const Level* level,
                           int matlIndex /*= -1*/)
{
  checkGetAccess(label, matlIndex, 0);
  
  d_lvlock.readLock();

  if(!d_levelDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
                              "on reduction", __FILE__, __LINE__));
  }
  d_levelDB.get(label, matlIndex, level, var);

  d_lvlock.readUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(SoleVariableBase& var,
                           const VarLabel* label, 
                           const Level* level,
                           int matlIndex /*= -1*/)
{
  checkGetAccess(label, matlIndex, 0);
 
  d_lvlock.readLock();

  if(!d_levelDB.exists(label, matlIndex, level)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
                              "on sole", __FILE__, __LINE__));
  }
  d_levelDB.get(label, matlIndex, level, var);

  d_lvlock.readUnlock();
}
//______________________________________________________________________
//
bool
OnDemandDataWarehouse::exists(const VarLabel* label, int matlIndex,
                              const Patch* patch) const
{
  d_lock.readLock();
  if (patch && d_varDB.exists(label, matlIndex, patch)) {
    d_lock.readUnlock();
    return true;
  }
  d_lock.readUnlock();

  // level-independent reduction vars can be stored with a null level
  d_lvlock.readLock();
  if (d_levelDB.exists(label, matlIndex, patch?patch->getLevel():0)) {
    d_lvlock.readUnlock();
    return true;
  }
  d_lvlock.readUnlock();
  
  d_pvlock.readLock();
  if (patch && d_pvarDB.exists(label, matlIndex, patch)) {
    d_pvlock.readUnlock();
    return true;
  }
  d_pvlock.readUnlock();

  return false;
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::exists(const VarLabel* label) const
{
  d_lock.readLock();
  
  // level-independent reduction vars can be stored with a null level
   if(d_levelDB.exists(label, -1, 0) ) {
     d_lock.readUnlock();
     return true;
   } else {
     d_lock.readUnlock();
     return false;
   }
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::sendMPI(DependencyBatch* batch,
                               const VarLabel* pos_var,
                               BufferInfo& buffer,
                               OnDemandDataWarehouse* old_dw,
                               const DetailedDep* dep,
                               LoadBalancer* lb)
{
  if (dep->isNonDataDependency()) {
    // A non-data dependency -- send an empty message.
    // This would be used, for example, when a task is to modify data that
    // was previously required with ghost-cells.
    //buffer.add(0, 0, MPI_INT, false);
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

        d_pvlock.readLock();
        if(!d_pvarDB.exists(label, matlIndex, patch)) {
          SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
                "in sendMPI", __FILE__, __LINE__));
        }
        ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>(d_pvarDB.get(label, matlIndex, patch));
        d_pvlock.readUnlock();

        int dest = batch->toTasks.front()->getAssignedResourceIndex();
        ASSERTRANGE(dest, 0, d_myworld->size());

        ParticleSubset* sendset = 0;
        // first check to see if the receiving proc alrady has the (old) data
        // if data is relocating (of a regrid or re-load-balance), then the other
        // proc may already have it (since in most cases particle data comes from the old dw)
        // if lb is non-null, that means the particle data is on the old dw
        if (lb && lb->getOldProcessorAssignment(0, patch, 0) == dest) {
          if (this == old_dw) {
            // We don't need to know how many particles there are OR send any particle data...
            return;
          }
          ASSERT(old_dw->haveParticleSubset(matlIndex, patch));
          sendset = old_dw->getParticleSubset(matlIndex, patch);
        }
        else {
          sendset = old_dw->ss_.find_sendset(dest, patch, matlIndex, low, high, old_dw->d_generation);
        }
        if (!sendset) {
          // new dw send.  The NewDW doesn't yet know (on the first time) about this subset if it is on a different
          // processor.  Go ahead and calculate it, but there is no need to send it, since the other proc
          // already knows about it.
          ASSERT(old_dw != this);
          ParticleSubset* pset = var->getParticleSubset();
          sendset = scinew ParticleSubset(0, matlIndex, patch, low, high);
          constParticleVariable<Point> pos;
          old_dw->get(pos, pos_var, pset);
          for(ParticleSubset::iterator iter = pset->begin();
              iter != pset->end(); iter++){
            particleIndex idx = *iter;
            if(Patch::containsIndex(low,high,patch->getCellIndex(pos[idx]))) {
              sendset->addParticle(idx);
            }
          }
          old_dw->ss_.add_sendset(sendset, dest, patch, matlIndex, low, high, old_dw->d_generation);
          // cout << d_myworld->myrank() << "  NO SENDSET: " << patch->getID() << " matl " << matlIndex 
          //      << " " << low << " " << high << "\n";
        }
        ASSERT(sendset);
        if( sendset->numParticles() > 0) {
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
        d_lock.readLock();
        if(!d_varDB.exists(label, matlIndex, patch)) {
          cout << d_myworld->myrank() << "  Needed by " << *dep << " on task " << *dep->toTasks.front() << endl;
          SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
                "in sendMPI", __FILE__, __LINE__));
        }
        GridVariableBase* var;
        var = dynamic_cast<GridVariableBase*>(d_varDB.get(label, matlIndex, patch));
        d_lock.readUnlock();
        var->getMPIBuffer(buffer, dep->low, dep->high);
        buffer.addSendlist(var->getRefCounted());
      }
      break;
    default:
      SCI_THROW(InternalError("sendMPI not implemented for "+label->getFullName(matlIndex, patch), __FILE__, __LINE__));
  } // end switch( label->getType() );
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::exchangeParticleQuantities(DetailedTasks* dts, 
                                                  LoadBalancer* lb, 
                                                  const VarLabel* pos_var,
                                                  int iteration)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::exchangeParticleQuantities");
  if( hasRestarted_ ) {
    // If this DW is being used for a timestep restart, then it has already done this...
    return;
  }

  ParticleExchangeVar& recvs = dts->getParticleRecvs();
  ParticleExchangeVar& sends = dts->getParticleSends();

  // need to be sized here, otherwise you risk reallocating the array after a send/recv has been posted
  vector<vector<int> > senddata(sends.size()), recvdata(recvs.size());
  //int senddata[10][10] = {{-2}}, recvdata[10][10] = {{-2}};

  vector<MPI_Request> sendrequests, recvrequests;

  int data_index = 0;
  for (ParticleExchangeVar::iterator iter = recvs.begin(); iter != recvs.end(); iter++) {
    set<PSPatchMatlGhostRange>& r = iter->second;
    if (r.size() > 0) {
      recvdata[data_index].resize(r.size());
      // particles << d_myworld->myrank() << " Posting PARTICLES receives for " << r.size() 
      //           << " subsets from proc " << iter->first << " index " << data_index <<  endl;
      MPI_Request req;
      MPI_Irecv(&(recvdata[data_index][0]), r.size(), MPI_INT, iter->first, 16666,
                d_myworld->getComm(), &req);
      recvrequests.push_back(req);
      data_index++;
    }
  }

  data_index = 0;
  for (ParticleExchangeVar::iterator iter = sends.begin(); iter != sends.end(); iter++) {
    set<PSPatchMatlGhostRange>& s = iter->second;
    if (s.size() > 0) {
      vector<int>& data = senddata[data_index];
      data.resize(s.size());
      int i = 0;
      for (set<PSPatchMatlGhostRange>::iterator siter = s.begin(); siter != s.end(); siter++, i++) {
        const PSPatchMatlGhostRange& pmg = *siter;
        if ((pmg.dwid_ == DetailedDep::FirstIteration && iteration > 0) || 
            (pmg.dwid_ == DetailedDep::SubsequentIterations && iteration == 0)) {
          // not used
          data[i] = -2;
        }
        else if ( pmg.dwid_ == DetailedDep::FirstIteration && iteration == 0 &&
                  lb->getOldProcessorAssignment(0, pmg.patch_, 0) == iter->first ) {
          // signify that the recving proc already has this data.  Only use for the FirstIteration after a LB
          // send -1 rather than force the recving end above to iterate through its set
          data[i] = -1; 
        }
        else {
          if(!d_pvarDB.exists(pos_var, pmg.matl_, pmg.patch_)) {
            cout << d_myworld->myrank() << "  Naughty: patch " << pmg.patch_->getID() << " matl " << pmg.matl_ 
                 << " id " << pmg.dwid_  << endl;
            SCI_THROW(UnknownVariable(pos_var->getName(), getID(), pmg.patch_, pmg.matl_,
                                      "in exchangeParticleQuantities", __FILE__, __LINE__));
          } 
          // Make sure sendset is unique...
          ASSERT(!ss_.find_sendset(iter->first, pmg.patch_, pmg.matl_, pmg.low_, pmg.high_, d_generation)); 
          ParticleSubset* sendset = scinew ParticleSubset(0, pmg.matl_, pmg.patch_, pmg.low_, pmg.high_);
          constParticleVariable<Point> pos;
          get(pos, pos_var, pmg.matl_, pmg.patch_);
          ParticleSubset* pset = pos.getParticleSubset();
          for(ParticleSubset::iterator piter = pset->begin(); piter != pset->end(); piter++){
            if(Patch::containsIndex(pmg.low_,pmg.high_,pmg.patch_->getCellIndex(pos[*piter]))) {
              sendset->addParticle(*piter);
            }
          }
          ss_.add_sendset(sendset, iter->first, pmg.patch_, pmg.matl_, pmg.low_, pmg.high_, d_generation);
          data[i] = sendset->numParticles();
        }
        particles2 << d_myworld->myrank() << " Sending PARTICLES to proc " << iter->first << ": patch " 
                   << pmg.patch_->getID() << " matl " << pmg.matl_ << " low " << pmg.low_ << " high "
                   << pmg.high_ << " index " << i << ": " << senddata[data_index][i] << " particles\n";
      }
      // particles << d_myworld->myrank() << " Sending PARTICLES: " << s.size() << " subsets to proc " 
      //           << iter->first << " index " << data_index << endl;
      MPI_Request req;
      MPI_Isend(&(senddata[data_index][0]), s.size(), MPI_INT, iter->first, 16666,
                d_myworld->getComm(), &req);
      sendrequests.push_back(req);
      data_index++;
    }
  }

  MPI_Waitall(recvrequests.size(), &recvrequests[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(sendrequests.size(), &sendrequests[0], MPI_STATUSES_IGNORE);

  data_index = 0;
  // create particle subsets from recvs
  for (ParticleExchangeVar::iterator iter = recvs.begin(); iter != recvs.end(); iter++) {
    set<PSPatchMatlGhostRange>& r = iter->second;
    if (r.size() > 0) {
      vector<int>& data = recvdata[data_index];
      int i = 0;
      for (set<PSPatchMatlGhostRange>::iterator riter = r.begin(); riter != r.end(); riter++, i++) {
        const PSPatchMatlGhostRange& pmg = *riter;
        particles2 << d_myworld->myrank() << " Recving PARTICLES from proc " << iter->first << ": patch " 
                   << pmg.patch_->getID() << " matl " << pmg.matl_ << " low " << pmg.low_ << " high " << pmg.high_ 
                   << ": " << data[i] << "\n";
        if (data[i] == -2) {
          continue;
        }
        if (data[i] == -1) {
          ASSERT(pmg.dwid_ == DetailedDep::FirstIteration && iteration == 0 && haveParticleSubset(pmg.matl_, pmg.patch_));
          continue;
        }
        
        int & foreign_particles = d_foreignParticleQuantities[make_pair(pmg.matl_, pmg.patch_)];
        ParticleSubset* subset = createParticleSubset(data[i], pmg.matl_, pmg.patch_, pmg.low_, pmg.high_);

        // make room for other multiple subsets pointing into one variable - additional subsets 
        // referenced at the index above the last index of the previous subset
        if (data[i] > 0 && foreign_particles > 0) {
          // cout << d_myworld->myrank() << "  adjusting particles by " << foreign_particles << endl;
          for (ParticleSubset::iterator iter = subset->begin(); iter != subset->end(); iter++) {
            *iter = *iter + foreign_particles;
          }
        }
        foreign_particles += data[i];
        // cout << d_myworld->myrank() << "  Setting foreign particles of patch " << pmg.patch_->getID() 
        //      << " matl " << pmg.matl_ << " " << foreign_particles << endl;
      }
      data_index++;
    }
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::recvMPI(DependencyBatch* batch,
                               BufferInfo& buffer, 
                               OnDemandDataWarehouse* old_dw,
                               const DetailedDep* dep, LoadBalancer* lb)
{
  if (dep->isNonDataDependency()) {
    // A non-data dependency -- send an empty message.
    // This would be used, for example, for dependencies between a modifying
    // task and a task the requires the data before it is to be modified.
    //buffer.add(0, 0, MPI_INT, false);
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
      bool whole_patch_pset = false;
      // First, get the particle set.  We should already have it
      //      if(!old_dw->haveParticleSubset(matlIndex, patch, gt, ngc)){

      // if we already have a subset for the entire patch, there's little point 
      // in getting another one (and if we did, it would cause synchronization problems - see
      // comment in sendMPI)
      ParticleSubset* recvset = 0;      
      if (lb && (lb->getOldProcessorAssignment(0, patch, 0) == d_myworld->myrank() ||
                 lb->getPatchwiseProcessorAssignment(patch) == d_myworld->myrank())) {
        // first part of the conditional means "we used to own the ghost data so use the same particles"
        // second part means "we were just assigned to this patch and need to receive the whole thing"
        // we will never get here if they are both true, as mpi wouldn't need to be scheduled
        ASSERT(old_dw->haveParticleSubset(matlIndex, patch));
        recvset = old_dw->getParticleSubset(matlIndex, patch);
        whole_patch_pset = true;
      }
      else {
        recvset = old_dw->getParticleSubset(matlIndex,patch,low, high);
      }
      ASSERT(recvset);

      ParticleVariableBase* var = 0;
      d_pvlock.readLock();
      if (d_pvarDB.exists(label, matlIndex, patch)) {
        var = dynamic_cast<ParticleVariableBase*>(d_pvarDB.get(label, matlIndex, patch));
        d_pvlock.readUnlock();
        ASSERT(var->isForeign())
      }
      else {

        var = dynamic_cast<ParticleVariableBase*>(label->typeDescription()->createInstance());
        ASSERT(var != 0);
        var->setForeign();
        
        // set the foreign before the allocate (allocate CAN take multiple P Subsets, but only if it's foreign)
        if (whole_patch_pset)
        {
          MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::recvMPI(whole patch pset):" + label->getName());
          var->allocate(recvset);
        }
        else {
          MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::recvMPI:" + label->getName());
          // don't give this a pset as it could be a conatiner for several
          int allocated_particles = old_dw->d_foreignParticleQuantities[make_pair(matlIndex, patch)];
          var->allocate(allocated_particles);
        }
        d_pvlock.readUnlock();  //upgrade to write lock
        d_pvlock.writeLock();
        d_pvarDB.put(label, matlIndex, patch, var, true);
        d_pvlock.writeUnlock();
      }

      if(recvset->numParticles() > 0 && !(lb && lb->getOldProcessorAssignment(0, patch, 0) == d_myworld->myrank() && this == old_dw)){
        var->getMPIBuffer(buffer, recvset);
      }
    }
    break;
  case TypeDescription::NCVariable:
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
  {
    MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::recvMPI(cell variable):" + label->getName());
    //allocate the variable
    GridVariableBase* var = dynamic_cast<GridVariableBase*>(label->typeDescription()->createInstance());
    var->allocate(dep->low, dep->high);

    //set the var as foreign
    var->setForeign();
    var->setInvalid();

    //add the var to the dependency batch and set it as invalid.  The variable is now invalid because there is outstanding MPI pointing to the variable.
    batch->addVar(var);
    d_lock.writeLock();
    d_varDB.putForeign(label, matlIndex, patch, var); //put new var in data warehouse

    d_lock.writeUnlock();

    var->getMPIBuffer(buffer, dep->low, dep->high);
  }
  break;
  default:
    SCI_THROW(InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch), __FILE__, __LINE__));
  } // end switch( label->getType() );
} // end recvMPI()
//______________________________________________________________________
//
void
OnDemandDataWarehouse::reduceMPI(const VarLabel* label,
                                 const Level* level,
                                 const MaterialSubset* inmatls,
                                 int nComm)
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
  MPI_Op op = MPI_OP_NULL;
  MPI_Datatype datatype = MPI_DATATYPE_NULL;

  for(int m=0;m<nmatls;m++){
    int matlIndex = matls->get(m);

    ReductionVariableBase* var;
    d_lvlock.writeLock();
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
      //                                "on reduceMPI", __FILE__, __LINE__));
    }
    d_lvlock.writeUnlock();
    int sendcount;
    MPI_Datatype senddatatype = MPI_DATATYPE_NULL;
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
  MPI_Pack_size(count, datatype, d_myworld->getgComm(nComm), &packsize);
  vector<char> sendbuf(packsize);

  int packindex=0;
  for(int m=0;m<nmatls;m++){
    int matlIndex = matls->get(m);

    ReductionVariableBase* var;
    try {
      d_lvlock.readLock();
      var = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
      d_lvlock.readUnlock();
    } catch (UnknownVariable) {
      SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
                                "on reduceMPI(pass 2)", __FILE__, __LINE__));
    }
    var->getMPIData(sendbuf, packindex);
  }

  vector<char> recvbuf(packsize);

  if( mixedDebug.active() ) {
    cerrLock.lock(); mixedDebug << "calling MPI_Allreduce\n";
    cerrLock.unlock();
  }

  mpidbg << d_myworld->myrank() << " allreduce, name " << label->getName() << " level " << (level?level->getID():-1) << endl;
  int error = MPI_Allreduce(&sendbuf[0], &recvbuf[0], count, datatype, op,
                            d_myworld->getgComm(nComm));

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

  int unpackindex=0;
  for(int m=0;m<nmatls;m++){
    int matlIndex = matls->get(m);

    ReductionVariableBase* var;
    try {
      d_lvlock.readLock();
      var = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
      d_lvlock.readUnlock();
    } catch (UnknownVariable) {
      SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
                                "on reduceMPI(pass 2)", __FILE__, __LINE__));
    }
    var->putMPIData(recvbuf, unpackindex);
  }
  if(matls != inmatls)
    delete matls;
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(const ReductionVariableBase& var,
                           const VarLabel* label,
                           const Level* level,
                           int matlIndex /* = -1 */)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::put(Reduction):" + label->getName());
  ASSERT(!d_finalized);

  checkPutAccess(label, matlIndex, 0,
                 false /* it actually may be replaced, but it doesn't need
                          to explicitly modify with multiple reduces in the
                          task graph */);
  // Put it in the database
  d_lvlock.writeLock();
  if (!d_levelDB.exists(label, matlIndex, level))
    d_levelDB.put(label, matlIndex, level, var.clone(), false);
  else {
    ReductionVariableBase* foundVar
      = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
    foundVar->reduce(var);
  }
  d_lvlock.writeUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::override(const ReductionVariableBase& var,
                                const VarLabel* label, 
                                const Level* level,
                                int matlIndex /*=-1*/)
{
  checkPutAccess(label, matlIndex, 0, true);

  d_lvlock.writeLock();  
  // Put it in the database, replace whatever may already be there
  d_levelDB.put(label, matlIndex, level, var.clone(), true);
   
  d_lvlock.writeUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(const SoleVariableBase& var,
                           const VarLabel* label, 
                           const Level* level,
                           int matlIndex /* = -1 */)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::put(Sole Variable):" + label->getName());
  ASSERT(!d_finalized);

  checkPutAccess(label, matlIndex, 0,
                 false /* it actually may be replaced, but it doesn't need
                          to explicitly modify with multiple soles in the
                          task graph */);
  d_lvlock.writeLock();
  // Put it in the database
  if (!d_levelDB.exists(label, matlIndex, level))
    d_levelDB.put(label, matlIndex, level, var.clone(), false);
  
  d_lvlock.writeUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::override(const SoleVariableBase& var,
                                const VarLabel* label, 
                                const Level* level,
                                int matlIndex /*=-1*/)
{

  checkPutAccess(label, matlIndex, 0, true);

  d_lvlock.writeLock();  
  // Put it in the database, replace whatever may already be there
  d_levelDB.put(label, matlIndex, level, var.clone(), true);
   
  d_lvlock.writeUnlock();
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::createParticleSubset(particleIndex numParticles,
                                            int matlIndex, const Patch* patch,
                                            IntVector low /* = (0,0,0) */,
                                            IntVector high /* = (0,0,0) */)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::createParticleSubset):");

  if (low == high && high == IntVector(0,0,0)) {
    low = patch->getExtraCellLowIndex();
    high = patch->getExtraCellHighIndex();
  }


  dbg << d_myworld->myrank() << " DW ID " << getID() << " createParticleSubset: MI: " << matlIndex << " P: " << patch->getID() << " (" << low << ", " << high << ") size: " << numParticles << "\n";

  ASSERT(!patch->isVirtual());

  ParticleSubset* psubset = 
    scinew ParticleSubset(numParticles, matlIndex, patch, low, high);

  insertPSetRecord(d_psetDB,patch,low,high,matlIndex,psubset);

  return psubset;
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::saveParticleSubset(ParticleSubset* psubset, 
                                          int matlIndex, 
                                          const Patch* patch,
                                          IntVector low /* = (0,0,0) */,
                                          IntVector high /* = (0,0,0) */)
{
  ASSERTEQ(psubset->getPatch(), patch);
  ASSERTEQ(psubset->getMatlIndex(), matlIndex);
  ASSERT(!patch->isVirtual());  

  if (low == high && high == IntVector(0,0,0)) {
    low = patch->getExtraCellLowIndex();
    high = patch->getExtraCellHighIndex();
  }

  dbg << d_myworld->myrank() << " DW ID " << getID() << " saveParticleSubset: MI: " << matlIndex << " P: " << patch->getID() << " (" << low << ", " << high << ") size: " << psubset->numParticles() << "\n";
  insertPSetRecord(d_psetDB,patch,low,high,matlIndex,psubset);
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::printParticleSubsets()
{
  cout << "----------------------------------------------\n";
  cout << "-- Particle Subsets: \n\n";
  psetDBType::iterator iter;
  cout << d_myworld->myrank() << " Availabel psets on DW " << d_generation << ":\n";
  for (iter = d_psetDB.begin(); iter != d_psetDB.end(); iter++) {
    cout << d_myworld->myrank() << " " <<*(iter->second) << endl;
  }
  cout << "----------------------------------------------\n";
}

//______________________________________________________________________
//
void OnDemandDataWarehouse::insertPSetRecord(psetDBType &subsetDB,
                                              const Patch* patch,
                                              IntVector low, 
                                              IntVector high,
                                              int matlIndex, 
                                              ParticleSubset *psubset)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::insertPSetRecord");
  psubset->setLow(low);
  psubset->setHigh(high);

#if SCI_ASSERTION_LEVEL >= 1
  ParticleSubset *subset=queryPSetDB(subsetDB,patch,matlIndex,low,high,0,true);
  if(subset!=0) {
    if (d_myworld->myrank() == 0) {
      cout << d_myworld->myrank() << "  Duplicate: " << patch->getID() << " matl " << matlIndex << " " << low << " " << high << endl;
      printParticleSubsets();
    }
    SCI_THROW(InternalError("tried to create a particle subset that already exists", __FILE__, __LINE__));
  }
#endif
  d_plock.writeLock();
  psetDBType::key_type key(patch->getRealPatch(), matlIndex, getID());
  subsetDB.insert(pair<psetDBType::key_type,ParticleSubset*>(key,psubset));
  psubset->addReference();
  d_plock.writeUnlock();
}
//______________________________________________________________________
ParticleSubset* 
OnDemandDataWarehouse::queryPSetDB(psetDBType &subsetDB, 
                                   const Patch* patch,                      
                                   int matlIndex,                           
                                   IntVector low,                           
                                   IntVector high,                          
                                   const VarLabel* pos_var,                 
                                   bool exact)                              
{

  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::queryPSetDB");
  ParticleSubset* subset=0;


  psetDBType::key_type key(patch->getRealPatch(), matlIndex, getID());
  int best_volume = INT_MAX;
  int target_volume = Region::getVolume(low,high);

  d_plock.readLock();
  pair<psetDBType::const_iterator, psetDBType::const_iterator> ret = subsetDB.equal_range(key);
  
  //search multimap for best subset
  for(psetDBType::const_iterator iter=ret.first; iter!=ret.second; ++iter){
  
    ParticleSubset *ss=iter->second;
    IntVector sslow  = ss->getLow();
    IntVector sshigh = ss->getHigh();
    int vol=Region::getVolume(sslow,sshigh);

    //check if volume is better than current best
    if(vol<best_volume)
    {
      //intersect ranges
      if(low.x() >= sslow.x() && 
         low.y() >= sslow.y() && 
         low.z() >= sslow.z() &&
         sshigh.x() >= high.x() &&
         sshigh.y() >= high.y() &&
          sshigh.z() >= high.z() )
      {
        //take this range
        subset = ss;
        best_volume = vol;

        //short circuit out if we have already found the best possible solution
        if(best_volume==target_volume)
          break;

      }
    }
  }
  d_plock.readUnlock();

  if(exact && best_volume!=target_volume)
  {
    return 0;
  }

  //if we don't need to filter or we already have an exact match just return this subset
  if(pos_var==0 || best_volume==target_volume)
  {
    return subset;
  }

  //otherwise filter out particles that are not within this range
  constParticleVariable<Point> pos;

  ASSERT(subset!=0);
  get(pos, pos_var, subset);

  ParticleSubset* newsubset=scinew ParticleSubset(0, matlIndex, patch->getRealPatch(),low,high);

  for(ParticleSubset::iterator iter = subset->begin();iter != subset->end(); iter++){
    particleIndex idx = *iter;
    if(Patch::containsIndex(low,high,patch->getCellIndex(pos[idx]))) {
      newsubset->addParticle(idx);
    }
  }

  //save subset for future queries
  d_plock.writeLock();
  subsetDB.insert(pair<psetDBType::key_type,ParticleSubset*>(key,newsubset));
  newsubset->addReference();
  d_plock.writeUnlock();

  return newsubset;
}
//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, const Patch* patch)
{
  return getParticleSubset(matlIndex, patch, patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
}
//______________________________________________________________________
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, 
                                         const Patch* patch,
                                         IntVector low, 
                                         IntVector high)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::getParticleSubset-a");
  TAU_PROFILE("OnDemandDataWarehouse::getParticleSubset-a", " ", TAU_USER);
  
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
  ParticleSubset* subset = 0;

  subset=queryPSetDB(d_psetDB,realPatch,matlIndex,low,high,0);
  
  // bulletproofing
  if (!subset){
    printParticleSubsets();
    ostringstream s;
    s << "ParticleSubset, (low: " << low << ", high: " << high <<  " DWID " << getID() << ')';
    SCI_THROW(UnknownVariable(s.str().c_str(), getID(), realPatch, matlIndex,
                                "Cannot find particle set on patch", __FILE__, __LINE__));
  }
  return subset;
}
//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, 
                                         const Patch* patch,
                                         IntVector low, 
                                         IntVector high, 
                                         const VarLabel *pos_var)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::getParticleSubset-b");
  TAU_PROFILE("OnDemandDataWarehouse::getParticleSubset-a", " ", TAU_USER);
  
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
  ParticleSubset* subset = 0;

  subset=queryPSetDB(d_psetDB,realPatch,matlIndex,low,high,pos_var);
  
  // bulletproofing
  if (!subset){
    printParticleSubsets();
    ostringstream s;
    s << "ParticleSubset, (low: " << low << ", high: " << high <<  " DWID " << getID() << ')';
    SCI_THROW(UnknownVariable(s.str().c_str(), getID(), realPatch, matlIndex,
                                "Cannot find particle set on patch", __FILE__, __LINE__));
  }
  return subset;
}
//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, 
                                         const Patch* patch,
                                         Ghost::GhostType gtype,
                                         int numGhostCells,
                                         const VarLabel* pos_var)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::getParticleSubset-b");
  IntVector lowIndex, highIndex;
  patch->computeVariableExtents(Patch::CellBased, pos_var->getBoundaryLayer(),
                                    gtype, numGhostCells, lowIndex, highIndex);
                                
  if(gtype == Ghost::None || (lowIndex == patch->getExtraCellLowIndex() && highIndex == patch->getExtraCellHighIndex())) {
    return getParticleSubset(matlIndex, patch);
  }

  return getParticleSubset(matlIndex, lowIndex, highIndex, patch, pos_var);
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(int matlIndex, 
                                         IntVector lowIndex, 
                                         IntVector highIndex, 
                                         const Patch* relPatch, 
                                         const VarLabel* pos_var,
                                         const Level* oldLevel )  //level is ONLY used when querying from an old grid, otherwise the level will be determined from the patch
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::getParticleSubset-c");
  TAU_PROFILE("OnDemandDataWarehouse::getParticleSubset-b", " ", TAU_USER);
  // relPatch can be NULL if trying to get a particle subset for an arbitrary spot on the level
  Patch::selectType neighbors;
 
  ASSERT(relPatch!=0); //you should pass in the patch on which the task was called on
  const Level* level=relPatch->getLevel(); 
 
  //compute intersection between query range and patch
  IntVector low=Min(lowIndex,relPatch->getExtraCellLowIndex());
  IntVector high=Max(highIndex,relPatch->getExtraCellHighIndex());


  //if the user passed in the old level then query its patches
  if(oldLevel!=0)
  {
    oldLevel->selectPatches(lowIndex, highIndex, neighbors);  //find all intersecting patches with the range
  }
  //if the query range is larger than the patch
  else if(low!=relPatch->getExtraCellLowIndex() || high!=relPatch->getExtraCellHighIndex())
  {
    level->selectPatches(lowIndex, highIndex, neighbors);  //find all intersecting patches with the range
  }
  else
  {
    //just add this patch, do not query the whole level
    neighbors.push_back(relPatch);
  }

  particleIndex totalParticles = 0;
  vector<ParticleVariableBase*> neighborvars;
  vector<ParticleSubset*> subsets;
  vector<const Patch*> vneighbors;
  
  for(int i=0;i<neighbors.size();i++){
    const Patch* neighbor = neighbors[i];
    const Patch* realNeighbor = neighbor->getRealPatch();
    if(neighbor){
      IntVector newLow; 
      IntVector newHigh; 

      if (level->getIndex() == 0) {
        newLow = Max(lowIndex, neighbor->getExtraCellLowIndex());
        newHigh = Min(highIndex, neighbor->getExtraCellHighIndex());
      }
      else {
        // if in a copy-data timestep, only grab extra cells if on domain boundary
        newLow = Max(lowIndex, neighbor->getLowIndexWithDomainLayer(Patch::CellBased));
        newHigh = Min(highIndex, neighbor->getHighIndexWithDomainLayer(Patch::CellBased));
      }

      if (neighbor->isVirtual()) {
        // rather than offsetting each point of pos_var's data,
        // just adjust the box to compare it with.
        Vector offset = neighbor->getVirtualOffsetVector();
        IntVector cellOffset = neighbor->getVirtualOffset();
        newLow -= cellOffset;
        newHigh -= cellOffset;
      }
      ParticleSubset* pset;

      if (relPatch->getLevel()==level && relPatch != neighbor) {
        relPatch->cullIntersection(Patch::CellBased, IntVector(0,0,0), realNeighbor, newLow, newHigh);
        if (newLow == newHigh) {
          continue;
        }
      }

      //get the particle subset for this patch
      pset = getParticleSubset(matlIndex, neighbor, newLow, newHigh, pos_var);
      
      //add subset to our current list
      totalParticles+=pset->numParticles();
      subsets.push_back(pset);
      vneighbors.push_back(neighbors[i]);

    }
  }
 
  //create a new subset
  ParticleSubset* newsubset = scinew ParticleSubset(totalParticles, matlIndex, relPatch,
                                                    lowIndex, highIndex, vneighbors, subsets);
  return newsubset;
}
//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getDeleteSubset(int matlIndex, const Patch* patch)
{
  
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
  ParticleSubset *subset=queryPSetDB(d_delsetDB,realPatch,matlIndex,patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex(),0);
  
  if(subset==0){
     SCI_THROW(UnknownVariable("DeleteSet", getID(), realPatch, matlIndex,
                           "Cannot find delete set on patch", __FILE__, __LINE__));
   }
   return subset;
}

map<const VarLabel*, ParticleVariableBase*>* 
OnDemandDataWarehouse::getNewParticleState(int matlIndex, const Patch* patch)
{
  d_pslock.readLock();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;
  psetAddDBType::key_type key(matlIndex, realPatch);
  psetAddDBType::iterator iter = d_addsetDB.find(key);
  if(iter == d_addsetDB.end()){
    d_pslock.readUnlock();
    return 0;
  }
  d_pslock.readUnlock();
  return iter->second;
}


//______________________________________________________________________
//
bool
OnDemandDataWarehouse::haveParticleSubset(int matlIndex, 
                                          const Patch* patch,
                                          IntVector low /* = (0,0,0) */,
                                          IntVector high /* = (0,0,0) */, 
                                          bool exact /*=false*/)
{
  if (low == high && high == IntVector(0,0,0)) {
    low = patch->getExtraCellLowIndex();
    high = patch->getExtraCellHighIndex();
  }
  const Patch* realPatch = patch->getRealPatch();
  //query subset
  ParticleSubset *subset=queryPSetDB(d_psetDB,realPatch,matlIndex,low,high,0);

  //if no subset was returned there are no suitable subsets
  if(subset==0)
    return false;

  if(exact) //check if the user wanted an exact match
    return subset->getLow()==low && subset->getHigh()==high;
  else
    return true;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(constParticleVariableBase& constVar,
                           const VarLabel* label, 
                           int matlIndex,
                           const Patch* patch)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::get()-1");

  checkGetAccess(label, matlIndex, patch);
  
  d_pvlock.readLock();

  if(!d_pvarDB.exists(label, matlIndex, patch)) {
    print();
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
  }
  constVar = *dynamic_cast<ParticleVariableBase*>(d_pvarDB.get(label, matlIndex, patch));
   
  d_pvlock.readUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(constParticleVariableBase& constVar,
                           const VarLabel* label,
                           ParticleSubset* pset)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::get()-2");
  TAU_PROFILE("OnDemandDataWarehouse::get(particle variable)", " ", TAU_USER);
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();

  // pset center patch and neighbor patch are not in same level
  // (probably on an AMR copy data timestep)
  if ((pset->getNeighbors().size() == 0) ||
      (pset->getNeighbors().front()->getLevel() == patch->getLevel() &&
       pset->getLow()  == patch->getExtraCellLowIndex()              &&
       pset->getHigh() == patch->getExtraCellHighIndex()             )){
    get(constVar, label, matlIndex, patch);
  }
  else {
    checkGetAccess(label, matlIndex, patch);
    ParticleVariableBase* var = constVar.cloneType();

    const vector<const Patch*>& neighborPatches     = pset->getNeighbors();
    const vector<ParticleSubset*>& neighbor_subsets = pset->getNeighborSubsets();

    vector<ParticleVariableBase*> neighborvars(neighborPatches.size());

    d_pvlock.readLock();
    for(size_t i=0;i<neighborPatches.size();i++){
      const Patch* neighborPatch=neighborPatches[i];

      if(!d_pvarDB.exists(label, matlIndex, neighborPatches[i])){
        SCI_THROW(UnknownVariable(label->getName(), getID(), neighborPatch, matlIndex,
              neighborPatch == patch?"on patch":"on neighbor", __FILE__, __LINE__));
      }

      neighborvars[i] = var->cloneType();

      d_pvarDB.get(label, matlIndex, neighborPatch, *neighborvars[i]);
    }
    d_pvlock.readUnlock();    

    // Note that when the neighbors are virtual patches (i.e. periodic
    // boundaries), then if var is a ParticleVariable<Point>, the points
    // of neighbors will be translated by its virtualOffset.

    var->gather(pset, neighbor_subsets, neighborvars, neighborPatches);

    constVar = *var;

    for (size_t i=0;i<neighborPatches.size();i++){
      delete neighborvars[i];
    }
    delete var;

  }
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::getModifiable(ParticleVariableBase& var,
                                     const VarLabel* label,
                                     ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();
   checkModifyAccess(label, matlIndex, patch);
   
   d_pvlock.readLock();
   if(pset->getLow() == patch->getExtraCellLowIndex() && pset->getHigh() == patch->getExtraCellHighIndex()){
     if(!d_pvarDB.exists(label, matlIndex, patch))
       SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
     d_pvarDB.get(label, matlIndex, patch, var);
   } else {
     SCI_THROW(InternalError("getModifiable (Particle Variable (" + label->getName() +") ).  The particleSubset low/high index does not match the patch low/high indices", __FILE__, __LINE__));
   }
   d_pvlock.readUnlock();
}
//______________________________________________________________________
//
ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable(const VarLabel* label,
                                           ParticleSubset* pset)
{
   int matlIndex = pset->getMatlIndex();
   const Patch* patch = pset->getPatch();

   if(pset->getLow() == patch->getExtraCellLowIndex() && pset->getHigh() == patch->getExtraCellHighIndex()){
     return getParticleVariable(label, matlIndex, patch);
   } else {
     SCI_THROW(InternalError("getParticleVariable (Particle Variable (" + label->getName() +") ).  The particleSubset low/high index does not match the patch low/high indices", __FILE__, __LINE__));
   }
}
//______________________________________________________________________
//
ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable(const VarLabel* label,
                                           int matlIndex, 
                                          const Patch* patch)
{
   ParticleVariableBase* var = 0;  

   // in case the it's a virtual patch -- only deal with real patches
   if (patch != 0) patch = patch->getRealPatch();
   
  
   checkModifyAccess(label, matlIndex, patch);
   
   d_pvlock.readLock();
   if(!d_pvarDB.exists(label, matlIndex, patch))
     SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
   var = dynamic_cast<ParticleVariableBase*>(d_pvarDB.get(label, matlIndex, patch));

   d_pvlock.readUnlock();
   return var;
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::allocateTemporary(ParticleVariableBase& var,
                                         ParticleSubset* pset)
{  
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::allocateTemporary(Particle Variable):");
  //TAU_PROFILE("allocateTemporary()", "OnDemand.cc", TAU_USER);
  var.allocate(pset);
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::allocateAndPut(ParticleVariableBase& var,
                                      const VarLabel* label,
                                      ParticleSubset* pset)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::allocateAndPut(Particle Variable):" + label->getName());
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();
  
  // Error checking
  d_pvlock.readLock();   
  if(d_pvarDB.exists(label, matlIndex, patch))
    SCI_THROW(InternalError("Particle variable already exists: " +
                            label->getName(), __FILE__, __LINE__));
  d_pvlock.readUnlock();
  
  var.allocate(pset);
  put(var, label);
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(ParticleVariableBase& var,
                           const VarLabel* label, 
                           bool replace /*= false*/)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::put(Particle Variable):" + label->getName());
  ASSERT(!d_finalized);  

  ParticleSubset* pset = var.getParticleSubset();

  const Patch* patch = pset->getPatch();

  if(pset->getLow() != patch->getExtraCellLowIndex() || pset->getHigh() != patch->getExtraCellHighIndex()){
    SCI_THROW(InternalError(" put(Particle Variable (" + label->getName() +") ).  The particleSubset low/high index does not match the patch low/high indices", __FILE__, __LINE__)); 
  }
  
  int matlIndex = pset->getMatlIndex();
  
  
  if( dbg.active() ) {
    cerrLock.lock();
  dbg << d_myworld->myrank() << " Putting: ";
  dbg<< left;dbg.width(20);
  dbg << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " \tinto DW: " << d_generation << "\n";
    cerrLock.unlock();
  }

  checkPutAccess(label, matlIndex, patch, replace);

  d_pvlock.writeLock();   
  // Put it in the database
  d_pvarDB.put(label, matlIndex, patch, var.clone(), replace);
  d_pvlock.writeUnlock();
}
//______________________________________________________________________
//
void OnDemandDataWarehouse::copyOut(ParticleVariableBase& var, 
                                    const VarLabel* label,
                                    ParticleSubset* pset)
{
  constParticleVariableBase* constVar = var.cloneConstType();
  this->get(*constVar, label, pset);
  var.copyData(&constVar->getBaseRep());
  delete constVar;
}
//______________________________________________________________________
//
void OnDemandDataWarehouse::getCopy(ParticleVariableBase& var, 
                                    const VarLabel* label,
                                    ParticleSubset* pset)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::getCopy(Particle Variable):" + label->getName());
  constParticleVariableBase* constVar = var.cloneConstType();
  this->get(*constVar, label, pset);
  var.allocate(pset);
  var.copyData(&constVar->getBaseRep());
  delete constVar;
}

//______________________________________________________________________
//
void 
OnDemandDataWarehouse::get(constGridVariableBase& constVar,
                           const VarLabel* label,
                           int matlIndex, 
                           const Patch* patch,
                           Ghost::GhostType gtype, 
                           int numGhostCells)
{
  GridVariableBase* var = constVar.cloneType();
  
  checkGetAccess(label, matlIndex, patch, gtype, numGhostCells);
  d_lock.readLock();
  getGridVar(*var, label, matlIndex, patch, gtype, numGhostCells);
  d_lock.readUnlock();
 
  constVar = *var;
  delete var;
}
//______________________________________________________________________
//
void 
OnDemandDataWarehouse::getModifiable(GridVariableBase& var,
                                     const VarLabel* label,      
                                     int matlIndex,              
                                     const Patch* patch,         
                                     Ghost::GhostType gtype,      
                                     int numGhostCells)          
{
 d_lock.readLock();  
 //checkModifyAccess(label, matlIndex, patch);
  getGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
 d_lock.readUnlock();  
}

//______________________________________________________________________
//
void 
OnDemandDataWarehouse:: allocateTemporary(GridVariableBase& var, 
                                          const Patch* patch,
                                          Ghost::GhostType gtype, 
                                          int numGhostCells )
{
  IntVector boundaryLayer(0, 0, 0); // Is this right?

  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::allocateTemporary(Grid Variable)");
  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), false);
  Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
                         numGhostCells, lowOffset, highOffset);
                      
  patch->computeExtents(basis, boundaryLayer, lowOffset, highOffset,lowIndex, highIndex);

  var.allocate(lowIndex, highIndex);
}
//______________________________________________________________________
//
void 
OnDemandDataWarehouse::allocateAndPut(GridVariableBase& var, 
                                      const VarLabel* label,       
                                      int matlIndex,               
                                      const Patch* patch,          
                                      Ghost::GhostType gtype,      
                                      int numGhostCells)           
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::allocateAndPut(Grid Variable):" + label->getName());
  if (d_finalized)
    cerr << "  DW " << getID() << " finalized!\n";
  ASSERT(!d_finalized);

  // Note: almost the entire function is write locked in order to prevent dual
  // allocations in a multi-threaded environment.  Whichever patch in a
  // super patch group gets here first, does the allocating for the entire
  // super patch group.

#if 0
  if (!hasRunningTask()) {
    SCI_THROW(InternalError("OnDemandDataWarehouse::AllocateAndPutGridVar can only be used when the dw has a running task associated with it.", __FILE__, __LINE__));
  }
#endif

  checkPutAccess(label, matlIndex, patch, false);  
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype,
                         numGhostCells, lowOffset, highOffset);
  patch->computeExtents(basis, label->getBoundaryLayer(),
                        lowOffset, highOffset, lowIndex, highIndex);
 
  d_lock.writeLock();
  bool exists = d_varDB.exists(label, matlIndex, patch);
  if(!d_combineMemory)
  {
    if (exists) 
    {
      // it had been allocated and put as part of the superpatch of
      // another patch
      d_varDB.get(label, matlIndex, patch, var);

      // The var's window should be the size of the patch or smaller than it.
      ASSERTEQ(Min(var.getLow(), lowIndex), lowIndex);
      ASSERTEQ(Max(var.getHigh(), highIndex), highIndex);

      // this is just a tricky way to uninitialize var
      Variable* tmpVar = dynamic_cast<Variable*>(var.cloneType());
      var.copyPointer(*tmpVar);
      delete tmpVar;
    }
    //allocate the memory
    var.allocate(lowIndex, highIndex);
    //put the variable in the database
    d_varDB.put(label, matlIndex, patch, var.clone() ,true);
  }
  else
  {
    if (exists) 
    {
      // it had been allocated and put as part of the superpatch of
      // another patch
      d_varDB.get(label, matlIndex, patch, var);

      // The var's window should be the size of the patch or smaller than it.
      ASSERTEQ(Min(var.getLow(), lowIndex), lowIndex);
      ASSERTEQ(Max(var.getHigh(), highIndex), highIndex);

      if (var.getLow() != patch->getExtraLowIndex(basis, label->getBoundaryLayer()) ||
          var.getHigh() != patch->getExtraHighIndex(basis, label->getBoundaryLayer()) ||
          var.getBasePointer() == 0 /* place holder for ghost patch */) 
      {
        // It wasn't allocated as part of another patch's superpatch;
        // it existed as ghost patch of another patch.. so we have no
        // choice but to blow it away and replace it.
        d_varDB.put(label, matlIndex, patch, 0, true);

        // this is just a tricky way to uninitialize var
        Variable* tmpVar = dynamic_cast<Variable*>(var.cloneType());
        var.copyPointer(*tmpVar);
        delete tmpVar;
      }
      else 
      {
        // It was allocated and put as part of the superpatch of another patch
        d_lock.writeUnlock();      
        var.rewindow(lowIndex, highIndex);
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

    var.allocate(superLowIndex, superHighIndex);

#if SCI_ASSERTION_LEVEL >= 3


    // check for dead portions of a variable (variable space that isn't covered by any patch).  
    // This will happen with L-shaped patch configs and ngc > extra cells.  
    // find all dead space and mark it with a bogus value.

    if (1) 
    { // numGhostCells > ec) { (numGhostCells is 0, query it from the superLowIndex...
      deque<Box> b1, b2, difference;
      b1.push_back(Box(Point(superLowIndex(0), superLowIndex(1), superLowIndex(2)), 
            Point(superHighIndex(0), superHighIndex(1), superHighIndex(2))));
      for (size_t i = 0; i < (*superPatchGroup).size(); i++) 
      {
        const Patch* p = (*superPatchGroup)[i];
        IntVector low = p->getExtraLowIndex(basis, label->getBoundaryLayer());
        IntVector high = p->getExtraHighIndex(basis, label->getBoundaryLayer());
        b2.push_back(Box(Point(low(0), low(1), low(2)), Point(high(0), high(1), high(2))));
      }
      difference = Box::difference(b1, b2);

#if 0
      if (difference.size() > 0) {
        cout << "Box difference: " << superLowIndex << " " << superHighIndex << " with patches " << endl;
        for (size_t i = 0; i < (*superPatchGroup).size(); i++) {
          const Patch* p = (*superPatchGroup)[i];
          cout << p->getExtraLowIndex(basis, label->getBoundaryLayer()) << " " << p->getExtraHighIndex(basis, label->getBoundaryLayer()) << endl;
        }

        for (size_t i = 0; i < difference.size(); i++) {
          cout << difference[i].lower() << " " << difference[i].upper() << endl;
        }
      }
#endif
      // get more efficient way of doing this...
      for (size_t i = 0; i < difference.size(); i++) 
      {
        Box b = difference[i];
        IntVector low((int)b.lower()(0), (int)b.lower()(1), (int)b.lower()(2));
        IntVector high((int)b.upper()(0), (int)b.upper()(1), (int)b.upper()(2));
        if (GridVariable<double>* typedVar = dynamic_cast<GridVariable<double>*>(&var)) 
        {
          for (CellIterator iter(low, high); !iter.done(); iter++)
            (*typedVar)[*iter] = -5.555555e256;
        }
        else if (GridVariable<Vector>* typedVar = dynamic_cast<GridVariable<Vector>*>(&var)) 
        {
          for (CellIterator iter(low, high); !iter.done(); iter++)
            (*typedVar)[*iter] = -5.555555e256;
        }
      }
    }
#endif 

    Patch::selectType encompassedPatches;
    if (requiredSuperLow == lowIndex && requiredSuperHigh == highIndex) 
    {
      // only encompassing the patch currently being allocated
      encompassedPatches.push_back(patch);
    }
    else 
    {
      // Use requiredSuperLow/High instead of superLowIndex/superHighIndex
      // so we don't put the var for patches in the datawarehouse that won't be
      // required (this is important for scrubbing).
      patch->getLevel()->selectPatches(requiredSuperLow, requiredSuperHigh,
          encompassedPatches);
    }

    // Make a set of the non ghost patches that
    // has quicker lookup than the vector.
    set<const Patch*> nonGhostPatches;
    for (size_t i = 0; i < superPatchGroup->size(); ++i) 
    {
      nonGhostPatches.insert((*superPatchGroup)[i]);
    }

    Patch::selectType::iterator iter = encompassedPatches.begin();    
    for (; iter != encompassedPatches.end(); ++iter) 
    {
      const Patch* patchGroupMember = *iter;
      GridVariableBase* clone = var.clone();
      IntVector groupMemberLowIndex = patchGroupMember->getExtraLowIndex(basis, label->getBoundaryLayer());
      IntVector groupMemberHighIndex = patchGroupMember->getExtraHighIndex(basis, label->getBoundaryLayer());
      IntVector enclosedLowIndex = Max(groupMemberLowIndex, superLowIndex);
      IntVector enclosedHighIndex = Min(groupMemberHighIndex, superHighIndex);

      clone->rewindow(enclosedLowIndex, enclosedHighIndex);
      if (patchGroupMember == patch) 
      {
        // this was checked already
        exists = false;
      }
      else 
      {
        exists = d_varDB.exists(label, matlIndex, patchGroupMember);
      }
      if (patchGroupMember->isVirtual()) 
      {
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
      if (exists) 
      {
        // variable section already exists in this patchGroupMember
        // (which is assumed to be a ghost patch)
        // so check if one is enclosed in the other.

        GridVariableBase* existingGhostVar =
          dynamic_cast<GridVariableBase*>(d_varDB.get(label, matlIndex, patchGroupMember));
        IntVector existingLow = existingGhostVar->getLow();
        IntVector existingHigh = existingGhostVar->getHigh();
        IntVector minLow = Min(existingLow, enclosedLowIndex);
        IntVector maxHigh = Max(existingHigh, enclosedHighIndex);

        if (existingGhostVar->isForeign()) 
        {
          // data already being received, so don't replace it
          delete clone;
        }
        else if (minLow == enclosedLowIndex && maxHigh == enclosedHighIndex) 
        {
          // this new ghost variable section encloses the old one,
          // so replace the old one
          d_varDB.put(label, matlIndex, patchGroupMember, clone, true);
        }
        else 
        {
          // Either the old ghost variable section encloses this new one
          // (so leave it), or neither encloses the other (so just forget
          // about it -- it'll allocate extra space for it when receiving
          // the ghost data in recvMPIGridVar if nothing else).
          delete clone;
        }
      }
      else 
      {
        // it didn't exist before -- add it
        d_varDB.put(label, matlIndex, patchGroupMember, clone, false);
      }
    }
  }
  d_lock.writeUnlock();
  var.rewindow(lowIndex, highIndex);
}
//______________________________________________________________________
//
void OnDemandDataWarehouse::copyOut(GridVariableBase& var, 
                                    const VarLabel* label,
                                    int matlIndex, 
                                    const Patch* patch,
                                    Ghost::GhostType gtype, 
                                    int numGhostCells)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::copyOut(Grid Variable):" + label->getName());
  GridVariableBase* tmpVar = var.cloneType();
  d_lock.readLock();
  getGridVar(*tmpVar, label, matlIndex, patch, gtype, numGhostCells);
  d_lock.readUnlock();
  var.copyData(tmpVar);
  delete tmpVar;
}
//______________________________________________________________________
//
void 
OnDemandDataWarehouse::getCopy(GridVariableBase& var, 
                               const VarLabel* label,
                               int matlIndex, 
                               const Patch* patch,
                               Ghost::GhostType gtype, 
                               int numGhostCells)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::getCopy(Grid Variable):" + label->getName());
  GridVariableBase* tmpVar = var.cloneType();
  d_lock.readLock();
  getGridVar(*tmpVar, label, matlIndex, patch, gtype, numGhostCells);
  d_lock.readUnlock();
  var.allocate(tmpVar);
  var.copyData(tmpVar);
  delete tmpVar;
}


//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(GridVariableBase& var,
                           const VarLabel* label,
                           int matlIndex, 
                           const Patch* patch,
                           bool replace /*= false*/)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::put(Grid Variable):" + label->getName());
  ASSERT(!d_finalized);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);
  ASSERTEQ(basis, Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));    

 checkPutAccess(label, matlIndex, patch, replace);

#if DAV_DEBUG
  cerr << "Putting: " << *label << " MI: " << matlIndex << " patch: " 
       << *patch << " into DW: " << d_generation << "\n";
#endif
   // Put it in the database
   IntVector low = patch->getExtraLowIndex(basis, label->getBoundaryLayer());
   IntVector high = patch->getExtraHighIndex(basis, label->getBoundaryLayer());
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
   d_lock.writeLock();  
   d_varDB.put(label, matlIndex, patch, var.clone(), true);
   d_lock.writeUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(PerPatchBase& var, 
                           const VarLabel* label,
                           int matlIndex, 
                           const Patch* patch)
{
  checkGetAccess(label, matlIndex, patch);
  d_lock.readLock();
  if(!d_varDB.exists(label, matlIndex, patch))
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex,
                           "perpatch data", __FILE__, __LINE__));
  d_varDB.get(label, matlIndex, patch, var);
  d_lock.readUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(PerPatchBase& var,
                           const VarLabel* label,
                           int matlIndex, 
                           const Patch* patch,
                           bool replace /*= false*/)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::put(Per Patch Variable):" + label->getName());
  ASSERT(!d_finalized);  
  //checkPutAccess(label, replace);
  checkPutAccess(label, matlIndex, patch, replace);
  

   d_lock.writeLock();
   // Put it in the database
   d_varDB.put(label, matlIndex, patch, var.clone(), true);
   d_lock.writeUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::getRegion(constGridVariableBase& constVar,
                                 const VarLabel* label,
                                 int matlIndex, 
                                 const Level* level,
                                 const IntVector& low, 
                                 const IntVector& high,
                                 bool useBoundaryCells /*=true*/)
{
  MALLOC_TRACE_TAG_SCOPE("OnDemandDataWarehouse::getRegion(Grid Variable):" + label->getName());

  GridVariableBase* var = constVar.cloneType();
  var->allocate(low, high);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  IntVector adjustment = IntVector(1,1,1);
  if (basis == Patch::XFaceBased){
     adjustment = IntVector(1,0,0);
  }else if (basis == Patch::YFaceBased){
    adjustment = IntVector(0,1,0);
  }else if (basis == Patch::ZFaceBased){ 
    adjustment = IntVector(0,0,1);
  }
  
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
  
  int totalCells=0;
  for(int i=0;i<patches.size();i++){
    const Patch* patch = patches[i];
    IntVector l, h;

    // the caller should determine whether or not he wants extra cells.
    // It will matter in AMR cases with corner-aligned patches
    if (useBoundaryCells) {
      l = Max(patch->getExtraLowIndex(basis, label->getBoundaryLayer()), low);
      h = Min(patch->getExtraHighIndex(basis, label->getBoundaryLayer()), high);
    }
    else {
      l = Max(patch->getLowIndex(basis), low);
      h = Min(patch->getHighIndex(basis), high);
    }
    
    if (l.x() >= h.x() || l.y() >= h.y() || l.z() >= h.z()){
      continue;
    }
    vector<Variable*> varlist;
    d_lock.readLock();
    d_varDB.getlist(label, matlIndex, patch, varlist);
    d_lock.readUnlock();
    GridVariableBase* v=NULL;
    
    for (vector<Variable*>::reverse_iterator rit = varlist.rbegin(); ; ++rit){
      if (rit == varlist.rend()){
        v = NULL;
        break;
      }
      v = dynamic_cast<GridVariableBase*>(*rit);
      //verify that the variable is valid and matches the dependencies requirements.
      if ((v!=NULL) && v->isValid() && Min(l, v->getLow()) == v->getLow()  &&  Max(h, v->getHigh()) == v->getHigh()){  //find a completed region
        break;
      }
    }
    
    // just like a "missing patch": got data on this patch, but it either corresponds to a different
    // region or is incomplete"
    if (v == NULL){
      missing_patches.push_back(patch->getRealPatch());
      continue;
    }
    
    GridVariableBase* tmpVar = var->cloneType();
    tmpVar->copyPointer(*v);
    
    if (patch->isVirtual()) {
      // if patch is virtual, it is probable a boundary layer/extra cell that has been requested (from AMR)
      // let Bryan know if this doesn't work.  We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }
    try {
      var->copyPatch(tmpVar, l, h);
    } catch (InternalError& e) {
      cout << " Bad range: " << low << " " << high << ", patch intersection: " << l << " " << h 
           << " actual patch " << patch->getLowIndex(basis) << " " << patch->getHighIndex(basis) 
           << " var range: "  << tmpVar->getLow() << " " << tmpVar->getHigh() << endl;
      throw e;
    }
    delete tmpVar;
    IntVector diff(h-l);
    totalCells += diff.x()*diff.y()*diff.z();
  }  // patches loop
  
  IntVector diff(high-low);

  if (diff.x()*diff.y()*diff.z() > totalCells && missing_patches.size() > 0) {
    cout << d_myworld->myrank() << "  Unknown Variable " << *label << ", matl " << matlIndex 
         << ", L-" << level->getIndex() << ", for patch(es): ";
    for (size_t i = 0; i < missing_patches.size(); i++) 
      cout << *missing_patches[i] << " ";
    cout << endl << " Original region: " << low << " " << high << endl;
    cout << " copied cells: " << totalCells << " requested cells: " << diff.x()*diff.y()*diff.z() << endl;
    throw InternalError("Missing patches in getRegion", __FILE__, __LINE__);
  }
  if (dbg.active()) {
    cerrLock.lock();
    dbg << d_myworld->myrank() << "  Variable " << *label << ", matl " << matlIndex << ", L-" << level->getIndex() 
          << " For region: " << low << " " << high << "  has been gotten" << endl;
    cerrLock.unlock();
  }

  ASSERT(diff.x()*diff.y()*diff.z() <= totalCells);

  constVar = *dynamic_cast<GridVariableBase*>(var);
  delete var;
}
//______________________________________________________________________
//
void 
OnDemandDataWarehouse::emit(OutputContext& oc, 
                            const VarLabel* label,
                            int matlIndex, 
                            const Patch* patch)
{
  checkGetAccess(label, matlIndex, patch);

  Variable* var = NULL;
  IntVector l, h;
  if(patch) {
    // Save with the boundary layer, otherwise restarting from the DataArchive won't work.
    patch->computeVariableExtents( label->typeDescription()->getType(),
        label->getBoundaryLayer(), Ghost::None, 0,
        l, h );
   switch(label->typeDescription()->getType())
    {
      case TypeDescription::NCVariable:
      case TypeDescription::CCVariable:
      case TypeDescription::SFCXVariable:
      case TypeDescription::SFCYVariable:
      case TypeDescription::SFCZVariable:
        //get list
      {
        vector<Variable*> varlist;
        d_lock.readLock();
        d_varDB.getlist(label, matlIndex, patch, varlist);
        d_lock.readUnlock();

        GridVariableBase* v = NULL;
        for (vector<Variable*>::reverse_iterator rit = varlist.rbegin();; ++rit) {
          if (rit == varlist.rend()) {
            v = NULL;
            break;
          }
          v = dynamic_cast<GridVariableBase*> (*rit);
          //verify that the variable is valid and matches the dependencies requirements.
          if (v && v->isValid() && Min(l, v->getLow()) == v->getLow() && Max(h, v->getHigh()) == v->getHigh()) //find a completed region
            break;
        }
        var = v;
      }
        break;
      case TypeDescription::ParticleVariable:
        d_lvlock.readLock();
        var=d_pvarDB.get(label, matlIndex, patch);
        d_lvlock.readUnlock();
        break;
      default:
        d_lock.readLock();
        var=d_varDB.get(label, matlIndex, patch);
        d_lock.readUnlock();
    }
  }
  else
  {
    l=h=IntVector(-1,-1,-1);

    const Level* level = patch?patch->getLevel():0;
    d_lvlock.readLock();
    if(d_levelDB.exists(label, matlIndex, level))
      var = d_levelDB.get(label, matlIndex, level);
    d_lvlock.readUnlock();
  }

  if (var == NULL) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "on emit", __FILE__, __LINE__));
  }
  var->emit(oc, l, h, label->getCompressionMode());
}
//______________________________________________________________________
//
void 
OnDemandDataWarehouse::print(ostream& intout, 
                             const VarLabel* label,
                             const Level* level, 
                             int matlIndex /* = -1 */)
{

  try {
    checkGetAccess(label, matlIndex, 0); 
    d_lvlock.readLock();
    ReductionVariableBase* var = dynamic_cast<ReductionVariableBase*>(d_levelDB.get(label, matlIndex, level));
    d_lvlock.readUnlock();
    var->print(intout);
  } catch (UnknownVariable) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex,
                          "on emit reduction", __FILE__, __LINE__));
  }
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::deleteParticles(ParticleSubset* delset)
{
  int matlIndex = delset->getMatlIndex();
  Patch* patch = (Patch*) delset->getPatch();
  const Patch* realPatch = (patch != 0) ? patch->getRealPatch() : 0;

  d_pslock.writeLock();
  psetDBType::key_type key(realPatch, matlIndex, getID());
  psetDBType::iterator iter = d_delsetDB.find(key);
  ParticleSubset* currentDelset;
  if(iter != d_delsetDB.end()) { //update existing delset
    //    SCI_THROW(InternalError("deleteParticles called twice for patch", __FILE__, __LINE__));
    // Concatenate the delsets into the delset that already exists in the DB.
    currentDelset = iter->second;
    for (ParticleSubset::iterator d=delset->begin(); d != delset->end(); d++)
      currentDelset->addParticle(*d);
   
    d_delsetDB.erase(key);
    d_delsetDB.insert(pair<psetDBType::key_type,ParticleSubset*>(key,currentDelset));

    delete delset;

  } else {
    d_delsetDB.insert(pair<psetDBType::key_type,ParticleSubset*>(key,delset));
    delset->addReference();
  }
  d_pslock.writeUnlock();
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::addParticles(const Patch* patch, 
                                    int matlIndex,
                                    map<const VarLabel*, 
                                    ParticleVariableBase*>* addedState)
{
 d_pslock.writeLock();
  psetAddDBType::key_type key(matlIndex, patch);
  psetAddDBType::iterator iter = d_addsetDB.find(key);
  if(iter  != d_addsetDB.end()) 
    // SCI_THROW(InternalError("addParticles called twice for patch", __FILE__, __LINE__));
    cerr << "addParticles called twice for patch" << endl;
  
  else
    d_addsetDB[key]=addedState;
  
 d_pslock.writeUnlock();
}
//______________________________________________________________________
//
int
OnDemandDataWarehouse::decrementScrubCount(const VarLabel* var, 
                                           int matlIndex,
                                                const Patch* patch)
{

  int count = 0;
  switch(var->typeDescription()->getType()){
  case TypeDescription::NCVariable:
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
  case TypeDescription::PerPatch:
    //try {
      d_lock.readLock();
      count = d_varDB.decrementScrubCount(var, matlIndex, patch);
      d_lock.readUnlock();
      //}
      //catch (AssertionFailed& e) {
      //cout << d_myworld->myrank() << " DW " << getID() << " caught exception.\n";
      //throw e;
      //}
    break;
  case TypeDescription::ParticleVariable:
    d_pvlock.readLock();
    count = d_pvarDB.decrementScrubCount(var, matlIndex, patch);
    d_pvlock.readUnlock();
    break;
  case TypeDescription::SoleVariable:
    SCI_THROW(InternalError("decrementScrubCount called for sole variable: "+var->getName(), __FILE__, __LINE__));
  case TypeDescription::ReductionVariable:
    SCI_THROW(InternalError("decrementScrubCount called for reduction variable: "+var->getName(), __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("decrementScrubCount for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
  }
  return count;
}
//______________________________________________________________________
//
DataWarehouse::ScrubMode
OnDemandDataWarehouse::setScrubbing(ScrubMode scrubMode)
{
  ScrubMode oldmode = d_scrubMode;
  d_scrubMode = scrubMode;
  return oldmode;
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::setScrubCount(const VarLabel* var, 
                                     int matlIndex,
                                     const Patch* patch, 
                                     int count)
{
  switch(var->typeDescription()->getType()){
  case TypeDescription::NCVariable:
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
  case TypeDescription::PerPatch:
    d_lock.readLock();
    d_varDB.setScrubCount(var, matlIndex, patch, count);
    d_lock.readUnlock();
    break;
  case TypeDescription::ParticleVariable:
    d_pvlock.readLock();
    d_pvarDB.setScrubCount(var, matlIndex, patch, count);
    d_pvlock.readUnlock();
    break;
  case TypeDescription::SoleVariable:
    SCI_THROW(InternalError("setScrubCount called for sole variable: "+var->getName(), __FILE__, __LINE__));
  case TypeDescription::ReductionVariable:
    // Reductions are not scrubbed
    SCI_THROW(InternalError("setScrubCount called for reduction variable: "+var->getName(), __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("setScrubCount for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
  }
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::scrub(const VarLabel* var, 
                             int matlIndex,
                             const Patch* patch)
{
  switch(var->typeDescription()->getType()){
  case TypeDescription::NCVariable:
  case TypeDescription::CCVariable:
  case TypeDescription::SFCXVariable:
  case TypeDescription::SFCYVariable:
  case TypeDescription::SFCZVariable:
  case TypeDescription::PerPatch:
    d_lock.readLock();
    d_varDB.scrub(var, matlIndex, patch);
    d_lock.readUnlock();
    break;
  case TypeDescription::ParticleVariable:
    d_pvlock.readLock();
    d_pvarDB.scrub(var, matlIndex, patch);
    d_pvlock.readUnlock();
    break;
  case TypeDescription::SoleVariable:
    SCI_THROW(InternalError("scrub called for sole variable: "+var->getName(), __FILE__, __LINE__));
  case TypeDescription::ReductionVariable:
    // Reductions are not scrubbed
    SCI_THROW(InternalError("scrub called for reduction variable: "+var->getName(), __FILE__, __LINE__));
  default:
    SCI_THROW(InternalError("scrub for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
  }
}
//______________________________________________________________________
//
void
OnDemandDataWarehouse::initializeScrubs(int dwid, 
        const FastHashTable<ScrubItem>* scrubcounts, bool add)
{
  d_lock.readLock();
  d_varDB.initializeScrubs(dwid, scrubcounts, add);
  d_lock.readUnlock();
  d_plock.readLock();
  d_pvarDB.initializeScrubs(dwid, scrubcounts, add);
  d_plock.readUnlock();
}
//______________________________________________________________________
//
void 
OnDemandDataWarehouse::getGridVar(GridVariableBase& var, 
                                   const VarLabel* label, 
                                   int matlIndex, 
                                   const Patch* patch,
                                   Ghost::GhostType gtype, 
                                   int numGhostCells)
{
  TAU_PROFILE("OnDemandDataWarehouse::getGridVar", " ", TAU_USER);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);
  ASSERTEQ(basis,Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));  

  if(!d_varDB.exists(label, matlIndex, patch)) {
    //print();
    cout << d_myworld->myrank() << " unable to find variable '" << label->getName() << " on patch: " << patch->getID() << " matl: " << matlIndex << endl;
    //WAIT_FOR_DEBUGGER();
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
  }
  if(patch->isVirtual()){
    d_varDB.get(label, matlIndex, patch->getRealPatch(), var);
    var.offsetGrid(patch->getVirtualOffset());
  } else {
    d_varDB.get(label, matlIndex, patch, var);
  }


  IntVector low = patch->getExtraLowIndex(basis, label->getBoundaryLayer());
  IntVector high = patch->getExtraHighIndex(basis, label->getBoundaryLayer());

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
                                  lowIndex, highIndex);
    
    if (numGhostCells > 0)
      patch->getLevel()->selectPatches(lowIndex, highIndex, neighbors);
    else
      neighbors.push_back(patch);
    IntVector oldLow = var.getLow(), oldHigh = var.getHigh();
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
        //warned = true;
        //static ProgressiveWarning rw("Warning: Reallocation needed for ghost region you requested.\nThis means the data you get back will be a copy of what's in the DW", 100);
        //if (rw.invoke()) {
          // print out this message if the ProgressiveWarning does
          /*ostringstream errmsg;
                    errmsg << d_myworld->myrank() << " This occurrence for " << label->getName(); 
          if (patch)
            errmsg << " on patch " << patch->getID();
          errmsg << " for material " << matlIndex;

          errmsg << ".  Old range: " << oldLow << " " << oldHigh << " - new range " << lowIndex << " " << highIndex << " NGC " << numGhostCells;
          warn << errmsg.str() << '\n';*/
          
        //}
      }
    }

    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      if(neighbor && (neighbor != patch)){
        IntVector low = Max(neighbor->getExtraLowIndex(basis, label->getBoundaryLayer()), lowIndex);
        IntVector high = Min(neighbor->getExtraHighIndex(basis, label->getBoundaryLayer()), highIndex);

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
    
    vector<Variable*> varlist;
    d_varDB.getlist(label, matlIndex, neighbor, varlist);
    GridVariableBase* v=NULL;

    for (vector<Variable*>::reverse_iterator rit = varlist.rbegin(); ; ++rit){
      if (rit == varlist.rend()) {
        v = NULL;
        break;
      }

      v = dynamic_cast<GridVariableBase*>(*rit);
      //verify that the variable is valid and matches the depedencies requirements
      if((v!=NULL) && (v->isValid()))
      {
        if(neighbor->isVirtual()){
          if (Min(v->getLow(), low-neighbor->getVirtualOffset()) == v->getLow()  && Max(v->getHigh(),high-neighbor->getVirtualOffset()) == v->getHigh())  
          {
            break;
          }
        } else {
          if (Min(v->getLow(), low) == v->getLow()  && Max(v->getHigh(),high) == v->getHigh())  
          {
            break;
          }
        }
      }
    } //end for vars
    if (v==NULL) {
     // cout << d_myworld->myrank()  << " cannot copy var " << *label << " from patch " << neighbor->getID()
       // << " " << low << " " << high <<  ", DW has " << srcvar->getLow() << " " << srcvar->getHigh() << endl;
            SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor,
                                    matlIndex, neighbor == patch?
                                      "on patch":"on neighbor", __FILE__, __LINE__));
    }
          
    GridVariableBase* srcvar = var.cloneType();
    srcvar->copyPointer(*v);

    if(neighbor->isVirtual())
              srcvar->offsetGrid(neighbor->getVirtualOffset());

    try {
       var.copyPatch(srcvar, low, high);
    } catch (InternalError& e) {
        cout << " Bad range: " << low << " " << high  
        << " source var range: "  << srcvar->getLow() << " " << srcvar->getHigh() << endl;
       throw e;
    }
    delete srcvar;
    dn = high-low;
    total+=dn.x()*dn.y()*dn.z();
    } //end if neigbor
  } //end for neigbours 
    
    //dn = highIndex - lowIndex;
    //long wanted = dn.x()*dn.y()*dn.z();
    //ASSERTEQ(wanted, total);
  }
}
//______________________________________________________________________
//
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
          d_lock.writeLock();
          if(!fromDW->d_varDB.exists(var, matl, patch))
            SCI_THROW(UnknownVariable(var->getName(), fromDW->getID(), patch, matl,
                                      "in transferFrom", __FILE__, __LINE__));
          GridVariableBase* v = dynamic_cast<GridVariableBase*>(fromDW->d_varDB.get(var, matl, patch))->clone();
          d_varDB.put(var, matl, copyPatch, v, replace);
          d_lock.writeUnlock();
        }
        break;
      case TypeDescription::ParticleVariable:
        {
          d_plock.writeLock();
          if(!fromDW->d_pvarDB.exists(var, matl, patch))
            SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
                                      "in transferFrom", __FILE__, __LINE__));

          // or else the readLock in haveParticleSubset will hang
          ParticleSubset* subset;
          if (!haveParticleSubset(matl, copyPatch)) {
            ParticleSubset* oldsubset = fromDW->getParticleSubset(matl, patch);
            subset = createParticleSubset(oldsubset->numParticles(), matl, copyPatch);
          }
          else
            subset = getParticleSubset(matl, copyPatch);
          ParticleVariableBase* v = dynamic_cast<ParticleVariableBase*>(fromDW->d_pvarDB.get(var, matl, patch));
          if (patch == copyPatch)
            d_pvarDB.put(var, matl, copyPatch, v->clone(), replace);
          else {
            ParticleVariableBase* newv = v->cloneType();
            newv->copyPointer(*v);
            newv->setParticleSubset(subset);
            d_pvarDB.put(var, matl, copyPatch, newv, replace);
          }
          d_pvlock.writeUnlock();
        }
        break;
      case TypeDescription::PerPatch:
        {
          d_lock.writeLock();
          if(!fromDW->d_varDB.exists(var, matl, patch))
            SCI_THROW(UnknownVariable(var->getName(), getID(), patch, matl,
                                      "in transferFrom", __FILE__, __LINE__));
          PerPatchBase* v = dynamic_cast<PerPatchBase*>(fromDW->d_varDB.get(var, matl, patch));
          d_varDB.put(var, matl, copyPatch, v->clone(), replace);
          d_lock.writeUnlock();
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
}
//______________________________________________________________________
//
void OnDemandDataWarehouse::logMemoryUse(ostream& out, 
                                         unsigned long& total,
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
//______________________________________________________________________
//
inline void
OnDemandDataWarehouse::checkGetAccess(const VarLabel* label,
                                      int matlIndex, 
                                      const Patch* patch,
                                      Ghost::GhostType gtype,
                                      int numGhostCells)
{
#if 1
#if SCI_ASSERTION_LEVEL >= 1
  list<RunningTaskInfo>* runningTasks = getRunningTasksInfo();
  
  if (runningTasks != 0) {
    for (list<RunningTaskInfo>::iterator iter = runningTasks->begin();
        iter != runningTasks->end(); iter++) {
      RunningTaskInfo& runningTaskInfo = *iter;
      
      //   RunningTaskInfo& runningTaskInfo = runningTasks->back();
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
      findIter = runningTaskAccesses.find(VarLabelMatl<Patch>(label, matlIndex, patch));
      

      if (!hasGetAccess(runningTask, label, matlIndex, patch, lowOffset,highOffset,&runningTaskInfo) &&
          !hasPutAccess(runningTask, label, matlIndex, patch, true) &&
          !hasPutAccess(runningTask, label, matlIndex, patch, false)) {

        // If it was accessed by the current task already, then it should
        // have get access (i.e. if you put it in, you should be able to get it
        // right back out).
        if (findIter != runningTaskAccesses.end() &&
            lowOffset  == IntVector(0, 0, 0) && 
            highOffset == IntVector(0, 0, 0)){
          // allow non ghost cell get if any access (get, put, or modify) is allowed
          //cout << "allowing non-ghost cell access\n";
          return;
        }
        
        if (runningTask == 0 ||!(string(runningTask->getName()) == "Relocate::relocateParticles" ||
                                 string(runningTask->getName()) == "SchedulerCommon::copyDataToNewGrid")){
          string has;
          switch (getWhichDW(&runningTaskInfo))
          {
            case Task::NewDW:
              has="Task::NewDW";
              break;
            case Task::OldDW:
              has="Task::OldDW";
              break;
            case Task::ParentNewDW:
              has="Task::ParentNewDW";
              break;
            case Task::ParentOldDW:
              has="Task::ParentOldDW";
              break;
            default:
              has="UnknownDW";
          }
          has += " datawarehouse get";
          
          if (numGhostCells > 0) {
            ostringstream ghost_str;
            ghost_str << " for " << numGhostCells << " layer";
            
            if (numGhostCells > 1){
             ghost_str << "s";
            }
            ghost_str << " of ghosts around " << Ghost::getGhostTypeName(gtype);
            has += ghost_str.str();
          }     
          string needs = "task requires";
#if 1
          SCI_THROW(DependencyException(runningTask, label, matlIndex, patch,
                has, needs, __FILE__, __LINE__));
#else
          if ( d_myworld->myrank()  == 0 ){
            cout << DependencyException::makeMessage(runningTask, label, matlIndex, patch,has, needs) << endl;
            //WAIT_FOR_DEBUGGER();
          }
#endif    
        }
      } else {
        // access granted
        if (findIter == runningTaskAccesses.end()) {
          AccessInfo& accessInfo =
            runningTaskAccesses[VarLabelMatl<Patch>(label, matlIndex, patch)];
          accessInfo.accessType = GetAccess;
          accessInfo.encompassOffsets(lowOffset, highOffset);

          int ID = 0;
          if( patch ){ 
            ID = patch->getID();
          }
          string varname = "noname";
          if( label ){
            varname = label->getName();
          }
         if( dbg.active() ) {
          cerrLock.lock();
          dbg << d_myworld->myrank() << " Task running is: " << runningTask->getName();
          dbg<< left;dbg.width(10);
          dbg<< "\t"<< varname;
          dbg<< left;dbg.width(10);
          dbg<< " \t on patch " << ID << " and matl: " << matlIndex << " has been gotten\n";
	  cerrLock.unlock();
	 }
        } else {
          findIter->second.encompassOffsets(lowOffset, highOffset);
        }
      }
    }
  }  // running task loop
#endif
#endif
}
//______________________________________________________________________
//
inline void
OnDemandDataWarehouse::checkPutAccess(const VarLabel* label, 
                                      int matlIndex,
                                      const Patch* patch, 
                                      bool replace)
{
#if 1
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
          switch (getWhichDW(&runningTaskInfo))
          {
            case Task::NewDW:
              has="Task::NewDW";
              break;
            case Task::OldDW:
              has="Task::OldDW";
              break;
            case Task::ParentNewDW:
              has="Task::ParentNewDW";
              break;
            case Task::ParentOldDW:
              has="Task::ParentOldDW";
              break;
            default:
              has="UnknownDW";
          }
          if (replace) {
            has += " datawarehouse put";
            needs = "task computes(replace)";
          }
          else {
            has += " datawarehouse put";
            needs = "task computes";
          }
#if 1
          SCI_THROW(DependencyException(runningTask, label, matlIndex,
                patch, has, needs, __FILE__, __LINE__));
#else
          if ( d_myworld->myrank()  == 0 )
           cout << DependencyException::makeMessage(runningTask, label, matlIndex, patch,
             has, needs) << endl;
          //WAIT_FOR_DEBUGGER();
#endif
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

//______________________________________________________________________
//
inline 
Task::WhichDW OnDemandDataWarehouse::getWhichDW(RunningTaskInfo *info)
{
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::NewDW,info))
    return Task::NewDW;
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::OldDW,info))
    return Task::OldDW;
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::ParentNewDW,info))
    return Task::ParentNewDW;
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::ParentOldDW,info))
    return Task::ParentOldDW;
  throw InternalError("Unknown DW\n",__FILE__,__LINE__);
}
//______________________________________________________________________
//
inline bool
OnDemandDataWarehouse::hasGetAccess(const Task* runningTask,
                                    const VarLabel* label, int matlIndex,
                                    const Patch* patch, IntVector lowOffset,
                                    IntVector highOffset,RunningTaskInfo *info)
{ 
  return runningTask->hasRequires(label, matlIndex, patch, lowOffset, highOffset, getWhichDW(info));
}
//______________________________________________________________________
//
inline
bool OnDemandDataWarehouse::hasPutAccess(const Task* runningTask,
                                         const VarLabel* label, int matlIndex,
                                         const Patch* patch, bool replace)
{
    return runningTask->hasComputes(label, matlIndex, patch);
}
//______________________________________________________________________
//
void OnDemandDataWarehouse::pushRunningTask(const Task* task,
                                            vector<OnDemandDataWarehouseP>* dws)
{
  ASSERT(task);
  d_runningTasks[Thread::self()->myid()].push_back(RunningTaskInfo(task, dws));
}
//______________________________________________________________________
//
void OnDemandDataWarehouse::popRunningTask()
{
  d_runningTasks[Thread::self()->myid()].pop_back();
}
//______________________________________________________________________
//
inline list<OnDemandDataWarehouse::RunningTaskInfo>*
OnDemandDataWarehouse::getRunningTasksInfo()
{
  if (d_runningTasks[Thread::self()->myid()].empty()) return 0;
  else return &d_runningTasks[Thread::self()->myid()];
}
//______________________________________________________________________
//
inline bool OnDemandDataWarehouse::hasRunningTask()
{
  if (d_runningTasks[Thread::self()->myid()].empty()) return false;
  else return true;
}
//______________________________________________________________________
//
inline OnDemandDataWarehouse::RunningTaskInfo*
OnDemandDataWarehouse::getCurrentTaskInfo()
{
  if (d_runningTasks[Thread::self()->myid()].empty()) return 0;
  else return &d_runningTasks[Thread::self()->myid()].back();
}
//______________________________________________________________________
//
DataWarehouse*
OnDemandDataWarehouse::getOtherDataWarehouse(Task::WhichDW dw, RunningTaskInfo* info)
{
  int dwindex = info->d_task->mapDataWarehouse(dw);
  DataWarehouse* result = (*info->dws)[dwindex].get_rep();
  return result;
}
//______________________________________________________________________
//
DataWarehouse*
OnDemandDataWarehouse::getOtherDataWarehouse(Task::WhichDW dw)
{
  RunningTaskInfo* info = getCurrentTaskInfo();
  int dwindex = info->d_task->mapDataWarehouse(dw);
  DataWarehouse* result = (*info->dws)[dwindex].get_rep();
  return result;
}
//______________________________________________________________________
//
void 
OnDemandDataWarehouse::checkTasksAccesses(const PatchSubset* /*patches*/,
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
//______________________________________________________________________
//
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

    constHandle<PatchSubset> patches =  dep->getPatchesUnderDomain(domainPatches);
    constHandle<MaterialSubset> matls = dep->getMaterialsUnderDomain(domainMatls);
    
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
 
    if( strcmp(currentTask->getName(), "Relocate::relocateParticles")==0 ){
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
          //WAIT_FOR_DEBUGGER();
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

          //WAIT_FOR_DEBUGGER();
          SCI_THROW(DependencyException(currentTask, label, matl, patch,
                                        has, needs, __FILE__, __LINE__));
        }
      }
    }
  }
}

//______________________________________________________________________
//
// For timestep abort/restart
bool
OnDemandDataWarehouse::timestepAborted()
{
  return aborted;
}
//__________________________________
//
bool
OnDemandDataWarehouse::timestepRestarted()
{
  return restart;
}
//__________________________________
//
void
OnDemandDataWarehouse::abortTimestep()
{
  // BJW - timestep aborting does not work in MPI - disabling until we get fixed.
  if (d_myworld->size() == 0) {
    aborted = true;
  }
}
//__________________________________
//
void
OnDemandDataWarehouse::restartTimestep()
{
  restart = true;
}
//__________________________________
//
void
OnDemandDataWarehouse::getVarLabelMatlLevelTriples( vector<VarLabelMatl<Level> >& vars ) const
{
  d_levelDB.getVarLabelMatlTriples(vars);
}

void
OnDemandDataWarehouse::print()
{
  cout << d_myworld->myrank() << " VARIABLES in DW " << getID() << "\n"
       << d_myworld->myrank() << " Variable Patch Material\n"
       << "  -----------------------\n";
  d_varDB.print(cout, d_myworld->myrank());
  d_levelDB.print(cout, d_myworld->myrank());  
}
