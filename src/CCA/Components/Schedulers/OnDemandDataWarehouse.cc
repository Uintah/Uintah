/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/DependencyException.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/TypeMismatchException.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PSPatchMatlGhost.h>
#include <Core/Malloc/Allocator.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/BufferInfo.h>
#include <Core/Parallel/CrowdMonitor.hpp>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ProgressiveWarning.h>

#ifdef HAVE_CUDA
  #include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
  #include <Core/Grid/Variables/GPUStencil7.h>
  #include <Core/Geometry/GPUVector.h>
  #include <Core/Util/DebugStream.h>
#endif

#include <climits>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace Uintah;


namespace Uintah {

  extern Dout g_mpi_dbg;

#ifdef HAVE_CUDA
  extern DebugStream gpudbg;
#endif

}


namespace {

  // Tags for each CrowdMonitor
  struct varDB_tag{};
  struct levelDB_tag{};
  struct psetDB_tag{};
  struct addsetDB_tag{};
  struct delsetDB_tag{};
  struct task_access_tag{};

  using  varDB_monitor         = Uintah::CrowdMonitor<varDB_tag>;
  using  levelDB_monitor       = Uintah::CrowdMonitor<levelDB_tag>;
  using  psetDB_monitor        = Uintah::CrowdMonitor<psetDB_tag>;
  using  addsetDB_monitor      = Uintah::CrowdMonitor<addsetDB_tag>;
  using  delsetDB_monitor      = Uintah::CrowdMonitor<delsetDB_tag>;
  using  task_access_monitor   = Uintah::CrowdMonitor<task_access_tag>;

  Dout  g_foreign_dbg(    "ForeignVariables"   , "OnDemandDataWarehouse", "report when foreign variable is added to DW" , false );
  Dout  g_dw_get_put_dbg( "OnDemandDW"         , "OnDemandDataWarehouse", "report general dbg info for OnDemandDW"      , false );
  Dout  g_particles_dbg(  "DWParticleExchanges", "OnDemandDataWarehouse", "report MPI particle exchanges (sends/recvs)" , false );
  Dout  g_check_accesses( "DWCheckTaskAccess"  , "OnDemandDataWarehouse", "report on task DW access checking (DBG-only)", false );
  Dout  g_warnings_dbg(   "DWWarnings"         , "OnDemandDataWarehouse", "report DW GridVar progressive warnings"      , false );

  Uintah::MasterLock g_running_tasks_lock{};

}

// we want a particle message to have a unique tag per patch/matl/batch/dest.
// we only have 32K message tags, so this will have to do.
//   We need this because the possibility exists (particularly with DLB) of
//   two messages with the same tag being sent from the same processor.  Even
//   if these messages are sent to different processors, they can get crossed in the mail
//   or one can overwrite the other.
#define PARTICLESET_TAG 0x4000|batch->messageTag

bool OnDemandDataWarehouse::s_combine_memory = false;


//______________________________________________________________________
//
OnDemandDataWarehouse::OnDemandDataWarehouse( const ProcessorGroup * myworld
                                            ,       Scheduler      * scheduler
                                            , const int              generation
                                            , const GridP          & grid
                                            , const bool             isInitializationDW /* = false */
                                            )
    : DataWarehouse( myworld, scheduler, generation )
    , m_grid{ grid }
    , m_is_initialization_DW{ isInitializationDW }
{
  doReserve();

  varLock = new Uintah::MasterLock{};

#ifdef HAVE_CUDA

  if (Uintah::Parallel::usingDevice()) {
    int numDevices;
    CUDA_RT_SAFE_CALL(cudaGetDeviceCount(&numDevices));

    for (int i = 0; i < numDevices; i++) {
      //those gpuDWs should only live host side.
      //Ideally these don't need to be created at all as a separate datawarehouse,
      //but could be contained within this datawarehouse

      GPUDataWarehouse* gpuDW = (GPUDataWarehouse*)malloc(sizeof(GPUDataWarehouse) - sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS);
      std::ostringstream out;
      out << "Host-side GPU DW";

      gpuDW->init(i, out.str());
      gpuDW->setDebug(gpudbg.active());
      d_gpuDWs.push_back(gpuDW);
    }
  }

#endif

}

//______________________________________________________________________
//
OnDemandDataWarehouse::~OnDemandDataWarehouse()
{
  clear();
  delete varLock;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::clear()
{
  {
    psetDB_monitor psetDB_lock{ Uintah::CrowdMonitor<psetDB_tag>::WRITER };

    for (psetDBType::const_iterator iter = m_pset_db.begin(); iter != m_pset_db.end(); ++iter) {
      if (iter->second->removeReference()) {
        delete iter->second;
      }
    }

    for (psetDBType::const_iterator iter = m_delset_DB.begin(); iter != m_delset_DB.end(); ++iter) {
      if (iter->second->removeReference()) {
        delete iter->second;
      }
    }

    for (psetAddDBType::const_iterator iter = m_addset_DB.begin(); iter != m_addset_DB.end(); ++iter) {
      std::map<const VarLabel*, ParticleVariableBase*>::const_iterator pvar_itr;
      for (pvar_itr = iter->second->begin(); pvar_itr != iter->second->end(); pvar_itr++) {
        delete pvar_itr->second;
      }
      delete iter->second;
    }
  }

  m_var_DB.clear();
  m_level_DB.clear();
  m_running_tasks.clear();


#ifdef HAVE_CUDA

  if (Uintah::Parallel::usingDevice()) {
    //clear out the host side GPU Datawarehouses.  This does NOT touch the task DWs.
    for (size_t i = 0; i < d_gpuDWs.size(); i++) {
      d_gpuDWs[i]->clear();
      d_gpuDWs[i]->cleanup();
      free(d_gpuDWs[i]);
      d_gpuDWs[i] = nullptr;
    }
  }

#endif
}


//______________________________________________________________________
//
bool
OnDemandDataWarehouse::isFinalized() const
{
  return m_finalized;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::finalize()
{
  m_var_DB.cleanForeign();
  m_finalized = true;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::unfinalize()
{
  // this is for processes that need to make small modifications to the DW after it has been finalized.
  m_finalized = false;
}

//__________________________________
//
void
OnDemandDataWarehouse::refinalize()
{
  m_finalized = true;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(       Variable * var
                          , const VarLabel * label
                          ,       int        matlIndex
                          , const Patch    * patch
                          )
{
  union {
      ReductionVariableBase * reduction;
      SoleVariableBase      * sole;
      PerPatchBase          * perpatch;
      ParticleVariableBase  * particle;
      GridVariableBase      * grid;
  } castVar;

  if ((castVar.reduction = dynamic_cast<ReductionVariableBase*>(var)) != nullptr) {
    put(*castVar.reduction, label, patch ? patch->getLevel() : nullptr, matlIndex);
  }
  else if ((castVar.sole = dynamic_cast<SoleVariableBase*>(var)) != nullptr) {
    put(*castVar.sole, label, patch ? patch->getLevel() : nullptr, matlIndex);
  }
  else if ((castVar.perpatch = dynamic_cast<PerPatchBase*>(var)) != nullptr) {
    put(*castVar.perpatch, label, matlIndex, patch);
  }
  else if ((castVar.particle = dynamic_cast<ParticleVariableBase*>(var)) != nullptr) {
    put(*castVar.particle, label);
  }
  else if ((castVar.grid = dynamic_cast<GridVariableBase*>(var)) != nullptr) {
    put(*castVar.grid, label, matlIndex, patch);
  }
  else {
    SCI_THROW(InternalError("Unknown Variable type", __FILE__, __LINE__));
  }
}

//
//______________________________________________________________________
void
OnDemandDataWarehouse::copyKeyDB( KeyDatabase<Patch> & varkeyDB
                                , KeyDatabase<Level> & levelkeyDB
                                )
{
  m_var_key_DB.merge( varkeyDB );
  m_level_key_DB.merge( levelkeyDB );
}

//
//______________________________________________________________________
void
OnDemandDataWarehouse::doReserve() {
  m_var_DB.doReserve(&m_var_key_DB);
  m_level_DB.doReserve(&m_level_key_DB);
}

//
//______________________________________________________________________
void
OnDemandDataWarehouse::get(       ReductionVariableBase & var
                          , const VarLabel              * label
                          , const Level                 * level     /* = nullptr */
                          ,       int                     matlIndex /* = -1 */
                          )
{
  checkGetAccess( label, matlIndex, nullptr );

  if( !m_level_DB.exists( label, matlIndex, level ) ) {
    proc0cout << "get() failed in dw: " << this << ", level: " << level << "\n";
    SCI_THROW( UnknownVariable(label->getName(), getID(), level, matlIndex, "on reduction", __FILE__, __LINE__) );
  }

  m_level_DB.get( label, matlIndex, level, var );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(       SoleVariableBase& var
                          , const VarLabel*         label
                          , const Level*            level     /* = nullptr */
                          ,       int               matlIndex /* = -1 */
                          )
{
  checkGetAccess( label, matlIndex, nullptr );

  if( !m_level_DB.exists( label, matlIndex, level ) ) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex, "on sole", __FILE__, __LINE__) );
  }

  m_level_DB.get( label, matlIndex, level, var );
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::exists( const VarLabel * label
                             ,       int        matlIndex
                             , const Patch    * patch
                             ) const
{
  if (patch && m_var_DB.exists(label, matlIndex, patch)) {
    return true;
  }

  // level-independent reduction vars can be stored with a null level
  if (m_level_DB.exists(label, matlIndex, patch ? patch->getLevel() : nullptr)) {
    return true;
  }

  return false;
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::exists( const VarLabel * label
                             ,       int        matlIndex
                             , const Level    * level
                             ) const
{
  if (level && m_level_DB.exists(label, matlIndex, level)) {
    return true;
  }

  return false;
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::exists( const VarLabel* label ) const
{
  {
    levelDB_monitor levelDB_lock{ Uintah::CrowdMonitor<levelDB_tag>::READER };

    // level-independent reduction vars can be stored with a null level
    if (m_level_DB.exists(label, -1, nullptr)) {
      return true;
    }
    else {
      return false;
    }
  }
}

//______________________________________________________________________
//
ReductionVariableBase*
OnDemandDataWarehouse::getReductionVariable( const VarLabel * label
                                           , int              matlIndex
                                           , const Level    * level
                                           ) const
{
  if (m_level_DB.exists(label, matlIndex, level)) {
    ReductionVariableBase* var = dynamic_cast<ReductionVariableBase*>(m_level_DB.get(label, matlIndex, level));
    return var;
  }
  else {
    return nullptr;
  }
}

#ifdef HAVE_CUDA

void
OnDemandDataWarehouse::uintahSetCudaDevice(int deviceNum) {
  //  CUDA_RT_SAFE_CALL( cudaSetDevice(deviceNum) );
}

int
OnDemandDataWarehouse::getNumDevices() {
  int numDevices = 0;

  if (Uintah::Parallel::usingDevice()) {
    numDevices = 1;
  }

  //if multiple devices are desired, use this:
   CUDA_RT_SAFE_CALL(cudaGetDeviceCount(&numDevices));

  return numDevices;
}

size_t
OnDemandDataWarehouse::getTypeDescriptionSize(const TypeDescription::Type& type) {
  switch(type){
    case TypeDescription::double_type : {
      return sizeof(double);
      break;
    }
    case TypeDescription::float_type : {
      return sizeof(float);
      break;
    }
    case TypeDescription::int_type : {
      return sizeof(int);
      break;
    }
    case TypeDescription::Stencil7 : {
      return sizeof(Stencil7);
      break;
    }
    case TypeDescription::Vector : {
      return sizeof(Vector);
      break;
    }
    default : {
      SCI_THROW(InternalError("OnDemandDataWarehouse::getTypeDescriptionSize unsupported GPU Variable base type: " + type, __FILE__, __LINE__));
    }
  }
}


GPUGridVariableBase*
OnDemandDataWarehouse::createGPUGridVariable(const TypeDescription::Type& type)
{
  //Note: For C++11, these should return a unique_ptr.
  GPUGridVariableBase* device_var = nullptr;
  switch(type){
    case TypeDescription::double_type : {
      device_var = new GPUGridVariable<double>();
      break;
    }
    case TypeDescription::float_type : {
      device_var = new GPUGridVariable<float>();
      break;
    }
    case TypeDescription::int_type : {
      device_var = new GPUGridVariable<int>();
      break;
    }
    case TypeDescription::Stencil7 : {
      device_var = new GPUGridVariable<GPUStencil7>();
      break;
    }
    case TypeDescription::Vector : {
      device_var = new GPUGridVariable<gpuVector>();
      break;
    }
    default : {
      SCI_THROW(InternalError("createGPUGridVariable, unsupported GPUGridVariable type: ", __FILE__, __LINE__));
    }
  }
  return device_var;
}


GPUPerPatchBase*
OnDemandDataWarehouse::createGPUPerPatch(const TypeDescription::Type& type)
{
  GPUPerPatchBase* device_var = nullptr;

  switch(type){
     case TypeDescription::double_type : {
       device_var = new GPUPerPatch<double>();
       break;
     }
     case TypeDescription::float_type : {
       device_var = new GPUPerPatch<float>();
       break;
     }
     case TypeDescription::int_type : {
       device_var = new GPUPerPatch<int>();
       break;
     }
     case TypeDescription::Stencil7 : {
       device_var = new GPUPerPatch<GPUStencil7>();
       break;
     }
     case TypeDescription::Vector : {
       device_var = new GPUPerPatch<gpuVector>();
       break;
     }
     default : {
       SCI_THROW(InternalError("createGPUPerPatch, unsupported GPUPerPatch type: ", __FILE__, __LINE__));
     }
   }

  return device_var;
}

GPUReductionVariableBase*
OnDemandDataWarehouse::createGPUReductionVariable(const TypeDescription::Type& type)
{
  GPUReductionVariableBase* device_var = nullptr;

  switch(type){
    case TypeDescription::double_type : {
     device_var = new GPUReductionVariable<double>();
     break;
    }
    case TypeDescription::float_type : {
     device_var = new GPUReductionVariable<float>();
     break;
    }
    case TypeDescription::int_type : {
     device_var = new GPUReductionVariable<int>();
     break;
    }
    case TypeDescription::Stencil7 : {
     device_var = new GPUReductionVariable<GPUStencil7>();
     break;
    }
    case TypeDescription::Vector : {
     device_var = new GPUReductionVariable<gpuVector>();
     break;
    }
    default : {
     SCI_THROW(InternalError("createGPUReductionVariable, unsupported GPUReductionVariable type: ", __FILE__, __LINE__));
    }
  }

  return device_var;
}

#endif


//______________________________________________________________________
//
void
OnDemandDataWarehouse::exchangeParticleQuantities(       DetailedTasks * dts
                                                 ,       LoadBalancer  * lb
                                                 , const VarLabel      * pos_var
                                                 ,       int             iteration
                                                 )
{
  // If this DW is being used for a time step recompute, then this
  // step has already been performed.
  if (m_exchange_particle_quantities == false) {
    return;
  }

  m_exchange_particle_quantities = false;

  ParticleExchangeVar& recvs = dts->getParticleRecvs();
  ParticleExchangeVar& sends = dts->getParticleSends();

  // need to be sized here, otherwise you risk reallocating the array after a send/recv has been posted
  std::vector<std::vector<int> > senddata( sends.size() ), recvdata( recvs.size() );

  std::vector<MPI_Request> sendrequests, recvrequests;

  int data_index = 0;
  for (auto iter = recvs.begin(); iter != recvs.end(); ++iter) {
    std::set<PSPatchMatlGhostRange>& r = iter->second;
    if (r.size() > 0) {
      recvdata[data_index].resize(r.size());

      if (g_particles_dbg.active()) {
        std::stringstream mesg;
        mesg << d_myworld->myRank() << " Posting PARTICLES receives for " << r.size() << " subsets from proc " << iter->first
             << " index " << data_index;
        DOUT(true, mesg.str());
      }

      MPI_Request req;
      Uintah::MPI::Irecv(&(recvdata[data_index][0]), r.size(), MPI_INT, iter->first, 16666, d_myworld->getComm(), &req);
      recvrequests.push_back(req);
      data_index++;
    }
  }

  data_index = 0;
  for( auto iter = sends.begin(); iter != sends.end(); iter++ ) {
    std::set<PSPatchMatlGhostRange>& s = iter->second;
    if( s.size() > 0 ) {
      std::vector<int>& data = senddata[data_index];
      data.resize( s.size() );
      int i = 0;
      for( auto siter = s.begin(); siter != s.end(); siter++, i++ ) {
        const PSPatchMatlGhostRange& pmg = *siter;
        if( (pmg.dwid_ == DetailedDep::FirstIteration && iteration > 0)
            || (pmg.dwid_ == DetailedDep::SubsequentIterations && iteration == 0) ) {
          // not used
          data[i] = -2;
        }
        else if( pmg.dwid_ == DetailedDep::FirstIteration && iteration == 0
            && lb->getOldProcessorAssignment( pmg.patch_ ) == iter->first ) {
          // signify that the recving proc already has this data.  Only use for the FirstIteration after a LB
          // send -1 rather than force the recving end above to iterate through its set
          data[i] = -1;
        }
        else {
          if( !m_var_DB.exists( pos_var, pmg.matl_, pmg.patch_ ) ) {
            std::cout << d_myworld->myRank() << "  Naughty: patch " << pmg.patch_->getID() << " matl "
                 << pmg.matl_ << " id " << pmg.dwid_ << std::endl;
            SCI_THROW( UnknownVariable(pos_var->getName(), getID(), pmg.patch_, pmg.matl_,
                                       "in exchangeParticleQuantities", __FILE__, __LINE__) );
          }
          // Make sure sendset is unique...
          ASSERT( !m_send_state.find_sendset( iter->first, pmg.patch_, pmg.matl_, pmg.low_, pmg.high_, d_generation ) );
          ParticleSubset* sendset = scinew ParticleSubset( 0, pmg.matl_, pmg.patch_, pmg.low_, pmg.high_ );
          constParticleVariable<Point> pos;
          get( pos, pos_var, pmg.matl_, pmg.patch_ );
          ParticleSubset* pset = pos.getParticleSubset();
          for( auto piter = pset->begin(); piter != pset->end(); ++piter ) {
            if( Patch::containsIndex( pmg.low_, pmg.high_, pmg.patch_->getCellIndex( pos[*piter] ) ) ) {
              sendset->addParticle( *piter );
            }
          }
          m_send_state.add_sendset( sendset, iter->first, pmg.patch_, pmg.matl_, pmg.low_, pmg.high_, d_generation );
          data[i] = sendset->numParticles();
        }

        // debug
        if (g_particles_dbg) {
          std::ostringstream mesg;
          mesg << d_myworld->myRank() << " Sending PARTICLES to proc " << iter->first
               << ": patch " << pmg.patch_->getID() << " matl " << pmg.matl_ << " low "
               << pmg.low_ << " high " << pmg.high_ << " index " << i << ": "
               << senddata[data_index][i] << " particles";
          DOUT(true, mesg.str());
        }

      }
      DOUT(g_particles_dbg, d_myworld->myRank() << " Sending PARTICLES: " << s.size() << " subsets to proc " << iter->first << " index " << data_index);

      MPI_Request req;
      Uintah::MPI::Isend( &(senddata[data_index][0]), s.size(), MPI_INT, iter->first, 16666, d_myworld->getComm(), &req );

      sendrequests.push_back( req );
      data_index++;
    }
  }

  Uintah::MPI::Waitall( recvrequests.size(), &recvrequests[0], MPI_STATUSES_IGNORE );
  Uintah::MPI::Waitall( sendrequests.size(), &sendrequests[0], MPI_STATUSES_IGNORE );

  // create particle subsets from recvs
  data_index = 0;
  for( auto iter = recvs.begin(); iter != recvs.end(); ++iter ) {
    std::set<PSPatchMatlGhostRange>& r = iter->second;
    if( r.size() > 0 ) {
      std::vector<int>& data = recvdata[data_index];
      int i = 0;
      for( auto riter = r.begin(); riter != r.end();
          riter++, i++ ) {
        const PSPatchMatlGhostRange& pmg = *riter;

        // debug
        if (g_particles_dbg) {
          std::ostringstream mesg;
          mesg << d_myworld->myRank() << " Recving PARTICLES from proc " << iter->first
               << ": patch " << pmg.patch_->getID() << " matl " << pmg.matl_ << " low "
               << pmg.low_ << " high " << pmg.high_ << ": " << data[i];

          DOUT(true, mesg.str());
        }

        if( data[i] == -2 ) {
          continue;
        }
        if( data[i] == -1 ) {
          ASSERT( pmg.dwid_ == DetailedDep::FirstIteration && iteration == 0 && haveParticleSubset( pmg.matl_, pmg.patch_ ) );
          continue;
        }

        int & foreign_particles = m_foreign_particle_quantities[std::make_pair( pmg.matl_, pmg.patch_ )];
        ParticleSubset* subset = createParticleSubset( data[i], pmg.matl_, pmg.patch_, pmg.low_,
                                                       pmg.high_ );

        // make room for other multiple subsets pointing into one variable - additional subsets
        // referenced at the index above the last index of the previous subset
        if( data[i] > 0 && foreign_particles > 0 ) {
          DOUTR(g_particles_dbg,  "  adjusting particles by " << foreign_particles);

          for( ParticleSubset::iterator iter = subset->begin(); iter != subset->end(); iter++ ) {
            *iter = *iter + foreign_particles;
          }
        }
        foreign_particles += data[i];

        DOUTR(g_particles_dbg, "  Setting foreign particles of patch " << pmg.patch_->getID()
                               << " matl " << pmg.matl_ << " foreign_particles" << foreign_particles);
      }
      data_index++;
    }
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::sendMPI(       DependencyBatch       * batch
                              , const VarLabel              * pos_var
                              ,       BufferInfo            & buffer
                              ,       OnDemandDataWarehouse * old_dw
                              , const DetailedDep           * dep
                              ,       LoadBalancer          * lb
                              )
{
  if (dep->isNonDataDependency()) {
    // A non-data dependency -- send an empty message.
    // This would be used, for example, when a task is to modify data that was previously required with ghost-cells.
    //buffer.add(0, 0, MPI_INT, false);
    return;
  }

  const VarLabel* label = dep->m_req->m_var;
  const Patch* patch = dep->m_from_patch;
  int matlIndex = dep->m_matl;

  switch ( label->typeDescription()->getType() ) {
    case TypeDescription::ParticleVariable : {

      IntVector low   = dep->m_low;
      IntVector high = dep->m_high;

      if (!m_var_DB.exists(label, matlIndex, patch)) {
        SCI_THROW( UnknownVariable(label->getName(), getID(), patch, matlIndex, "in sendMPI", __FILE__, __LINE__) );
      }

      ParticleVariableBase* var = dynamic_cast<ParticleVariableBase*>( m_var_DB.get( label, matlIndex, patch ) );

      int dest = batch->m_to_tasks.front()->getAssignedResourceIndex();
      ASSERTRANGE( dest, 0, d_myworld->nRanks() );

      ParticleSubset* sendset = nullptr;
      // first check to see if the receiving proc already has the (old) data
      // if data is relocating (of a regrid or re-load-balance), then the other
      // proc may already have it (since in most cases particle data comes from the old dw)
      // if lb is non-null, that means the particle data is on the old dw
      if (lb && lb->getOldProcessorAssignment(patch) == dest) {
        if (this == old_dw) {
          // We don't need to know how many particles there are OR send any particle data...
          return;
        }
        ASSERT(old_dw->haveParticleSubset(matlIndex, patch));
        sendset = old_dw->getParticleSubset(matlIndex, patch);
      }
      else {
        sendset = old_dw->m_send_state.find_sendset(dest, patch, matlIndex, low, high, old_dw->d_generation);
      }

      if (!sendset) {
        // new dw send.  The NewDW doesn't yet know (on the first time) about this subset if it is on a different process.
        // Go ahead and calculate it, but there is no need to send it, since the other proc already knows about it.
        ASSERT( old_dw != this );

        ParticleSubset* pset = var->getParticleSubset();
        sendset = scinew ParticleSubset( 0, matlIndex, patch, low, high );

        constParticleVariable<Point> pos;
        old_dw->get( pos, pos_var, pset );

        for( auto iter = pset->begin(); iter != pset->end(); ++iter ) {
          particleIndex idx = *iter;
          if( Patch::containsIndex( low, high, patch->getCellIndex( pos[idx] ) ) ) {
            sendset->addParticle( idx );
          }
        }

        old_dw->m_send_state.add_sendset( sendset, dest, patch, matlIndex, low, high, old_dw->d_generation );
        DOUTR(g_particles_dbg, "  NO SENDSET: posVarLabel: " << pos_var->getName() << " Patch: " << patch->getID()
                               << " matl " << matlIndex << " "
                               << low << " " << high
                               << " dest: " << dest <<"\n    " << *sendset);
        old_dw->m_send_state.print();
      }

      ASSERT( sendset );
      if( sendset->numParticles() > 0 ) {
        var->getMPIBuffer( buffer, sendset );
        buffer.addSendlist( var->getRefCounted() );
        buffer.addSendlist( var->getParticleSubset() );
      }
      break;
    }
    case TypeDescription::NCVariable :
    case TypeDescription::CCVariable :
    case TypeDescription::SFCXVariable :
    case TypeDescription::SFCYVariable :
    case TypeDescription::SFCZVariable : {
      if (!m_var_DB.exists(label, matlIndex, patch)) {
        DOUT(true, d_myworld->myRank() << "  Needed by " << *dep << " on task " << *dep->m_to_tasks.front());
        SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "in Task OnDemandDataWarehouse::sendMPI", __FILE__, __LINE__));
      }
      GridVariableBase* var;
      var = dynamic_cast<GridVariableBase*>( m_var_DB.get( label, matlIndex, patch ) );
      var->getMPIBuffer( buffer, dep->m_low, dep->m_high );
      buffer.addSendlist( var->getRefCounted() );
      break;
    }
    case TypeDescription::PerPatch :
    case TypeDescription::ReductionVariable :
    case TypeDescription::SoleVariable :
    default : {
      SCI_THROW( InternalError("sendMPI not implemented for " + label->getFullName(matlIndex, patch), __FILE__, __LINE__) );
    }
  }  // end switch( label->getType() );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::recvMPI(       DependencyBatch       * batch
                              ,       BufferInfo            & buffer
                              ,       OnDemandDataWarehouse * old_dw
                              , const DetailedDep           * dep
                              ,       LoadBalancer          * lb
                              )
{
  if (dep->isNonDataDependency()) {
    // A non-data dependency -- send an empty message.
    // This would be used, for example, for dependencies between a modifying
    // task and a task the requires the data before it is to be modified.

//     // Is this needed? APH, 11/28/18
//    buffer.add(nullptr, 0, MPI_INT, false);

    return;
  }

  const VarLabel* label = dep->m_req->m_var;
  const Patch* patch = dep->m_from_patch;
  int matlIndex = dep->m_matl;
  int my_rank = d_myworld->myRank();

  switch ( label->typeDescription()->getType() ) {
    case TypeDescription::ParticleVariable : {
      IntVector low = dep->m_low;
      IntVector high = dep->m_high;
      bool whole_patch_pset = false;

      // First, get the particle set.  We should already have it
      //      if(!old_dw->haveParticleSubset(matlIndex, patch, gt, ngc)){
      //
      // if we already have a subset for the entire patch, there's little point
      // in getting another one (and if we did, it would cause synchronization problems - see
      // comment in sendMPI)

      ParticleSubset* recvset = nullptr;
      if( lb && (lb->getOldProcessorAssignment( patch ) == my_rank || lb->getPatchwiseProcessorAssignment( patch ) == my_rank) ) {
        // first part of the conditional means "we used to own the ghost data so use the same particles"
        // second part means "we were just assigned to this patch and need to receive the whole thing"
        // we will never get here if they are both true, as MPI wouldn't need to be scheduled
        ASSERT( old_dw->haveParticleSubset( matlIndex, patch ) );
        recvset = old_dw->getParticleSubset( matlIndex, patch );
        whole_patch_pset = true;
      }
      else {
        recvset = old_dw->getParticleSubset( matlIndex, patch, low, high );
      }
      ASSERT( recvset );

      ParticleVariableBase* var = nullptr;
      if (m_var_DB.exists(label, matlIndex, patch)) {
        var = dynamic_cast<ParticleVariableBase*>( m_var_DB.get( label, matlIndex, patch ) );
        ASSERT( var->isForeign() )
      }
      else {
        var = dynamic_cast<ParticleVariableBase*>( label->typeDescription()->createInstance() );
        ASSERT( var != nullptr );
        var->setForeign();

        // set the foreign before the allocate (allocate CAN take multiple P Subsets, but only if it's foreign)
        if( whole_patch_pset ) {
          var->allocate( recvset );
        }
        else {
          // don't give this a pset as it could be a container for several
          int allocated_particles = old_dw->m_foreign_particle_quantities[std::make_pair( matlIndex, patch )];
          var->allocate( allocated_particles );
        }
        m_var_DB.put( label, matlIndex, patch, var, d_scheduler->copyTimestep(), true );
      }

      if( recvset->numParticles() > 0 && !(lb && lb->getOldProcessorAssignment( patch ) == d_myworld->myRank()
                                      && this == old_dw) ) {
        var->getMPIBuffer( buffer, recvset );
      }
      break;
    }
    case TypeDescription::NCVariable :
    case TypeDescription::CCVariable :
    case TypeDescription::SFCXVariable :
    case TypeDescription::SFCYVariable :
    case TypeDescription::SFCZVariable : {

      //allocate the variable
      GridVariableBase* var = dynamic_cast<GridVariableBase*>( label->typeDescription()->createInstance() );
      var->allocate( dep->m_low, dep->m_high );

      //set the var as foreign
      var->setForeign();
      var->setInvalid();

      // add the var to the dependency batch and set it as invalid.  The variable is now invalid because there is outstanding MPI pointing to the variable.
      batch->addVar( var );
      IntVector low, high, size;
      var->getSizes(low, high, size);

      DOUT( g_foreign_dbg, "Rank-" << Parallel::getMPIRank() << "  adding foreign var: "
                                   << std::setw(10) << *label << "  patch: "
                                   << patch->getID() << "  matl: " << matlIndex << "  level: " << patch->getLevel()->getIndex()
                                   << "  from proc: " << lb->getPatchwiseProcessorAssignment( patch )
                                   << "  low: " << low << "  high: " << high << " sizes: " << size
                                   << "  num ghost cells: " << dep->m_req->m_num_ghost_cells);

      m_var_DB.putForeign( label, matlIndex, patch, var, d_scheduler->copyTimestep() );  //put new var in data warehouse
      var->getMPIBuffer( buffer, dep->m_low, dep->m_high );

      break;
    }
    case TypeDescription::PerPatch :
    case TypeDescription::ReductionVariable :
    case TypeDescription::SoleVariable :
    default : {
      SCI_THROW( InternalError("recvMPI not implemented for "+label->getFullName(matlIndex, patch), __FILE__, __LINE__) );
    }
  }  // end switch( label->getType() );
}  // end recvMPI()

//______________________________________________________________________
//
void
OnDemandDataWarehouse::reduceMPI( const VarLabel       * label
                                , const Level          * level
                                , const MaterialSubset * inmatls
                                , const int              nComm
                                )
{
  const MaterialSubset* matls;
  if( !inmatls ) {
    MaterialSubset* tmpmatls = scinew MaterialSubset();
    tmpmatls->add( -1 );
    matls = tmpmatls;
  }
  else {
    matls = inmatls;
  }

  // Count the number of data elements in the reduction array
  int nmatls = matls->size();
  int count = 0;
  MPI_Op op = MPI_OP_NULL;
  MPI_Datatype datatype = MPI_DATATYPE_NULL;

  for( int m = 0; m < nmatls; m++ ) {

    int matlIndex = matls->get( m );

    ReductionVariableBase* var;

    if( m_level_DB.exists( label, matlIndex, level ) ) {
      var = dynamic_cast<ReductionVariableBase*>( m_level_DB.get( label, matlIndex, level ) );
    }
    else {
      //  Create and initialize the variable if it doesn't exist
      var = dynamic_cast<ReductionVariableBase*>( label->typeDescription()->createInstance() );
      var->setBenignValue();

      DOUT(g_mpi_dbg, "Rank-" << d_myworld->myRank() << " reduceMPI: initializing (" <<label->getName() <<")" );
      m_level_DB.put( label, matlIndex, level, var, d_scheduler->copyTimestep(), true );
    }

    int sendcount;
    MPI_Datatype senddatatype = MPI_DATATYPE_NULL;
    MPI_Op sendop = MPI_OP_NULL;
    var->getMPIInfo( sendcount, senddatatype, sendop );
    if( m == 0 ) {
      op = sendop;
      datatype = senddatatype;
    }
    else {
      ASSERTEQ( op, sendop );
      ASSERTEQ( datatype, senddatatype );
    }
    count += sendcount;
  }

  //__________________________________
  //
  int packsize;
  Uintah::MPI::Pack_size( count, datatype, d_myworld->getGlobalComm( nComm ), &packsize );
  std::vector<char> sendbuf( packsize );

  int packindex = 0;
  for( int m = 0; m < nmatls; m++ ) {
    int matlIndex = matls->get( m );

    ReductionVariableBase* var;
    try {
      var = dynamic_cast<ReductionVariableBase*>( m_level_DB.get( label, matlIndex, level ) );
    }
    catch( UnknownVariable& ) {
      SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex, "on reduceMPI(pass 2)", __FILE__, __LINE__) );
    }
    var->getMPIData( sendbuf, packindex );
  }

  std::vector<char> recvbuf( packsize );

  DOUTR(g_mpi_dbg,  " allreduce, name " << label->getName()
       << " sendbuf.size() " << sendbuf.size() << " level " << (level ? level->getID() : -1));

  int error = Uintah::MPI::Allreduce( &sendbuf[0], &recvbuf[0], count, datatype, op, d_myworld->getGlobalComm( nComm ) );

  DOUTR(g_mpi_dbg,  " allreduce, done " << label->getName()
      << " recvbuf.size() " << recvbuf.size() << " level " << (level ? level->getID() : -1));

  if( error ) {
    DOUT(true, "reduceMPI: Uintah::MPI::Allreduce error: " << error);
    SCI_THROW( InternalError("reduceMPI: MPI error", __FILE__, __LINE__) );
  }

  int unpackindex = 0;
  for( int m = 0; m < nmatls; m++ ) {
    int matlIndex = matls->get( m );

    ReductionVariableBase* var;
    try {
      var = dynamic_cast<ReductionVariableBase*>( m_level_DB.get( label, matlIndex, level ) );
    }
    catch( UnknownVariable& ) {
      SCI_THROW(UnknownVariable(label->getName(), getID(), level, matlIndex, "on reduceMPI(pass 2)", __FILE__, __LINE__) );
    }
    var->putMPIData( recvbuf, unpackindex );
  }

  if( matls != inmatls ) {
    delete matls;
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::put( const ReductionVariableBase & var
                          , const VarLabel              * label
                          , const Level                 * level     /* = nullptr */
                          ,       int                     matlIndex /* = -1 */
                          )
{
  ASSERT( !m_finalized );

  // it actually may be replaced, but it doesn't need to explicitly modify with multiple ReductionVars in the task graph
  checkPutAccess( label, matlIndex, nullptr, false);

  printDebuggingPutInfo( label, matlIndex, level, __LINE__ );

  // Put it in the database
  bool init = (d_scheduler->copyTimestep()) || !(m_level_DB.exists( label, matlIndex, level ));
  m_level_DB.putReduce( label, matlIndex, level, var.clone(), init );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::override( const ReductionVariableBase & var
                               , const VarLabel              * label
                               , const Level                 * level     /* = nullptr */
                               ,       int                     matlIndex /* = -1 */
                               )
{
  checkPutAccess( label, matlIndex, nullptr, true );

  printDebuggingPutInfo( label, matlIndex, level, __LINE__ );

  // Put it in the database, replace whatever may already be there
  m_level_DB.put( label, matlIndex, level, var.clone(), true, true );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::override( const SoleVariableBase & var
                               , const VarLabel         * label
                               , const Level            * level     /* = nullptr */
                               ,       int                matlIndex /* = -1 */
                               )
{
  checkPutAccess(label, matlIndex, nullptr, true);

  printDebuggingPutInfo( label, matlIndex, level, __LINE__ );

  // Put it in the database, replace whatever may already be there
  m_level_DB.put(label, matlIndex, level, var.clone(), d_scheduler->copyTimestep(), true);
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::put( const SoleVariableBase & var
                          , const VarLabel         * label
                          , const Level            * level     /* = nullptr */
                          ,       int                matlIndex /* = -1 */
                          )
{
  ASSERT(!m_finalized);

  // it actually may be replaced, but it doesn't need to explicitly modify with multiple SoleVars in the task graph
  checkPutAccess(label, matlIndex, nullptr, false);

  printDebuggingPutInfo( label, matlIndex, level, __LINE__ );

  // Put it in the database
  if (!m_level_DB.exists(label, matlIndex, level)) {
    m_level_DB.put(label, matlIndex, level, var.clone(), d_scheduler->copyTimestep(), false);
  }
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::createParticleSubset(       particleIndex   numParticles
                                           ,       int             matlIndex
                                           , const Patch         * patch
                                           ,       IntVector       low  /* = IntVector(0,0,0) */
                                           ,       IntVector       high /* = IntVector(0,0,0) */
                                           )
{
  if (low == high && high == IntVector(0, 0, 0)) {
    low = patch->getExtraCellLowIndex();
    high = patch->getExtraCellHighIndex();
  }

  if (g_dw_get_put_dbg.active()) {
    std::stringstream mesg;
    mesg << " DW ID " << getID() << " createParticleSubset: MI: " << matlIndex << " P: " << patch->getID()
         << " (" << low << ", " << high << ") size: " << numParticles << "\n";
    DOUTR(true, mesg.str());
  }

  ASSERT(!patch->isVirtual());

  ParticleSubset* psubset = scinew ParticleSubset( numParticles, matlIndex, patch, low, high );
  insertPSetRecord( m_pset_db, patch, low, high, matlIndex, psubset );

  return psubset;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::saveParticleSubset(       ParticleSubset * psubset
                                         ,       int              matlIndex
                                         , const Patch          * patch
                                         ,       IntVector        low  /* = IntVector(0,0,0) */
                                         ,       IntVector        high /* = IntVector(0,0,0) */
                                         )
{
  ASSERTEQ( psubset->getPatch(), patch );
  ASSERTEQ( psubset->getMatlIndex(), matlIndex );
  ASSERT( !patch->isVirtual() );

  if( low == high && high == IntVector( 0, 0, 0 ) ) {
    low = patch->getExtraCellLowIndex();
    high = patch->getExtraCellHighIndex();
  }

  if (g_dw_get_put_dbg.active()) {
    std::stringstream mesg;
    mesg << " DW ID " << getID() << " saveParticleSubset: MI: " << matlIndex
         << " P: " << patch->getID() << " (" << low << ", " << high << ") size: "
         << psubset->numParticles() << "\n";
    DOUTR(true, mesg.str());
  }

  insertPSetRecord( m_pset_db, patch, low, high, matlIndex, psubset );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::printParticleSubsets()
{
  DOUTR( true,  "----------------------------------------------");
  DOUTR( true,  "-- Particle Subsets: Available psets on DW " << d_generation << ":" );

  psetDBType::iterator iter;
  for (iter = m_pset_db.begin(); iter != m_pset_db.end(); iter++) {
    DOUTR( true, *(iter->second) );
  }
  DOUTR( true,   "----------------------------------------------" );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::insertPSetRecord(       psetDBType     & subsetDB
                                       , const Patch          * patch
                                       ,       IntVector        low
                                       ,       IntVector        high
                                       ,       int              matlIndex
                                       ,       ParticleSubset * psubset
                                       )
{
  psubset->setLow(low);
  psubset->setHigh(high);

#if SCI_ASSERTION_LEVEL >= 1
  ParticleSubset* subset=queryPSetDB(subsetDB,patch,matlIndex,low,high,0,true);
  if (subset != nullptr) {
    if (d_myworld->myRank() == 0) {
      DOUTR( true, "  Duplicate: " << patch->getID() << " matl " << matlIndex << " " << low << " " << high );
      printParticleSubsets();
    }
    SCI_THROW(InternalError("tried to create a particle subset that already exists", __FILE__, __LINE__));
  }
#endif

  {
    psetDB_monitor psetDB_lock{ Uintah::CrowdMonitor<psetDB_tag>::WRITER };

    psetDBType::key_type key(patch->getRealPatch(), matlIndex, getID());
    subsetDB.insert(std::pair<psetDBType::key_type, ParticleSubset*>(key, psubset));
    psubset->addReference();
  }
}
//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::queryPSetDB(       psetDBType & subsetDB
                                  , const Patch      * patch
                                  ,       int          matlIndex
                                  ,       IntVector    low
                                  ,       IntVector    high
                                  , const VarLabel   * pos_var
                                  ,       bool         exact /* = false */
                                  )
{
  ParticleSubset* subset = nullptr;

  psetDBType::key_type key(patch->getRealPatch(), matlIndex, getID());
  int best_volume = std::numeric_limits<int>::max();
  int target_volume = Region::getVolume(low,high);


  {
    psetDB_monitor psetDB_write_lock{ Uintah::CrowdMonitor<psetDB_tag>::WRITER };

    std::pair<psetDBType::const_iterator, psetDBType::const_iterator> ret = subsetDB.equal_range(key);

    // search multimap for best subset
    for (psetDBType::const_iterator iter = ret.first; iter != ret.second; ++iter) {

      ParticleSubset *ss = iter->second;
      IntVector sslow = ss->getLow();
      IntVector sshigh = ss->getHigh();
      int vol = Region::getVolume(sslow, sshigh);

      // check if volume is better than current best
      if (vol < best_volume) {
        // intersect ranges
        if (low.x() >= sslow.x() && low.y() >= sslow.y() && low.z() >= sslow.z() && sshigh.x() >= high.x() && sshigh.y() >= high.y()
            && sshigh.z() >= high.z()) {
          // take this range
          subset = ss;
          best_volume = vol;

          // short circuit out if we have already found the best possible solution
          if (best_volume == target_volume) {
            break;
          }
        }
      }
    }
  } // end psetDB_write_lock{ Uintah::CrowdMonitor<psetDB_tag>::WRITER }


  if (exact && best_volume != target_volume) {
    return nullptr;
  }

  // if we don't need to filter or we already have an exact match just return this subset
  if (pos_var == nullptr || best_volume == target_volume) {
    return subset;
  }

  // otherwise filter out particles that are not within this range
  constParticleVariable<Point> pos;

  ASSERT(subset != nullptr);

  get(pos, pos_var, subset);

  ParticleSubset* newsubset = scinew ParticleSubset(0, matlIndex, patch->getRealPatch(),low,high);

  for (ParticleSubset::iterator iter = subset->begin(); iter != subset->end(); iter++) {
    particleIndex idx = *iter;
    if (Patch::containsIndex(low, high, patch->getCellIndex(pos[idx]))) {
      newsubset->addParticle(idx);
    }
  }

  // save subset for future queries
  {
    psetDB_monitor psetDB_write_lock{ Uintah::CrowdMonitor<psetDB_tag>::WRITER };

    subsetDB.insert(std::pair<psetDBType::key_type, ParticleSubset*>(key, newsubset));
    newsubset->addReference();
  }

  return newsubset;
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(       int     matlIndex
                                        , const Patch * patch
                                        )
{
  return getParticleSubset(matlIndex, patch, patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(       int         matlIndex
                                        , const Patch     * patch
                                        ,       IntVector   low
                                        ,       IntVector   high
                                        )
{
  const Patch* realPatch = (patch != nullptr) ? patch->getRealPatch() : nullptr;
  ParticleSubset* subset = nullptr;

  subset = queryPSetDB( m_pset_db, realPatch, matlIndex, low, high, nullptr );

  // bulletproofing
  if (!subset) {
    printParticleSubsets();
    std::ostringstream s;
    s << "ParticleSubset, (low: " << low << ", high: " << high << " DWID " << getID() << ')';
    SCI_THROW(UnknownVariable(s.str().c_str(), getID(), realPatch, matlIndex, "Cannot find particle set on patch", __FILE__, __LINE__) );
  }
  return subset;
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(       int         matlIndex
                                        , const Patch     * patch
                                        ,       IntVector   low
                                        ,       IntVector   high
                                        , const VarLabel  * posvar
                                        )
{
  const Patch* realPatch = (patch != nullptr) ? patch->getRealPatch() : nullptr;
  ParticleSubset* subset = nullptr;

  subset = queryPSetDB( m_pset_db, realPatch, matlIndex, low, high, posvar );

  // bulletproofing
  if (subset == nullptr) {
    printParticleSubsets();
    std::ostringstream s;
    s << "ParticleSubset, (low: " << low << ", high: " << high << " DWID " << getID() << ')';
    SCI_THROW(UnknownVariable(s.str().c_str(), getID(), realPatch, matlIndex, "Cannot find particle set on patch", __FILE__, __LINE__) );
  }

  return subset;
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(       int                matlIndex
                                        , const Patch*             patch
                                        ,       Ghost::GhostType   gtype
                                        ,       int                numGhostCells
                                        , const VarLabel         * posvar
                                        )
{
  IntVector lowIndex, highIndex;
  patch->computeVariableExtents(Patch::CellBased, posvar->getBoundaryLayer(), gtype, numGhostCells, lowIndex, highIndex);

  if (gtype == Ghost::None || (lowIndex == patch->getExtraCellLowIndex() && highIndex == patch->getExtraCellHighIndex())) {
    return getParticleSubset(matlIndex, patch);
  }

  return getParticleSubset(matlIndex, lowIndex, highIndex, patch, posvar);
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getParticleSubset(       int        matlIndex
                                        ,       IntVector  lowIndex
                                        ,       IntVector  highIndex
                                        , const Patch    * relPatch
                                        , const VarLabel * posvar
                                        , const Level    * oldLevel /* = nullptr */ //level is ONLY used when querying from an old grid, otherwise the level will be determined from the patch
                                        )
{
  // relPatch can be nullptr if trying to get a particle subset for an arbitrary spot on the level
  Patch::selectType neighbors;

  ASSERT(relPatch != nullptr); //you should pass in the patch on which the task was called on
  const Level* level=relPatch->getLevel();

  //compute intersection between query range and patch
  IntVector low=Min(lowIndex,relPatch->getExtraCellLowIndex());
  IntVector high=Max(highIndex,relPatch->getExtraCellHighIndex());


  //if the user passed in the old level then query its patches
  if( oldLevel != nullptr ) {
    oldLevel->selectPatches( lowIndex, highIndex, neighbors );  //find all intersecting patches with the range
  }
  //if the query range is larger than the patch
  else if( low != relPatch->getExtraCellLowIndex() || high != relPatch->getExtraCellHighIndex() ) {
    level->selectPatches( lowIndex, highIndex, neighbors );  //find all intersecting patches with the range
  }
  else {
    //just add this patch, do not query the whole level
    neighbors.push_back( relPatch );
  }

  particleIndex totalParticles = 0;
  std::vector<ParticleVariableBase*> neighborvars;
  std::vector<ParticleSubset*> subsets;
  std::vector<const Patch*> vneighbors;

  for( size_t i = 0; i < neighbors.size(); i++ ) {
    const Patch* neighbor = neighbors[i];
    const Patch* realNeighbor = neighbor->getRealPatch();
    if( neighbor ) {
      IntVector newLow;
      IntVector newHigh;

      if( level->getIndex() == 0 ) {
        newLow = Max( lowIndex, neighbor->getExtraCellLowIndex() );
        newHigh = Min( highIndex, neighbor->getExtraCellHighIndex() );
      }
      else {
        // if in a copy-data timestep, only grab extra cells if on domain boundary
        newLow = Max( lowIndex, neighbor->getLowIndexWithDomainLayer( Patch::CellBased ) );
        newHigh = Min( highIndex, neighbor->getHighIndexWithDomainLayer( Patch::CellBased ) );
      }

      if( neighbor->isVirtual() ) {
        // rather than offsetting each point of pos_var's data, just adjust the box to compare it with.
        IntVector cellOffset = neighbor->getVirtualOffset();
        newLow -= cellOffset;
        newHigh -= cellOffset;
      }

      if (relPatch != neighbor){
        relPatch->cullIntersection( Patch::CellBased, IntVector( 0, 0, 0 ), realNeighbor, newLow, newHigh );

        if( newLow == newHigh ) {
          continue;
        }
      }

      // get the particle subset for this patch
      ParticleSubset* pset = getParticleSubset( matlIndex, neighbor, newLow, newHigh, posvar );

      //add subset to our current list
      totalParticles += pset->numParticles();
      subsets.push_back( pset );
      vneighbors.push_back( neighbors[i] );

    }
  }

  //create a new subset
  ParticleSubset* newsubset = scinew ParticleSubset(totalParticles, matlIndex, relPatch, lowIndex, highIndex, vneighbors, subsets);
  return newsubset;
}

//______________________________________________________________________
//
ParticleSubset*
OnDemandDataWarehouse::getDeleteSubset(       int     matlIndex
                                      , const Patch * patch
                                      )
{

  const Patch* realPatch = (patch != nullptr) ? patch->getRealPatch() : nullptr;
  ParticleSubset* subset = queryPSetDB( m_delset_DB, realPatch, matlIndex,
                                        patch->getExtraCellLowIndex(),
                                        patch->getExtraCellHighIndex(), nullptr );

  if (subset == nullptr) {
    SCI_THROW(UnknownVariable("DeleteSet", getID(), realPatch, matlIndex, "Cannot find delete set on patch", __FILE__, __LINE__) );
  }

  return subset;
}

//______________________________________________________________________
//
std::map<const VarLabel*, ParticleVariableBase*>*
OnDemandDataWarehouse::getNewParticleState(       int     matlIndex
                                          , const Patch * patch
                                          )
{
  {
    addsetDB_monitor addset_lock{ Uintah::CrowdMonitor<addsetDB_tag>::READER };

    const Patch* realPatch = (patch != nullptr) ? patch->getRealPatch() : nullptr;
    psetAddDBType::key_type key(matlIndex, realPatch);

    auto iter = m_addset_DB.find(key);
    if (iter == m_addset_DB.end()) {
      return nullptr;
    }

    return iter->second;
  }
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::haveParticleSubset(       int         matlIndex
                                         , const Patch     * patch
                                         ,       IntVector   low   /* = IntVector(0,0,0) */
                                         ,       IntVector   high  /* = IntVector(0,0,0) */
                                         ,       bool        exact /* = false */
                                         )
{
  if (low == high && high == IntVector(0, 0, 0)) {
    low = patch->getExtraCellLowIndex();
    high = patch->getExtraCellHighIndex();
  }
  const Patch* realPatch = patch->getRealPatch();
  // query subset
  ParticleSubset* subset = queryPSetDB(m_pset_db, realPatch, matlIndex, low, high, nullptr);

  // if no subset was returned there are no suitable subsets
  if (subset == nullptr) {
    return false;
  }

  // check if the user wanted an exact match
  if (exact) {
    return subset->getLow() == low && subset->getHigh() == high;
  }
  else {
    return true;
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(       constParticleVariableBase & constVar
                          , const VarLabel                  * label
                          ,       int                         matlIndex
                          , const Patch                     * patch
                          )
{
  checkGetAccess( label, matlIndex, patch );

  if (!m_var_DB.exists(label, matlIndex, patch)) {
    print();
    SCI_THROW( UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__) );
  }
  constVar = *dynamic_cast<ParticleVariableBase*>( m_var_DB.get( label, matlIndex, patch ) );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(       constParticleVariableBase & constVar
                          , const VarLabel                  * label
                          ,       ParticleSubset            * pset
                          )
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();

  // pset center patch and neighbor patch are not in same level (probably on an AMR copy data timestep)
  if ((pset->getNeighbors().size() == 0)
      || (pset->getNeighbors().front()->getLevel() == patch->getLevel()
          && pset->getLow() == patch->getExtraCellLowIndex()
          && pset->getHigh() == patch->getExtraCellHighIndex()) ) {
    get( constVar, label, matlIndex, patch );
  }
  else {
    checkGetAccess( label, matlIndex, patch );
    ParticleVariableBase* var = constVar.cloneType();

    const std::vector<const Patch*>& neighborPatches = pset->getNeighbors();
    const std::vector<ParticleSubset*>& neighbor_subsets = pset->getNeighborSubsets();

    std::vector<ParticleVariableBase*> neighborvars( neighborPatches.size() );

    for (size_t i = 0u; i < neighborPatches.size(); i++) {
      const Patch* neighborPatch = neighborPatches[i];

    if (!m_var_DB.exists(label, matlIndex, neighborPatches[i])) {
        SCI_THROW(UnknownVariable(label->getName(), getID(), neighborPatch, matlIndex,
                                  neighborPatch == patch?"on patch":"on neighbor", __FILE__, __LINE__) );
      }

      neighborvars[i] = var->cloneType();

      m_var_DB.get( label, matlIndex, neighborPatch, *neighborvars[i] );
    }

    // Note that when the neighbors are virtual patches (i.e. periodic
    // boundaries), then if var is a ParticleVariable<Point>, the points
    // of neighbors will be translated by its virtualOffset.

    var->gather( pset, neighbor_subsets, neighborvars, neighborPatches );

    constVar = *var;

    for (size_t i = 0u; i < neighborPatches.size(); i++) {
      delete neighborvars[i];
    }
    delete var;
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::getModifiable(       ParticleVariableBase & var
                                    , const VarLabel             * label
                                    ,       ParticleSubset       * pset
                                    )
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();
  checkModifyAccess( label, matlIndex, patch );

  if (pset->getLow() == patch->getExtraCellLowIndex() && pset->getHigh() == patch->getExtraCellHighIndex()) {
    if (!m_var_DB.exists(label, matlIndex, patch)) {
      SCI_THROW( UnknownVariable(label->getName(), getID(), patch, matlIndex,
                                 "", __FILE__, __LINE__) );
    }
    m_var_DB.get( label, matlIndex, patch, var );
  }
  else {
    SCI_THROW(InternalError("getModifiable (Particle Variable (" + label->getName() +
                            ") ). The particleSubset low/high index does not match the patch low/high indices",
                            __FILE__, __LINE__) );
  }
}

//______________________________________________________________________
//
ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable( const VarLabel       * label
                                          ,       ParticleSubset * pset
                                          )
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();

  if (pset->getLow() == patch->getExtraCellLowIndex() && pset->getHigh() == patch->getExtraCellHighIndex()) {
    return getParticleVariable( label, matlIndex, patch );
  }
  else {
    SCI_THROW(
        InternalError("getParticleVariable (Particle Variable (" + label->getName() +") ).  The particleSubset low/high index does not match the patch low/high indices", __FILE__, __LINE__) );
  }
}

//______________________________________________________________________
//
ParticleVariableBase*
OnDemandDataWarehouse::getParticleVariable( const VarLabel * label
                                          ,       int        matlIndex
                                          , const Patch    * patch
                                          )
{
  ParticleVariableBase* var = nullptr;

  // in case the it's a virtual patch -- only deal with real patches
  if (patch != nullptr) {
    patch = patch->getRealPatch();
  }

  checkModifyAccess( label, matlIndex, patch );

  if (!m_var_DB.exists(label, matlIndex, patch)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "",  __FILE__, __LINE__) );
  }
  var = dynamic_cast<ParticleVariableBase*>( m_var_DB.get( label, matlIndex, patch ) );

  return var;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::allocateTemporary( ParticleVariableBase & var
                                        , ParticleSubset       * pset
                                        )
{
  var.allocate(pset);
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::allocateAndPut(       ParticleVariableBase & var
                                     , const VarLabel             * label
                                     ,       ParticleSubset       * pset
                                     )
{
  int matlIndex = pset->getMatlIndex();
  const Patch* patch = pset->getPatch();

  // Error checking
  if (m_var_DB.exists(label, matlIndex, patch)) {
    SCI_THROW(InternalError("Particle variable already exists: " + label->getName(), __FILE__, __LINE__));
  }

  var.allocate(pset);
  put(var, label);
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(       ParticleVariableBase & var
                          , const VarLabel             * label
                          ,       bool                   replace /* = false */
                          )
{
  ASSERT(!m_finalized);

  ParticleSubset* pset = var.getParticleSubset();
  const Patch* patch = pset->getPatch();

  if (pset->getLow() != patch->getExtraCellLowIndex() || pset->getHigh() != patch->getExtraCellHighIndex()) {
      SCI_THROW(InternalError(" put(Particle Variable (" + label->getName() +
                              ") ).  The particleSubset low/high index does not match the patch low/high indices",
                              __FILE__, __LINE__) );
    }

  int matlIndex = pset->getMatlIndex();

  checkPutAccess( label, matlIndex, patch, replace );

  // Put it in the database
  printDebuggingPutInfo( label, matlIndex, patch, __LINE__ );

  m_var_DB.put( label, matlIndex, patch, var.clone(), d_scheduler->copyTimestep(), replace );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::copyOut(       ParticleVariableBase & var
                              , const VarLabel             * label
                              ,       ParticleSubset       * pset
                              )
{
  constParticleVariableBase* constVar = var.cloneConstType();
  this->get( *constVar, label, pset );
  var.copyData( &constVar->getBaseRep() );
  delete constVar;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::getCopy(       ParticleVariableBase & var
                              , const VarLabel             * label
                              ,       ParticleSubset       * pset
                              )
{
  constParticleVariableBase* constVar = var.cloneConstType();
  this->get( *constVar, label, pset );
  var.allocate( pset );
  var.copyData( &constVar->getBaseRep() );
  delete constVar;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(       constGridVariableBase & constVar
                          , const VarLabel              * label
                          ,       int                     matlIndex
                          , const Patch                 * patch
                          ,       Ghost::GhostType        gtype
                          ,       int                     numGhostCells
                          )
{
  GridVariableBase* var = constVar.cloneType();

  checkGetAccess( label, matlIndex, patch, gtype, numGhostCells );
  getGridVar( *var, label, matlIndex, patch, gtype, numGhostCells );

  constVar = *var;
  delete var;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::getModifiable(       GridVariableBase & var
                                    , const VarLabel         * label
                                    ,       int                matlIndex
                                    , const Patch            * patch
                                    ,       Ghost::GhostType   gtype         /* = Ghost::None */
                                    ,       int                numGhostCells /* = 0 */
                                    )
{
 //checkModifyAccess(label, matlIndex, patch);
  getGridVar(var, label, matlIndex, patch, gtype, numGhostCells);
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse:: allocateTemporary(       GridVariableBase & var
                                         , const Patch            * patch
                                         ,       Ghost::GhostType   gtype
                                         ,       int                numGhostCells
                                         )
{
  IntVector boundaryLayer(0, 0, 0); // Is this right?
  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), false);
  Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype, numGhostCells, lowOffset, highOffset);

  patch->computeExtents(basis, boundaryLayer, lowOffset, highOffset,lowIndex, highIndex);

  var.allocate(lowIndex, highIndex);
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::allocateAndPut(       GridVariableBase & var
                                     , const VarLabel         * label
                                     ,       int                matlIndex
                                     , const Patch            * patch
                                     ,       Ghost::GhostType   gtype         /* = Ghost::None */
                                     ,       int                numGhostCells /* = 0 */
                                     )
{
#if SCI_ASSERTION_LEVEL >= 1
  const TypeDescription * varType = var.virtualGetTypeDescription();
  if( label->typeDescription()->getType() != varType->getType() ||
      label->typeDescription()->getSubType()->getType() != varType->getSubType()->getType() ) {
      std::cout << "OnDemandDataWarehouse::allocateAndPut():  Error: VarLabel type does not match Variable type!\n";
      std::cout << "  VarLabel Name: " << label->getName() << "\n";
      std::cout << "  VarLabel Type: " << label->typeDescription()->getName() << "\n";
      std::cout << "  Variable Type: " << var.virtualGetTypeDescription()->getName() << "\n";
      SCI_THROW(InternalError("OnDemandDataWarehouse::allocateAndPut(): Var and Label types do not match!", __FILE__, __LINE__));
    }
#endif

  ASSERT(!m_finalized);

  // Note: almost the entire function is write locked in order to prevent dual allocations in a multi-threaded environment.
  // Whichever patch in a super patch group gets here first, does the allocating for the entire super patch group.
#if 0
  if (!hasRunningTask()) {
    SCI_THROW(InternalError("OnDemandDataWarehouse::AllocateAndPutGridVar can only be used when the dw has a running task associated with it.", __FILE__, __LINE__));
  }
#endif

  checkPutAccess(label, matlIndex, patch, false);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;

  //DS 06162020 Allocate ghost layer needed by device variables (label->getMaxDeviceGhost()). Similar functionality to computeWithScratchGhost.
  //This avoids reallocation during rewindow and ensures that getGridVariable returns original window pointer for modification.
  //If rewindow reallocates the variable, it  does not update the original variable in m_var_DB, but creates a copy. As a results modifications
  //occur in copy and not in the original. The simplest option is to allocate max ghosts in advance for every variable  so that rewindow never reallocates.
  //getMaxDeviceGhost returns 0 for cpu only execution. So no difference to CPU only version.
  //TODO: getMaxDeviceGhost does not count RMCRT tasks graphs and might conflict. Need to fix later.
  //TODO: check the impact on super patch.
  //Check comments in OnDemandDW::allocateAndPut, OnDemandDW::getGridVar, Array3<T>::rewindowExact and UnifiedScheduler::initiateD2H
  if ( numGhostCells < label->getMaxDeviceGhost() ) {
    Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), label->getMaxDeviceGhostType(), label->getMaxDeviceGhost(), lowOffset, highOffset);
  } else {
    Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype, numGhostCells, lowOffset, highOffset);
  }

  patch->computeExtents(basis, label->getBoundaryLayer(), lowOffset, highOffset, lowIndex, highIndex);

  if (!s_combine_memory) {
    bool exists = m_var_DB.exists(label, matlIndex, patch);
    if (exists) {
      // it had been allocated and put as part of the superpatch of another patch
      m_var_DB.get(label, matlIndex, patch, var);

      // The var's window should be the size of the patch or smaller than it.
      ASSERTEQ(Min(var.getLow(), lowIndex), lowIndex);
      ASSERTEQ(Max(var.getHigh(), highIndex), highIndex);

      // this is just a tricky way to uninitialize var
      Variable* tmpVar = dynamic_cast<Variable*>(var.cloneType());
      var.copyPointer(*tmpVar);
      delete tmpVar;
    }
    // allocate the memory
    var.allocate(lowIndex, highIndex);

    // put the variable in the database
    printDebuggingPutInfo( label, matlIndex, patch, __LINE__ );
    m_var_DB.put(label, matlIndex, patch, var.clone(), d_scheduler->copyTimestep(), true);
  }
  else {
    {
      varDB_monitor varDB_lock{ Uintah::CrowdMonitor<varDB_tag>::WRITER };

      bool exists = m_var_DB.exists(label, matlIndex, patch);
      if (exists) {
        // it had been allocated and put as part of the superpatch of another patch
        m_var_DB.get(label, matlIndex, patch, var);

        // The var's window should be the size of the patch or smaller than it.
        ASSERTEQ(Min(var.getLow(), lowIndex), lowIndex);
        ASSERTEQ(Max(var.getHigh(), highIndex), highIndex);

        if (var.getLow() != patch->getExtraLowIndex(basis, label->getBoundaryLayer()) || var.getHigh()
            != patch->getExtraHighIndex(basis, label->getBoundaryLayer())
            || var.getBasePointer() == nullptr /* place holder for ghost patch */) {

          // It wasn't allocated as part of another patch's superpatch;
          // it existed as ghost patch of another patch.. so we have no
          // choice but to blow it away and replace it.
          m_var_DB.put(label, matlIndex, patch, nullptr, d_scheduler->copyTimestep(), true);

          // this is just a tricky way to uninitialize var
          Variable* tmpVar = dynamic_cast<Variable*>(var.cloneType());
          var.copyPointer(*tmpVar);
          delete tmpVar;
        } else {
          // It was allocated and put as part of the superpatch of another patch
          var.rewindow(lowIndex, highIndex);
          return;  // got it -- done
        }
      }

      IntVector superLowIndex, superHighIndex;
      // requiredSuper[Low/High]'s don't take numGhostCells into consideration
      // -- just includes ghosts that will be required by later tasks.
      IntVector requiredSuperLow, requiredSuperHigh;

      const std::vector<const Patch*>* superPatchGroup = d_scheduler->getSuperPatchExtents(label, matlIndex, patch, gtype,
                                                                                           numGhostCells, requiredSuperLow,
                                                                                           requiredSuperHigh, superLowIndex,
                                                                                           superHighIndex);
      ASSERT(superPatchGroup != nullptr);

      var.allocate(superLowIndex, superHighIndex);

#if SCI_ASSERTION_LEVEL >= 3

      // check for dead portions of a variable (variable space that isn't covered by any patch).
      // This will happen with L-shaped patch configs and ngc > extra cells.
      // find all dead space and mark it with a bogus value.

      if (1) {  // numGhostCells > ec) { (numGhostCells is 0, query it from the superLowIndex...
        std::deque<Box> b1, b2, difference;
        b1.push_back( Box(Point(superLowIndex(0), superLowIndex(1), superLowIndex(2)),
                          Point(superHighIndex(0), superHighIndex(1), superHighIndex(2))));
        for (size_t i = 0u; i < (*superPatchGroup).size(); i++) {
          const Patch* p = (*superPatchGroup)[i];
          IntVector low = p->getExtraLowIndex(basis, label->getBoundaryLayer());
          IntVector high = p->getExtraHighIndex(basis, label->getBoundaryLayer());
          b2.push_back(Box(Point(low(0), low(1), low(2)), Point(high(0), high(1), high(2))));
        }
        difference = Box::difference(b1, b2);

#if 0
        if (difference.size() > 0) {
          std::cout << "Box difference: " << superLowIndex << " " << superHighIndex << " with patches " << std::endl;
          for (size_t i = 0u; i < (*superPatchGroup).size(); i++) {
            const Patch* p = (*superPatchGroup)[i];
            std::cout << p->getExtraLowIndex(basis, label->getBoundaryLayer()) << " " << p->getExtraHighIndex(basis, label->getBoundaryLayer()) << std::endl;
          }

          for (size_t i = 0; i < difference.size(); i++) {
            std::cout << difference[i].lower() << " " << difference[i].upper() << std::endl;
          }
        }
#endif
        // get more efficient way of doing this...
        for (size_t i = 0; i < difference.size(); i++) {
          Box b = difference[i];
          IntVector low((int)b.lower()(0), (int)b.lower()(1), (int)b.lower()(2));
          IntVector high((int)b.upper()(0), (int)b.upper()(1), (int)b.upper()(2));
          if (GridVariable<double>* typedVar = dynamic_cast<GridVariable<double>*>(&var)) {
            for (CellIterator iter(low, high); !iter.done(); iter++) {
              (*typedVar)[*iter] = -5.555555e256;
            }
          } else if (GridVariable<Vector>* typedVar = dynamic_cast<GridVariable<Vector>*>(&var)) {
            for (CellIterator iter(low, high); !iter.done(); iter++) {
              (*typedVar)[*iter] = -5.555555e256;
            }
          }
        }
      }
#endif

      Patch::selectType encompassedPatches;
      if (requiredSuperLow == lowIndex && requiredSuperHigh == highIndex) {
        // only encompassing the patch currently being allocated
        encompassedPatches.push_back(patch);
      } else {
        // Use requiredSuperLow/High instead of superLowIndex/superHighIndex
        // so we don't put the var for patches in the datawarehouse that won't be
        // required (this is important for scrubbing).
        patch->getLevel()->selectPatches(requiredSuperLow, requiredSuperHigh, encompassedPatches);
      }

      // Make a set of the non ghost patches that
      // has quicker lookup than the vector.
      std::set<const Patch*> nonGhostPatches;
      for (size_t i = 0u; i < superPatchGroup->size(); ++i) {
        nonGhostPatches.insert((*superPatchGroup)[i]);
      }

      Patch::selectType::iterator iter = encompassedPatches.begin();
      for (; iter != encompassedPatches.end(); ++iter) {
        const Patch* patchGroupMember = *iter;

        GridVariableBase* clone = var.clone();

        IntVector groupMemberLowIndex = patchGroupMember->getExtraLowIndex(basis, label->getBoundaryLayer());
        IntVector groupMemberHighIndex = patchGroupMember->getExtraHighIndex(basis, label->getBoundaryLayer());

        IntVector enclosedLowIndex = Max(groupMemberLowIndex, superLowIndex);
        IntVector enclosedHighIndex = Min(groupMemberHighIndex, superHighIndex);

        clone->rewindow(enclosedLowIndex, enclosedHighIndex);
        if (patchGroupMember == patch) {
          // this was checked already
          exists = false;
        } else {
          exists = m_var_DB.exists(label, matlIndex, patchGroupMember);
        }

        if (patchGroupMember->isVirtual()) {
          // Virtual patches can only be ghost patches.
          ASSERT(nonGhostPatches.find(patchGroupMember) == nonGhostPatches.end());
          clone->offsetGrid(IntVector(0, 0, 0) - patchGroupMember->getVirtualOffset());
          enclosedLowIndex = clone->getLow();
          enclosedHighIndex = clone->getHigh();
          patchGroupMember = patchGroupMember->getRealPatch();
          IntVector dummy;
          if (d_scheduler->getSuperPatchExtents(label, matlIndex, patchGroupMember, gtype, numGhostCells, dummy, dummy, dummy, dummy) != nullptr) {
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

          GridVariableBase* existingGhostVar = dynamic_cast<GridVariableBase*>(m_var_DB.get(label, matlIndex, patchGroupMember));
          IntVector existingLow = existingGhostVar->getLow();
          IntVector existingHigh = existingGhostVar->getHigh();
          IntVector minLow = Min(existingLow, enclosedLowIndex);
          IntVector maxHigh = Max(existingHigh, enclosedHighIndex);

          if (existingGhostVar->isForeign()) {
            // data already being received, so don't replace it
            delete clone;
          } else if (minLow == enclosedLowIndex && maxHigh == enclosedHighIndex) {
            // this new ghost variable section encloses the old one,
            // so replace the old one
            printDebuggingPutInfo(label, matlIndex, patchGroupMember, __LINE__);

            m_var_DB.put(label, matlIndex, patchGroupMember, clone, d_scheduler->copyTimestep(), true);
          } else {
            // Either the old ghost variable section encloses this new one
            // (so leave it), or neither encloses the other (so just forget
            // about it -- it'll allocate extra space for it when receiving
            // the ghost data in recvMPIGridVar if nothing else).
            delete clone;
          }
        } else {
          // it didn't exist before -- add it
          printDebuggingPutInfo(label, matlIndex, patchGroupMember, __LINE__);

          m_var_DB.put(label, matlIndex, patchGroupMember, clone, d_scheduler->copyTimestep(), false);
        }
      }
    } // end varDB_lock{ Uintah::CrowdMonitor<varDB_tag>::WRITER }
    var.rewindow(lowIndex, highIndex);
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::copyOut(       GridVariableBase & var
                              , const VarLabel         * label
                              ,       int                matlIndex
                              , const Patch            * patch
                              ,       Ghost::GhostType   gtype         /* = Ghost::None */
                              ,       int                numGhostCells /* = 0 */
                              )
{
  GridVariableBase* tmpVar = var.cloneType();
  getGridVar( *tmpVar, label, matlIndex, patch, gtype, numGhostCells );
  var.copyData( tmpVar );
  delete tmpVar;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::getCopy(       GridVariableBase& var
                              , const VarLabel*         label
                              ,       int               matlIndex
                              , const Patch*            patch
                              ,       Ghost::GhostType  gtype         /* = Ghost::None */
                              ,       int               numGhostCells /* = 0 */
                              )
{
  GridVariableBase* tmpVar = var.cloneType();
  getGridVar( *tmpVar, label, matlIndex, patch, gtype, numGhostCells );
  var.allocate(tmpVar);
  var.copyData(tmpVar);
  delete tmpVar;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(       GridVariableBase & var
                          , const VarLabel         * label
                          ,       int                matlIndex
                          , const Patch            * patch
                          ,       bool               replace /*= false */
                          )
{
  ASSERT(!m_finalized);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);
  ASSERTEQ(basis, Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));

  checkPutAccess(label, matlIndex, patch, replace);

  DOUTR(g_dw_get_put_dbg, "Putting: " << *label << " MI: " << matlIndex << " patch: " << *patch << " into DW: " << d_generation);

   // Put it in the database
   IntVector low = patch->getExtraLowIndex(basis, label->getBoundaryLayer());
   IntVector high = patch->getExtraHighIndex(basis, label->getBoundaryLayer());
   if (Min(var.getLow(), low) != var.getLow() || Max(var.getHigh(), high) != var.getHigh()) {
     std::ostringstream msg_str;
     msg_str << "put: Variable's window (" << var.getLow() << " - " << var.getHigh() << ") must encompass patches extent (" << low << " - " << high;
     SCI_THROW(InternalError(msg_str.str(), __FILE__, __LINE__));
   }
   USE_IF_ASSERTS_ON(bool no_realloc =) var.rewindow(low, high);
   // error would have been thrown above if the any reallocation would be needed
   ASSERT(no_realloc);
   printDebuggingPutInfo( label, matlIndex, patch, __LINE__ );

   m_var_DB.put(label, matlIndex, patch, var.clone(), d_scheduler->copyTimestep(),true);
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::get(       PerPatchBase & var
                          , const VarLabel     * label
                          ,       int            matlIndex
                          , const Patch        * patch
                          )
{
  checkGetAccess(label, matlIndex, patch);
  if (!m_var_DB.exists(label, matlIndex, patch)) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "perpatch data", __FILE__, __LINE__));
  }
  m_var_DB.get(label, matlIndex, patch, var);
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::put(       PerPatchBase & var
                          , const VarLabel     * label
                          ,       int            matlIndex
                          , const Patch        * patch
                          ,       bool           replace /* = false */
                          )
{
  ASSERT( !m_finalized );
  checkPutAccess( label, matlIndex, patch, replace );

  // Put it in the database
  printDebuggingPutInfo( label, matlIndex, patch, __LINE__ );
  m_var_DB.put( label, matlIndex, patch, var.clone(), d_scheduler->copyTimestep(), true );
}

//______________________________________________________________________
// This returns a constGridVariable for *ALL* patches on a level.
// This method is essentially identical to "getRegion" except the call to
// level->selectPatches( ) has been replaced by level->allPatches()
// For grids containing a large number of patches selectPatches() is very slow
// This assumes that the variable is not in the DWDatabase<Level>  m_level_DB;
//______________________________________________________________________
void
OnDemandDataWarehouse::getLevel(       constGridVariableBase & constGridVar
                               , const VarLabel              * label
                               ,       int                     matlIndex
                               , const Level                 * level
                               )
{
  IntVector level_lowIndex, level_highIndex;
  level->findCellIndexRange(level_lowIndex, level_highIndex);  // including extra cells

  GridVariableBase* gridVar = constGridVar.cloneType();
  gridVar->allocate(level_lowIndex, level_highIndex);
  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  std::vector<const Patch*> missing_patches;     // for bulletproofing

  //__________________________________
  // define the patches for the entire level
  const PatchSet* myPatchesSet = level->allPatches();
  std::vector<const Patch*> patches(level->numPatches());
  for (int m = 0; m < myPatchesSet->size(); m++) {
    const PatchSubset* myPatches = myPatchesSet->getSubset(m);

    for (int p = 0; p < myPatches->size(); ++p) {
      patches[p] = myPatches->get(p);
    }
  }

  int nCellsCopied = 0;

  for (size_t i = 0u; i < patches.size(); ++i) {
    const Patch* patch = patches[i];

    std::vector<Variable*> varlist;
    m_var_DB.getlist(label, matlIndex, patch, varlist);
    GridVariableBase* this_var = nullptr;

    //__________________________________
    //  is this variable on this patch?
    for (auto iter = varlist.begin();; ++iter) {
      if (iter == varlist.end()) {
        this_var = nullptr;
        break;
      }

      //verify that the variable is valid
      this_var = dynamic_cast<GridVariableBase*>(*iter);

      if ((this_var != nullptr) && this_var->isValid()) {
        break;
      }
    }

    // just like a "missing patch": got data on this patch, but it either corresponds to a different
    // region or is incomplete"
    if (this_var == nullptr) {
      missing_patches.push_back(patch->getRealPatch());
      continue;
    }

    GridVariableBase* tmpVar = gridVar->cloneType();
    tmpVar->copyPointer(*this_var);

    // if patch is virtual, it is probably a boundary layer/extra cell that has been requested (from AMR)
    if (patch->isVirtual()) {
      tmpVar->offset(patch->getVirtualOffset());
    }

    //__________________________________
    //  copy this patch's data
    IntVector lo = patch->getExtraLowIndex(basis, label->getBoundaryLayer());
    IntVector hi = patch->getExtraHighIndex(basis, label->getBoundaryLayer());

    try {
      gridVar->copyPatch(tmpVar, lo, hi);
    }
    catch (InternalError& e) {
      std::cout << "OnDemandDataWarehouse::getLevel ERROR: failed copying patch data.\n "
                << " Level- " << level->getIndex()
                << " patch "<< lo << " " << hi
                << " variable range: " << tmpVar->getLow() << " "<< tmpVar->getHigh() << std::endl;
      throw e;
    }

    delete tmpVar;
    IntVector diff(hi - lo);
    nCellsCopied += diff.x() * diff.y() * diff.z();
  }  // patches loop

  //bulletproofing
  long totalLevelCells = level->totalCells();

  if (totalLevelCells != nCellsCopied ) {
    std::cout << d_myworld->myRank() << "  Unknown Variable " << *label << ", matl " << matlIndex << ", L-" << level->getIndex()
              << ", Patches on which the variable wasn't found: ";

    for (size_t i = 0u; i < missing_patches.size(); ++i) {
      std::cout << *missing_patches[i] << " ";
    }
    std::cout << " copied cells: " << nCellsCopied << " requested cells: " << totalLevelCells << std::endl;
    throw InternalError("Missing variable in getLevel().  Unable to find the patch variable over the requested region.", __FILE__, __LINE__);

  }

  //__________________________________
  //  Diagnostics
  DOUTR(g_dw_get_put_dbg, " getLevel:  Variable " << *label << ", matl " << matlIndex << ", L-" << level->getIndex());

  constGridVar = *dynamic_cast<GridVariableBase*>(gridVar);
  delete gridVar;
}

//______________________________________________________________________
//This putLevel is meant for the Unified Scheduler only.
void
OnDemandDataWarehouse::putLevelDB(       GridVariableBase * gridVar
                                 , const VarLabel         * label
                                 , const Level            * level
                                 ,       int                matlIndex /* = -1 */
                                 )
{
  // Put it in the level database
  bool init = (d_scheduler->copyTimestep()) || !(m_level_DB.exists( label, matlIndex, level ));

  printDebuggingPutInfo( label, matlIndex, level, __LINE__ );

  m_level_DB.put( label, matlIndex, level, gridVar, init, true );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::getRegion(       constGridVariableBase & constVar
                                , const VarLabel              * label
                                ,       int                     matlIndex
                                , const Level                 * level
                                , const IntVector             & low
                                , const IntVector             & high
                                ,       bool                    useBoundaryCells /* = true */
                                )
{
  GridVariableBase* var = constVar.cloneType();
  getRegionModifiable( *var, label, matlIndex, level, low, high, useBoundaryCells);
  constVar = *var;
  delete var;

}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::getRegionModifiable(       GridVariableBase & var
                                          , const VarLabel         * label
                                          ,       int                matlIndex
                                          , const Level            * level
                                          , const IntVector        & reqLow
                                          , const IntVector        & reqHigh
                                          ,       bool               useBoundaryCells /* = true */
                                          )
{
  var.allocate(reqLow, reqHigh);

  TypeDescription::Type varType = label->typeDescription()->getType();
  Patch::VariableBasis basis = Patch::translateTypeToBasis( varType, false );

  // Enlarge the requested region, sometimes we only want extra cells.
  // select patches has difficulties with that request.
  IntVector adjustment = IntVector(1, 1, 1);
  if (basis == Patch::XFaceBased) {
    adjustment = IntVector(1, 0, 0);
  }
  else if (basis == Patch::YFaceBased) {
    adjustment = IntVector(0, 1, 0);
  }
  else if (basis == Patch::ZFaceBased) {
    adjustment = IntVector(0, 0, 1);
  }

  IntVector tmpLow(  reqLow  - adjustment);
  IntVector tmpHigh( reqHigh + adjustment);

  Patch::selectType patches;
  level->selectPatches(tmpLow, tmpHigh, patches);

  // bulletproofing vars
  int nCellsCopied = 0;                       // number of cells copied
  bool foundInDB   = false;                   // found variable in DB
  std::vector<const Patch*> missing_patches;  // patches that do not contain the label

  for (unsigned int i = 0; i < patches.size(); i++) {
    const Patch* patch = patches[i];

    // After regridding selected patches may return stale patches so
    // make sure the variable exists on the patch.
    foundInDB = exists( label, matlIndex, patch );
    if( !foundInDB ) {
     continue;
    }

    //__________________________________
    //  For this patch find the intersection of the requested region
    IntVector patchLo = patch->getLowIndex(basis);
    IntVector patchHi = patch->getHighIndex(basis);
    if (useBoundaryCells) {
      patchLo = patch->getExtraLowIndex(  basis, label->getBoundaryLayer() );
      patchHi = patch->getExtraHighIndex( basis, label->getBoundaryLayer() );
    }

    IntVector offset ( 0, 0, 0 );
    if (patch->isVirtual()) {
      offset = patch->getVirtualOffset();
    }

    patchLo -= offset;
    patchHi -= offset;

    IntVector l = Max( patchLo, reqLow );
    IntVector h = Min( patchHi, reqHigh );

    if (l.x() >= h.x() || l.y() >= h.y() || l.z() >= h.z()) {
      continue;
    }

    //__________________________________
    //  search varDB for variable
    std::vector<Variable*> varlist;
    m_var_DB.getlist(label, matlIndex, patch, varlist);
    GridVariableBase* v = nullptr;
    bool varFound = false;

    for (std::vector<Variable*>::iterator rit = varlist.begin();; ++rit) {

      // variable not found
      if (rit == varlist.end()) {
        v = nullptr;
        break;
      }

      // Variable found in dataBase, does it cover the region?
      v = dynamic_cast<GridVariableBase*>(*rit);

      IntVector varLo = v->getLow();
      IntVector varHi = v->getHigh();

      bool doesCoverRegion = ( Min(l, varLo) == varLo && Max(h, varHi) == varHi );
      if ( (v != nullptr) && v->isValid() && doesCoverRegion) {
        varFound  = true;
        break;
      }
    }

    // Variable was not found on this patch.  Add patch to missing patches
    if ( varFound == false ) {
      missing_patches.push_back(patch->getRealPatch());
      continue;
    }

    //__________________________________
    // Copy data into the variable
    GridVariableBase* tmpVar = var.cloneType();
    tmpVar->copyPointer(*v);

    if (patch->isVirtual()) {
      // if patch is virtual, it is probably a boundary layer/extra cell that has been requested (from AMR)
      // We need to adjust the source but not the dest by the virtual offset
      tmpVar->offset(patch->getVirtualOffset());
    }

    try {
      var.copyPatch(tmpVar, l, h);
    }
    catch (InternalError& e) {
      std::cout << "OnDemandDataWarehouse::getRegionModifiable ERROR: failed copying patch data.\n "
                << " Level- " << level->getIndex()
                << " region Requested: " << reqLow << " " << reqHigh << ", patch intersection: " << l << " " << h
                << " patch "<< patchLo << " " << patchHi
                << " variable range: " << tmpVar->getLow() << " "<< tmpVar->getHigh() << std::endl;
      throw e;
    }
    delete tmpVar;

    // keep track of the number of cells copied.
    IntVector diff(h - l);
    nCellsCopied += diff.x() * diff.y() * diff.z();
  }  // patches loop

  //__________________________________
  //  BULLETPROOFING  Verify that the correct number of cells were copied
  //
  // compute the number of cells in the region
  long requestedCells = level->getTotalCellsInRegion( varType, label->getBoundaryLayer(), reqLow, reqHigh );

  // In non-cubic levels there may be overlapping patches that need to be accounted for.
  std::pair<int, int> overLapCells_range = std::make_pair( 0,0 );

  if ( level->isNonCubic() ){
    overLapCells_range = level->getOverlapCellsInRegion( patches, reqLow, reqHigh);
  }

  //  The number of cells copied = requested cells  OR is within the range of possible overlapping cells
  // In domains with multiple overlapping patches (inside corners in 3D) the number of cells copied can fall
  // within a range
  bool cond1 = ( nCellsCopied != requestedCells );
  bool cond2 = ( nCellsCopied < requestedCells + overLapCells_range.first );
  bool cond3 = ( nCellsCopied > requestedCells + overLapCells_range.second );

  if ( nCellsCopied == 0  || ( cond1 && cond2 && cond3 ) ) {

    DOUT(true,  d_myworld->myRank() << "  Unknown Variable " << *label << ", matl " << matlIndex << ", L-" << level->getIndex()
              << ", DW " << getID() << ", Variable exists in DB: " << foundInDB << "\n"
              << "   Requested region: " << reqLow << " " << reqHigh
              << ", Physical Units: " << level->getCellPosition(reqLow) << ", " << level->getCellPosition(reqHigh) << "\n"
              << "   #copied cells: " << nCellsCopied << ", #requested cells: " << requestedCells
              << ",  #overlapping Cells min:" << overLapCells_range.first << " max: " << overLapCells_range.second
              << "\n cond1: " << cond1 << " cond2: " << cond2 << " cond3 " << cond3 );

    if (missing_patches.size() > 0) {
      DOUT(true, "  Patches on which the variable wasn't found:");

      for (size_t i = 0u; i < missing_patches.size(); i++) {

        const Patch* patch =  missing_patches[i];
        IntVector patchLo =  patch->getExtraCellLowIndex();
        IntVector patchHi =  patch->getExtraCellHighIndex();

        IntVector regionLo = Uintah::Max( reqLow,  patchLo );
        IntVector regionHi = Uintah::Min( reqHigh, patchHi );
        IntVector diff( regionHi - regionLo );
        int intersectionCells = diff.x() * diff.y() * diff.z();

        DOUT(true, "  " << *missing_patches[i] << " cells in missing patches: " << intersectionCells );
      }
    }
    throw InternalError("Missing variable in getRegionModifiable().  Unable to find the patch variable over the requested region.", __FILE__, __LINE__);
  }

  if (g_dw_get_put_dbg.active()) {
    std::ostringstream mesg;
    mesg << "  getRegionModifiable() Variable " << *label << ", matl " << matlIndex << ", L-" << level->getIndex()
         << " For region: " << reqLow << " " << reqHigh << "  has been copied";
    DOUTR(true, mesg.str());
  }
}

//______________________________________________________________________
//
size_t
OnDemandDataWarehouse::emit(       OutputContext & oc
                           , const VarLabel      * label
                           ,       int             matlIndex
                           , const Patch         * patch
                           )
{
  checkGetAccess(label, matlIndex, patch);

  Variable* var = nullptr;
  IntVector l, h;
  if (patch) {
    // Save with the boundary layer, otherwise restarting from the DataArchive won't work.
    patch->computeVariableExtents(label->typeDescription()->getType(), label->getBoundaryLayer(), Ghost::None, 0, l, h);
    switch (label->typeDescription()->getType()) {
      case TypeDescription::NCVariable :
      case TypeDescription::CCVariable :
      case TypeDescription::SFCXVariable :
      case TypeDescription::SFCYVariable :
      case TypeDescription::SFCZVariable : { // get list
        std::vector<Variable*> varlist;
        m_var_DB.getlist(label, matlIndex, patch, varlist);

        GridVariableBase* v = nullptr;
        for (auto rit = varlist.begin();; ++rit) {
          if (rit == varlist.end()) {
            v = nullptr;
            break;
          }
          v = dynamic_cast<GridVariableBase*>(*rit);
          //verify that the variable is valid and matches the dependencies requirements.
          if (v && v->isValid() && Min(l, v->getLow()) == v->getLow() && Max(h, v->getHigh()) == v->getHigh())  //find a completed region
            break;
        }
        var = v;
        break;
      }

      case TypeDescription::ParticleVariable :
      case TypeDescription::PerPatch :
      default : {
        if (m_var_DB.exists(label, matlIndex, patch)) {
          var = m_var_DB.get(label, matlIndex, patch);
        }
      }
    }
  }
  else { // reduction and sole variables
    switch (label->typeDescription()->getType()) {
      case TypeDescription::ReductionVariable :
      case TypeDescription::SoleVariable : {
        l = h = IntVector(-1, -1, -1);
        const Level* level = patch ? patch->getLevel() : nullptr;
        if (m_level_DB.exists(label, matlIndex, level)) {
          var = m_level_DB.get(label, matlIndex, level);
        }
        break;
      }
      default : {
      }
    }
  }

  if (var == nullptr) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "on emit", __FILE__, __LINE__));
  }
  size_t bytes;
  bytes = var->emit( oc, l, h, label->getCompressionMode() );
  return bytes;
}

#if HAVE_PIDX
void
OnDemandDataWarehouse::emitPIDX(       PIDXOutputContext & pc
                               , const VarLabel          * label
                               ,       int                 matlIndex
                               , const Patch             * patch
                               ,       unsigned char     * buffer
                               , const size_t              bufferSize
                               )
{
  checkGetAccess( label, matlIndex, patch );

  Variable* var = nullptr;
  IntVector l, h;

  if( patch ) {
    // Save with the boundary layer, otherwise restarting from the DataArchive won't work.
    patch->computeVariableExtents( label->typeDescription()->getType(), label->getBoundaryLayer(),
                                   Ghost::None, 0, l, h );

    switch ( label->typeDescription()->getType() ) {
    case TypeDescription::NCVariable :
    case TypeDescription::CCVariable :
    case TypeDescription::SFCXVariable :
    case TypeDescription::SFCYVariable :
    case TypeDescription::SFCZVariable :
      //get list
      {
        std::vector<Variable*> varlist;
        m_var_DB.getlist( label, matlIndex, patch, varlist );

        GridVariableBase* v = nullptr;
        for( std::vector<Variable*>::iterator rit = varlist.begin();; ++rit ) {
          if( rit == varlist.end() ) {
            v = nullptr;
            break;
          }
          v = dynamic_cast<GridVariableBase*>( *rit );

          //verify that the variable is valid and matches the dependencies requirements.
          if( v && v->isValid()
                && Min( l, v->getLow() ) == v->getLow()
                && Max( h, v->getHigh() ) == v->getHigh() ){  //find a completed region
            break;
          }
        }
        var = v;
      }
      break;
    case TypeDescription::ParticleVariable :
    case TypeDescription::PerPatch :
    default :
      if (m_var_DB.exists(label, matlIndex, patch)) {
        var = m_var_DB.get( label, matlIndex, patch );
      }
    }
  }
  else { // reduction and sole variables
    switch (label->typeDescription()->getType()) {
      case TypeDescription::ReductionVariable :
      case TypeDescription::SoleVariable : {
        l = h = IntVector(-1, -1, -1);
        const Level* level = patch ? patch->getLevel() : nullptr;
        if (m_level_DB.exists(label, matlIndex, level)) {
          var = m_level_DB.get(label, matlIndex, level);
        }
        break;
      }
      default : {
      }
    }
  }

  if( var == nullptr ) {
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "OnDemandDataWarehouse::emit ", __FILE__, __LINE__) );
  }

  var->emitPIDX( pc, buffer, l, h, bufferSize );
}

#endif

//______________________________________________________________________
//
void
OnDemandDataWarehouse::print(       std::ostream & intout
                            , const VarLabel     * label
                            , const Level        * level
                            ,       int            matlIndex /* = -1 */
                            )
{

  try {
    checkGetAccess( label, matlIndex, nullptr );
    ReductionVariableBase* var = dynamic_cast<ReductionVariableBase*>( m_level_DB.get( label, matlIndex, level ) );
    var->print( intout );
  }
  catch( UnknownVariable& ) {
    SCI_THROW( UnknownVariable(label->getName(), getID(), level, matlIndex, "on print ", __FILE__, __LINE__) );
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::deleteParticles( ParticleSubset * delset )
{
  int matlIndex = delset->getMatlIndex();
  Patch* patch = (Patch*)delset->getPatch();
  const Patch* realPatch = (patch != nullptr) ? patch->getRealPatch() : nullptr;

  {
    delsetDB_monitor delset_lock{ Uintah::CrowdMonitor<delsetDB_tag>::WRITER };

    psetDBType::key_type key(realPatch, matlIndex, getID());
    auto iter = m_delset_DB.find(key);
    ParticleSubset* currentDelset;
    if (iter != m_delset_DB.end()) {  //update existing delset
      // Concatenate the delsets into the delset that already exists in the DB.
      currentDelset = iter->second;
      for (auto iter = delset->begin(); iter != delset->end(); ++iter) {
        currentDelset->addParticle(*iter);
      }

      m_delset_DB.erase(key);
      m_delset_DB.insert(std::pair<psetDBType::key_type, ParticleSubset*>(key, currentDelset));

      delete delset;

    }
    else {
      m_delset_DB.insert(std::pair<psetDBType::key_type, ParticleSubset*>(key, delset));
      delset->addReference();
    }
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::addParticles( const Patch              * patch
                                   ,       int                  matlIndex
                                   ,       particleAddSetType * addedState
                                   )
{
  {
    addsetDB_monitor addset_lock{ Uintah::CrowdMonitor<addsetDB_tag>::WRITER };

    psetAddDBType::key_type key(matlIndex, patch);
    auto iter = m_addset_DB.find(key);
    if (iter != m_addset_DB.end()) {
      // SCI_THROW(InternalError("addParticles called twice for patch", __FILE__, __LINE__));
      std::cerr << "addParticles called twice for patch" << std::endl;
    } else {
      m_addset_DB[key] = addedState;
    }
  }
}

//______________________________________________________________________
//
int
OnDemandDataWarehouse::decrementScrubCount( const VarLabel * var
                                          ,        int       matlIndex
                                          ,  const Patch   * patch
                                          )
{

  int count = 0;
  switch (var->typeDescription()->getType()) {
    case TypeDescription::NCVariable :
    case TypeDescription::CCVariable :
    case TypeDescription::SFCXVariable :
    case TypeDescription::SFCYVariable :
    case TypeDescription::SFCZVariable :
    case TypeDescription::PerPatch :
    case TypeDescription::ParticleVariable : {
      count = m_var_DB.decrementScrubCount(var, matlIndex, patch);
      break;
    }
    case TypeDescription::SoleVariable : {
      SCI_THROW(InternalError("decrementScrubCount called for sole variable: " + var->getName(), __FILE__, __LINE__));
    }
    case TypeDescription::ReductionVariable : {
      // Reductions are not scrubbed
      SCI_THROW(InternalError("decrementScrubCount called for reduction variable: " + var->getName(), __FILE__, __LINE__));
    }
    default : {
      SCI_THROW(InternalError("decrementScrubCount for variable of unknown type: " + var->getName(), __FILE__, __LINE__));
    }
  }
  return count;
}

//______________________________________________________________________
//
DataWarehouse::ScrubMode
OnDemandDataWarehouse::setScrubbing( ScrubMode scrubMode )
{
  ScrubMode oldmode = m_scrub_mode;
  m_scrub_mode = scrubMode;
  return oldmode;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::setScrubCount( const VarLabel * var
                                    ,       int        matlIndex
                                    , const Patch    * patch
                                    ,       int        count
                                    )
{
  switch (var->typeDescription()->getType()) {
    case TypeDescription::NCVariable :
    case TypeDescription::CCVariable :
    case TypeDescription::SFCXVariable :
    case TypeDescription::SFCYVariable :
    case TypeDescription::SFCZVariable :
    case TypeDescription::PerPatch :
    case TypeDescription::ParticleVariable : {
      m_var_DB.setScrubCount(var, matlIndex, patch, count);
      break;
    }
    case TypeDescription::SoleVariable : {
      SCI_THROW(InternalError("setScrubCount called for sole variable: "+var->getName(), __FILE__, __LINE__));
    }
    case TypeDescription::ReductionVariable : {
      // Reductions are not scrubbed
      SCI_THROW(InternalError("setScrubCount called for reduction variable: "+var->getName(), __FILE__, __LINE__));
    }
    default : {
      SCI_THROW(InternalError("setScrubCount for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
    }
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::scrub( const VarLabel * var
                            ,       int        matlIndex
                            , const Patch    * patch
                            )
{
  switch (var->typeDescription()->getType()) {
    case TypeDescription::NCVariable :
    case TypeDescription::CCVariable :
    case TypeDescription::SFCXVariable :
    case TypeDescription::SFCYVariable :
    case TypeDescription::SFCZVariable :
    case TypeDescription::PerPatch :
    case TypeDescription::ParticleVariable : {
      m_var_DB.scrub(var, matlIndex, patch);
      break;
    }
    case TypeDescription::SoleVariable : {
      SCI_THROW(InternalError("scrub called for sole variable: "+var->getName(), __FILE__, __LINE__));
    }
    case TypeDescription::ReductionVariable : {
      // Reductions are not scrubbed
      SCI_THROW(InternalError("scrub called for reduction variable: "+var->getName(), __FILE__, __LINE__));
    }
    default : {
      SCI_THROW(InternalError("scrub for variable of unknown type: "+var->getName(), __FILE__, __LINE__));
    }
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::initializeScrubs(       int                        dwid
                                       , const FastHashTable<ScrubItem> * scrubcounts
                                       ,       bool                       add
                                       )
{
  m_var_DB.initializeScrubs( dwid, scrubcounts, add );
}

//______________________________________________________________________
//
//This is for the Unified Scheduler.  It retrieves a list of patches that are neighbors in the requested region
//It doesn't need a list of neighbor Variable objects in host-memory, as some patches may exist in GPU memory
//but not in host memory.  All we want are patches.  We'll let the Unified Scheduler figure out of those variables
//for these patches exists in host memory.
void OnDemandDataWarehouse::getNeighborPatches( const VarLabel                   * label
                                              , const Patch                      * patch
                                              ,       Ghost::GhostType             gtype
                                              ,       int                          numGhostCells
                                              ,       std::vector<const Patch *> & adjacentNeighbors
                                              )
{

  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  IntVector low = patch->getExtraLowIndex(basis, label->getBoundaryLayer());
  IntVector high = patch->getExtraHighIndex(basis, label->getBoundaryLayer());

  Patch::selectType neighbors;
  IntVector lowIndex, highIndex;
  patch->computeVariableExtents(basis, label->getBoundaryLayer(), gtype, numGhostCells, lowIndex, highIndex);

  if (numGhostCells > 0) {
    patch->getLevel()->selectPatches(lowIndex, highIndex, neighbors);
  }
  else {
    neighbors.push_back(patch);
  }

  for ( size_t i = 0u; i < neighbors.size(); ++i ) {
    const Patch* neighbor = neighbors[i];
    if (neighbor && (neighbor != patch)) {
      IntVector low  = Max( neighbor->getExtraLowIndex(  basis, label->getBoundaryLayer() ), lowIndex );
      IntVector high = Min( neighbor->getExtraHighIndex( basis, label->getBoundaryLayer() ), highIndex );

      patch->cullIntersection( basis, label->getBoundaryLayer(), neighbor, low, high );

      if (low == high) {
        continue;
      }

      //This patch works.
      adjacentNeighbors.push_back(neighbor);

    } //end if neighbor
  } //end for neighbors
}

//______________________________________________________________________
//
void OnDemandDataWarehouse::getValidNeighbors( const VarLabel                    * label
                                             ,       int                           matlIndex
                                             , const Patch                       * patch
                                             ,       Ghost::GhostType              gtype
                                             ,       int                           numGhostCells
                                             ,       std::vector<ValidNeighbors> & validNeighbors
                                             ,       bool                          ignoreMissingNeighbors /* = false */
                                             )
{

  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);

  IntVector low = patch->getExtraLowIndex(basis, label->getBoundaryLayer());
  IntVector high = patch->getExtraHighIndex(basis, label->getBoundaryLayer());

  Patch::selectType neighbors;
  IntVector lowIndex, highIndex;
  patch->computeVariableExtents(basis, label->getBoundaryLayer(), gtype, numGhostCells, lowIndex, highIndex);

  if (numGhostCells > 0) {
    patch->getLevel()->selectPatches(lowIndex, highIndex, neighbors);
  }
  else {
    neighbors.push_back(patch);
  }

  for( size_t i = 0u; i < neighbors.size(); ++i ) {
    const Patch* neighbor = neighbors[i];
    if( neighbor && (neighbor != patch) ) {
      IntVector low  = Max( neighbor->getExtraLowIndex( basis, label->getBoundaryLayer() ), lowIndex );
      IntVector high = Min( neighbor->getExtraHighIndex( basis, label->getBoundaryLayer() ),highIndex );

      patch->cullIntersection( basis, label->getBoundaryLayer(), neighbor, low, high );

      if (low == high) {
        continue;
      }

      if ( ignoreMissingNeighbors == false && m_var_DB.exists( label, matlIndex, neighbor ) ) {
        std::vector<Variable*> varlist;
        // Go through the main var plus any foreign vars for this label/material/patch
        m_var_DB.getlist( label, matlIndex, neighbor, varlist );

        GridVariableBase* v = nullptr;

        for (auto iter = varlist.begin();; ++iter) {
          if( iter == varlist.end() ) {
            v = nullptr;
            break;
          }
          v = dynamic_cast<GridVariableBase*>( *iter );
          // Verify that the variable is valid and matches the dependencies requirements
          if ((v != nullptr) && (v->isValid())) {
            if (neighbor->isVirtual()) {
              if (Min(v->getLow(), low - neighbor->getVirtualOffset()) == v->getLow() &&
                  Max(v->getHigh(), high - neighbor->getVirtualOffset()) == v->getHigh()) {
                break;
              }
            }
            else {
              if (Min(v->getLow(), low) == v->getLow() && Max(v->getHigh(), high) == v->getHigh()) {
                break;
              }
            }
          }
        }  //end for vars
        if (v == nullptr) {
          SCI_THROW(UnknownVariable(label->getName(), getID(), neighbor, matlIndex, neighbor == patch? "on patch":"on neighbor", __FILE__, __LINE__) );
        }
        ValidNeighbors temp;
        temp.validNeighbor = v;
        temp.neighborPatch = neighbor;
        temp.low = low;
        temp.high = high;
        validNeighbors.push_back(temp);
      }
      else {
        //We want to know about this patch what its low and high should be.  Perhaps
        //we will find this variable in the GPU instead of in host memory instead.

        ValidNeighbors temp;
        temp.validNeighbor = nullptr;
        temp.neighborPatch = neighbor;
        temp.low = low;
        temp.high = high;
        validNeighbors.push_back(temp);
      }
    } //end if neighbor
  } //end for neighbors
}

//______________________________________________________________________
//
/*
DS 06162020 numGhostCells related (ngc) scenarios in getGridVar:
With GPU:
G1. ngc = 0: No ghost cells needed
G2. ngc > 0 and <= label->getMaxDeviceGhost(): ghost cells for tasks in normal/regular task graph. The value passed should be <= max device ghost cells
G3. ngc > label->getMaxDeviceGhost() and <= SHRT_MAX: ghost cells for any RMCRT task variables from RMCRT task graph
Without GPU:
G4. ngc = 0: No ghost cells needed
G5. ngc > 0 and label->getMaxDeviceGhost() always returns 0 without gpu: ghost cells for tasks in normal/regular task graph.

Rewindow scenarios:
R1. no_reallocation_needed==false: Reallocation needed - each thread (or rather each call) will form a new copy, no race conditions. Copy will not be reused
R2. no_reallocation_needed==true : Reallocation NOT needed. Existing variable will be reused and could be shared among threads. Possible race condition.

Status flag update rules:
Status flag should be updated if and only if BOTH of the following conditions are satisfied:
S1. only for R2 (because R2 is shared and can be resued, but R1 is not shared and can not be reused.)
S2. final ngc>0 and the shared window size is EXACTLY same as patch + final ngc (i.e. low and high indices computed by computeVariableExtents)
    e.g. if final ngc = 2 and window size is 4, then there is a possibility that some task might request ngc=4 later (especially RMCRT), that's why
    window of size 4 was allocated at the beginning. So do not set status to Valid for final ngc < 4 or it might conflict with task with ngc=4.
    In the worst case, if the window is allocated which is greater than ALL ngc requests, ghost cells will be always gathered, but it will ensure correctness.


Handling combinations of G and R scenarios using status flag if needed (actual implementation):
Compute final ngc as:
  if(ngc > 0){
    final ngc = max(ngc, label->getMaxDeviceGhost())
  }
G* R1: Reallocation needed. Shared patch can not be reused so always copy values and never update status. (ideally G1 R1 and G4 R1 not possible)
G1 R2: set final ngc = 0 and return without setting flag to valid because ghost cells are not gathered.
G2 R2: set final ngc = label->getMaxDeviceGhost(). If status==valid, return existing var (ghost cells are valid), else gather ghost cells and set status to valid if S1 and S2 are met.
G3 R2: set final ngc = ngc, gather ghost cells again and update status
G4 R2: same as G1 R2
G5 R2: set final ngc = ngc, gather ghost cells and return. Do not worry about status
 */
void
OnDemandDataWarehouse::getGridVar(       GridVariableBase & var
                                 , const VarLabel         * label
                                 ,       int                matlIndex
                                 , const Patch            * patch
                                 ,       Ghost::GhostType   gtype
                                 ,       int                numGhostCells
                                 ,       int                exactWindow/*=0*/   //reallocate even if existing window is larger than requested. Exactly match dimensions
                                 )
{
  if ( numGhostCells > 0 && numGhostCells < label->getMaxDeviceGhost() ) {
      numGhostCells = label->getMaxDeviceGhost();
  }

  Patch::VariableBasis basis = Patch::translateTypeToBasis(label->typeDescription()->getType(), false);
  ASSERTEQ(basis, Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), true));

  if (!m_var_DB.exists(label, matlIndex, patch)) {
    std::cout << d_myworld->myRank() << " unable to find variable '" << label->getName() << " on patch: " << patch->getID() << " matl: " << matlIndex << "\n";
    SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "", __FILE__, __LINE__));
  }

  if (patch->isVirtual()) {
    m_var_DB.get(label, matlIndex, patch->getRealPatch(), var);
    var.offsetGrid(patch->getVirtualOffset());
  }
  else {
    m_var_DB.get(label, matlIndex, patch, var);
  }

  IntVector low = patch->getExtraLowIndex(basis, label->getBoundaryLayer());
  IntVector high = patch->getExtraHighIndex(basis, label->getBoundaryLayer());

  if ( gtype == Ghost::None ) {
    if (numGhostCells != 0) {
      SCI_THROW(InternalError("Ghost cells specified with type: None!\n", __FILE__, __LINE__));
    }
    // if this assertion fails, then it is having problems getting the
    // correct window of the data.
    USE_IF_ASSERTS_ON(bool no_realloc =) var.rewindow(low, high);
    ASSERT(no_realloc);
  }
  else {
    IntVector lowIndex, highIndex;
    patch->computeVariableExtents(basis, label->getBoundaryLayer(), gtype, std::max(label->getMaxDeviceGhost(), numGhostCells), lowIndex, highIndex); //DS 12132019: GPU Resize fix. Add scratchGhostCells to allocate extra memory

    //---------------------------------------------------------------------------------------------
    // NOTE: Though this works well now, not sure if we care about it.... ditch this? APH, 11/28/18
    //---------------------------------------------------------------------------------------------
    // reallocation needed: Ignore this if this is the initialization dw in its old state.
    // The reason for this is that during initialization it doesn't know what ghost cells will be required of it for the next timestep.
    // (This will be an issue whenever the task graph changes to require more ghost cells from the old datawarehouse).

    bool no_reallocation_needed = false;

    //DS 0106202: no_reallocation_needed = false, then rewindow will allocate a new pointer and copy everything. Remember, it does not modify
    //the pointer in m_var_DB. m_var_DB still points to old smaller pointer and same is returned if other thread requests the same variable.
    //Thus the new pointer thus becomes after resizing becomes exclusive to the calling thread and there are no race conditions - This
    //happens in most of the CPU tasks variable which are allocated without scratch ghost cells. In case of GPU tasks, data is allocated with
    //scratch ghost cells. Thus rewindow returns no_reallocation_needed = false, as the requested ghost cells fall within the already allocated
    //region. Thus all threads get the same pointer and that leads to data races. Not sure whether those data races are causing errors.
    //Possible fix: Use same max ghost cell count if  numGhostCells>0 and also use status flags to ensure only 1 thread gathers ghosts. Rest
    //of the threads can use those values as and when ready.

    //DS 06162020 Added logic to rewindowExact. Ensures the allocated space has exactly same size as the requested. This is needed for D2H copy.
    //Check comments in OnDemandDW::allocateAndPut, OnDemandDW::getGridVar, Array3<T>::rewindowExact and UnifiedScheduler::initiateD2H
    //TODO: Throwing error if allocated and requested spaces are not same might be a problem for RMCRT. Fix can be to create a temporary
    //variable (buffer) in UnifiedScheduler for D2H copy and then copy from buffer to actual variable. But lets try this solution first.
    if(exactWindow==0)
      no_reallocation_needed = var.rewindow( lowIndex, highIndex );
    else{
      no_reallocation_needed = var.rewindowExact( lowIndex, highIndex );
      if(no_reallocation_needed==false){
        printf("Error in rewindowing variable %s on patch %d\n", label->getName().c_str(), patch->getID() );
        SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matlIndex, "Error in rewindowing variable" , __FILE__, __LINE__) );
      }
    }

    if ( no_reallocation_needed == false && g_warnings_dbg ) {
      static bool warned = false;
             bool ignore = m_is_initialization_DW && m_finalized;
      if (!ignore && !warned) {
        warned = true;
        IntVector oldLow = var.getLow(), oldHigh = var.getHigh();
        static ProgressiveWarning rw("Warning: Reallocation needed for ghost region you requested.\nThis means the data you get back will be a copy of what's in the DW", 100);
        if (rw.invoke()) {
          // print out this message if the ProgressiveWarning does
          std::ostringstream errmsg;
          errmsg << "Rank-" << d_myworld->myRank() << " This occurrence for " << label->getName();
          if (patch != nullptr) {
            errmsg << " on patch " << patch->getID();
          }
          errmsg << " for material " << matlIndex << ".  Old range: " << oldLow << " " << oldHigh << " - new range " << lowIndex << " " << highIndex << " NGC " << numGhostCells;
          DOUT(true, errmsg.str());
        }
      }
    }

    if (numGhostCells == 0) { //Scenarios G1* and G4*
      return; // no need to gather ghost cells. Do not update status. Return. Scenarios G1* and G4*
    }
    bool should_gather = false;


    if (Parallel::usingDevice() == false) { //G5*
      should_gather = true; //G5 R2: set final ngc = ngc, gather ghost cells and return. Do not worry about status
    }
    else {
      if(no_reallocation_needed == true && numGhostCells == label->getMaxDeviceGhost()){ //G2 R2
        if (compareAndSwapAwaitingGhostDataOnCPU(label->getName().c_str(), patch->getID(), matlIndex, patch->getLevel()->getID())) {
          should_gather = true;
        }
      }
      else //G3 R2
        should_gather = true;
    }

    if (should_gather) {
      std::vector<ValidNeighbors> validNeighbors;
      getValidNeighbors(label, matlIndex, patch, gtype, numGhostCells, validNeighbors);
      for(auto iter = validNeighbors.begin(); iter != validNeighbors.end(); ++iter) {

        if (iter->validNeighbor) {
          GridVariableBase* srcvar = var.cloneType();
          GridVariableBase* tmp = iter->validNeighbor;
          srcvar->copyPointer(*tmp);
          if (iter->neighborPatch->isVirtual()) {
            srcvar->offsetGrid(iter->neighborPatch->getVirtualOffset());
          }
          try {
            var.copyPatch(srcvar, iter->low, iter->high);

          } catch (InternalError& e) {
            std::cout << " Bad range: " << iter->low << " " << iter->high
                      << " source var range: "  << iter->validNeighbor->getLow() << " " << iter->validNeighbor->getHigh()
                      << std::endl;
            throw e;
          }
          delete srcvar;
        }
      }
      if (Parallel::usingDevice() && no_reallocation_needed == true && numGhostCells == label->getMaxDeviceGhost()) {//this is need because rmcrt task graph might have different values of getMaxDeviceGhost. if condition avoids the conflict
        setValidWithGhostsOnCPU(label->getName().c_str(), patch->getID(), matlIndex, patch->getLevel()->getID() ); //ghosts are ready
      }
    }
    else { //threads which does not get to copy the data should wait until copy is completed.
      while (isValidWithGhostsOnCPU(label->getName().c_str(), patch->getID(), matlIndex, patch->getLevel()->getID()) == false );
    }
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::transferFrom(       DataWarehouse  * from
                                   , const VarLabel       * label
                                   , const PatchSubset    * patches
                                   , const MaterialSubset * matls
                                   )
{
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj;
  this->transferFrom( from, label, patches, matls, execObj, false, nullptr );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::transferFrom(       DataWarehouse  * from
                                   , const VarLabel       * label
                                   , const PatchSubset    * patches
                                   , const MaterialSubset * matls
                                   ,       bool             replace
                                   )
{
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj;
  this->transferFrom( from, label, patches, matls, execObj, replace, nullptr );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::transferFrom(       DataWarehouse  * from
                                   , const VarLabel       * label
                                   , const PatchSubset    * patches
                                   , const MaterialSubset * matls
                                   ,       bool             replace
                                   , const PatchSubset    * newPatches
                                   )
{
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj;
  this->transferFrom( from, label, patches, matls, execObj, replace, newPatches );
}

//______________________________________________________________________
//
//! Copy a var from the parameter DW to this one.  If newPatches
//! is not null, then it associates the copy of the variable with
//! newPatches, and otherwise it uses patches (the same it finds
//! the variable with.
//transferFrom() will perform a deep copy on the data if it's in the CPU or GPU.
//GPU transferFrom is not yet supported for GPU PerPatch variables.
//See the GPU's transferFrom() method for many more more details.
template <typename ExecSpace, typename MemSpace>
void
OnDemandDataWarehouse::transferFrom(       DataWarehouse                        * from
                                   , const VarLabel                             * label
                                   , const PatchSubset                          * patches
                                   , const MaterialSubset                       * matls
                                   ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                   ,       bool                                   replace
                                   , const PatchSubset                          * newPatches
                                   )
{
  OnDemandDataWarehouse* fromDW = dynamic_cast<OnDemandDataWarehouse*>( from );
  ASSERT( fromDW != nullptr );
  ASSERT( !m_finalized );

  for( auto p = 0; p < patches->size(); ++p ) {
    const Patch* patch = patches->get( p );
    const Patch* copyPatch = (newPatches ? newPatches->get( p ) : patch);
    for( auto m = 0; m < matls->size(); ++m ) {
      int matl = matls->get( m );
      checkPutAccess( label, matl, patch, replace );
      switch ( label->typeDescription()->getType() ) {
        case TypeDescription::NCVariable :
        case TypeDescription::CCVariable :
        case TypeDescription::SFCXVariable :
        case TypeDescription::SFCYVariable :
        case TypeDescription::SFCZVariable : {
          //See if it exists in the CPU or GPU
          bool found = false;
          if (fromDW->m_var_DB.exists(label, matl, patch)) {
            found = true;
            GridVariableBase* v = dynamic_cast<GridVariableBase*>( fromDW->m_var_DB.get( label, matl, patch ) )->clone();
            m_var_DB.put( label, matl, copyPatch, v, d_scheduler->copyTimestep(), replace );
          }


#ifdef HAVE_CUDA

          if (Uintah::Parallel::usingDevice()) {
            //See if it's in the GPU.  Both the source and destination must be in the GPU data warehouse,
            //both must be listed as "allocated", and both must have the same variable sizes.
            //If those conditions match, then it will do a device to device memcopy call.
            //hard coding it for the 0th GPU
            const Level * level = patch->getLevel();
            const int levelID = level->getID();
            const int patchID = patch->getID();
            GPUGridVariableBase* device_var_source = OnDemandDataWarehouse::createGPUGridVariable(label->typeDescription()->getSubType()->getType());
            GPUGridVariableBase* device_var_dest = OnDemandDataWarehouse::createGPUGridVariable(label->typeDescription()->getSubType()->getType());
            if(!execObj.getStream()) {
              std::cout << "ERROR! transferFrom() does not have access to the task and its associated CUDA stream."
                        << " You need to update the task's callback function to include more parameters which supplies this information."
                        << " Then you need to pass that detailed task pointer into the transferFrom method."
                        << " As an example, please see the parameters for Poisson1::timeAdvanceUnified."   << std::endl;
              throw InternalError("transferFrom() needs access to the task's pointer and its associated CUDA stream.\n", __FILE__, __LINE__);
            }
            //The GPU assigns streams per task.  For transferFrom to work, it *must* know which correct stream to use
            bool foundGPU = getGPUDW(0)->transferFrom((cudaStream_t*)execObj.getStream(),
                                                      *device_var_source, *device_var_dest,
                                                      from->getGPUDW(0),
                                                      label->getName().c_str(), patchID, matl, levelID);

            if (!found && foundGPU) {
              found = true;
            }
          }

#endif


          if (!found) {
            SCI_THROW(UnknownVariable(label->getName(), fromDW->getID(), patch, matl, "in transferFrom", __FILE__, __LINE__) );
          }
          break;
        }

        case TypeDescription::ParticleVariable : {
          if( !fromDW->m_var_DB.exists( label, matl, patch ) ) {
            SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matl, "in transferFrom", __FILE__, __LINE__) );
          }

          ParticleSubset* subset;
          if( !haveParticleSubset( matl, copyPatch ) ) {
            ParticleSubset* oldsubset = fromDW->getParticleSubset( matl, patch );
            subset = createParticleSubset( oldsubset->numParticles(), matl, copyPatch );
          }
          else {
            subset = getParticleSubset( matl, copyPatch );
          }

          ParticleVariableBase* v = dynamic_cast<ParticleVariableBase*>( fromDW->m_var_DB.get( label, matl, patch ) );
          if( patch == copyPatch ) {
            m_var_DB.put( label, matl, copyPatch, v->clone(), d_scheduler->copyTimestep(), replace );
          }
          else {
            ParticleVariableBase* newv = v->cloneType();
            newv->copyPointer( *v );
            newv->setParticleSubset( subset );
            m_var_DB.put( label, matl, copyPatch, newv, d_scheduler->copyTimestep(), replace );
          }
          break;
        }
        case TypeDescription::PerPatch : {
          if( !fromDW->m_var_DB.exists( label, matl, patch ) ) {
            SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matl, "in transferFrom", __FILE__, __LINE__) );
          }
          PerPatchBase* v = dynamic_cast<PerPatchBase*>( fromDW->m_var_DB.get( label, matl, patch ) );
          m_var_DB.put( label, matl, copyPatch, v->clone(), d_scheduler->copyTimestep(), replace );
          break;
        }
        case TypeDescription::SoleVariable : {
          if( !fromDW->m_var_DB.exists( label, matl, patch ) ) {
            SCI_THROW(UnknownVariable(label->getName(), getID(), patch, matl, "in transferFrom", __FILE__, __LINE__) );
          }
          SoleVariableBase* v = dynamic_cast<SoleVariableBase*>( fromDW->m_var_DB.get( label, matl, patch ) );
          m_var_DB.put( label, matl, copyPatch, v->clone(), d_scheduler->copyTimestep(), replace );
          break;
        }
        case TypeDescription::ReductionVariable : {
          SCI_THROW(InternalError("transferFrom not implemented for reduction variables: " + label->getName(), __FILE__, __LINE__) );
        }
        default : {
          SCI_THROW(InternalError("Unknown variable type in transferFrom: " + label->getName(), __FILE__, __LINE__) );
        }
      }
    }
  }
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::logMemoryUse(       std::ostream  & out
                                   ,       unsigned long & total
                                   , const std::string   & tag
                                   )
{
  int dwid = d_generation;
  m_var_DB.logMemoryUse(out, total, tag, dwid);

  // Log the psets.
  for (psetDBType::iterator iter = m_pset_db.begin(); iter != m_pset_db.end(); iter++) {
    ParticleSubset* pset = iter->second;
    std::ostringstream elems;
    elems << pset->numParticles();
    logMemory(out, total, tag, "particles", "ParticleSubset", pset->getPatch(), pset->getMatlIndex(), elems.str(),
              pset->numParticles() * sizeof(particleIndex), pset->getPointer(), dwid);
  }
}

//______________________________________________________________________
//
inline void
OnDemandDataWarehouse::checkGetAccess( const VarLabel         * label
                                     ,       int                matlIndex
                                     , const Patch            * patch
                                     ,       Ghost::GhostType   gtype         /* = Ghost::None */
                                     ,       int                numGhostCells /* = 0 */
                                     )
{
#if 0

#if SCI_ASSERTION_LEVEL >= 1

  std::map<std::thread::id, RunningTaskInfo>* runningTasks = getRunningTasksInfo();

  if (runningTasks != nullptr) {
    for (auto iter = runningTasks->begin(); iter != runningTasks->end(); ++iter) {
      RunningTaskInfo& runningTaskInfo = iter->second;
      const Task* runningTask = runningTaskInfo.m_task;

      // don't check if done outside of any task (i.e. SimulationController)
      if (runningTask == nullptr) {
        return;
      }

      IntVector lowOffset, highOffset;
      Patch::getGhostOffsets(label->typeDescription()->getType(), gtype, numGhostCells, lowOffset, highOffset);

      VarAccessMap& runningTaskAccesses = runningTaskInfo.m_accesses;

      std::map<VarLabelMatl<Patch>, AccessInfo>::iterator findIter;
      findIter = runningTaskAccesses.find(VarLabelMatl<Patch>(label, matlIndex, patch));

      if (!hasGetAccess(runningTask, label, matlIndex, patch, lowOffset, highOffset, &runningTaskInfo) && !hasPutAccess(runningTask, label, matlIndex, patch)) {

        // If it was accessed by the current task already, then it should have get access
        // (i.e. if you put it in, you should be able to get it right back out).
        if (findIter != runningTaskAccesses.end() && lowOffset == IntVector(0, 0, 0) && highOffset == IntVector(0, 0, 0)) {
          return;  // allow non ghost cell get if any access (get, put, or modify) is allowed
        }

        if (runningTask == nullptr || !(std::string(runningTask->getName()) == "Relocate::relocateParticles" || std::string(runningTask->getName()) == "SchedulerCommon::copyDataToNewGrid")) {
          std::string has{};
          switch (getWhichDW(&runningTaskInfo)) {
            case Task::NewDW : {
              has = "Task::NewDW";
              break;
            }
            case Task::OldDW : {
              has = "Task::OldDW";
              break;
            }
            case Task::ParentNewDW : {
              has = "Task::ParentNewDW";
              break;
            }
            case Task::ParentOldDW : {
              has = "Task::ParentOldDW";
              break;
            }
            default : {
              has = "UnknownDW";
            }
          }

          has += " datawarehouse get";

          if (numGhostCells > 0) {
            std::ostringstream ghost_str;
            ghost_str << " for " << numGhostCells << " layer";

            if (numGhostCells > 1) {
              ghost_str << "s";
            }
            ghost_str << " of ghosts around " << Ghost::getGhostTypeName(gtype);
            has += ghost_str.str();
          }
          std::string needs = "task requires";
#if 1
          SCI_THROW(DependencyException(runningTask, label, matlIndex, patch, has, needs, __FILE__, __LINE__));
#else
          if ( d_myworld->myRank() == 0 ) {
            DOUT(true, DependencyException::makeMessage(runningTask, label, matlIndex, patch, has, needs));
          }
#endif
        }
      }
      else {
        // access granted
        if (findIter == runningTaskAccesses.end()) {
          AccessInfo& accessInfo = runningTaskAccesses[VarLabelMatl<Patch>(label, matlIndex, patch)];
          accessInfo.accessType = GetAccess;
          accessInfo.encompassOffsets(lowOffset, highOffset);

          int ID = 0;
          if (patch) {
            ID = patch->getID();
          }
          std::string varname = "noname";
          if (label) {
            varname = label->getName();
          }
          if (g_dw_get_put_dbg.active()) {
            std::ostringstream mesg;
            mesg << " Task running is: " << runningTask->getName();
            mesg << std::left;
            mesg.width(10);
            mesg << "\t" << varname;
            mesg << std::left;
            mesg.width(10);
            mesg << " \t on patch " << ID << " and matl: " << matlIndex << " has been gotten\n";
            DOUTR(true , mesg.str());
          }
        }
        else {
          findIter->second.encompassOffsets(lowOffset, highOffset);
        }
      }
    }
  }  // running task loop

#endif // end #if 1

#endif // end #if SCI_ASSERTION_LEVEL >= 1
}

//______________________________________________________________________
//
inline void
OnDemandDataWarehouse::checkPutAccess( const VarLabel * label
                                     ,       int        matlIndex
                                     , const Patch    * patch
                                     ,       bool       replace
                                     )
{
#if 0

#if SCI_ASSERTION_LEVEL >= 1

  std::map<std::thread::id, RunningTaskInfo>* runningTasks = getRunningTasksInfo();

  if (runningTasks != nullptr) {
    for (auto iter = runningTasks->begin(); iter != runningTasks->end(); ++iter) {
      RunningTaskInfo& runningTaskInfo = iter->second;
      const Task* runningTask = runningTaskInfo.m_task;

      // don't check if outside of any task (i.e. SimulationController)
      if (runningTask == nullptr) {
        return;
      }

      VarAccessMap& runningTaskAccesses = runningTaskInfo.m_accesses;

      if (!hasPutAccess(runningTask, label, matlIndex, patch)) {
        if (std::string(runningTask->getName()) != "Relocate::relocateParticles") {
          std::string has{};
          std::string needs{};
          switch (getWhichDW(&runningTaskInfo)) {
            case Task::NewDW : {
              has = "Task::NewDW";
              break;
            }
            case Task::OldDW : {
              has = "Task::OldDW";
              break;
            }
            case Task::ParentNewDW : {
              has = "Task::ParentNewDW";
              break;
            }
            case Task::ParentOldDW : {
              has = "Task::ParentOldDW";
              break;
            }
            default : {
              has = "UnknownDW";
            }
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
          SCI_THROW(DependencyException(runningTask, label, matlIndex, patch, has, needs, __FILE__, __LINE__));
#else
          if ( d_myworld->myRank() == 0 ) {
            DOUT(true, DependencyException::makeMessage(runningTask, label, matlIndex, patch, has, needs));
          }
#endif
        }
      }
      else {
        runningTaskAccesses[VarLabelMatl<Patch>(label, matlIndex, patch)].accessType = replace ? ModifyAccess : PutAccess;
      }
    }
  }

#endif // end #if 1

#endif // end #if SCI_ASSERTION_LEVEL >= 1
}

//______________________________________________________________________
//
inline void
OnDemandDataWarehouse::checkModifyAccess( const VarLabel * label
                                        ,       int        matlIndex
                                        , const Patch    * patch
                                        )
{
  checkPutAccess(label, matlIndex, patch, true);
}

//______________________________________________________________________
//
inline Task::WhichDW
OnDemandDataWarehouse::getWhichDW( RunningTaskInfo * info )
{
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::NewDW, info)) {
    return Task::NewDW;
  }
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::OldDW, info)) {
    return Task::OldDW;
  }
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::ParentNewDW, info)) {
    return Task::ParentNewDW;
  }
  if (this == OnDemandDataWarehouse::getOtherDataWarehouse(Task::ParentOldDW, info)) {
    return Task::ParentOldDW;
  }

  throw InternalError("Unknown DW\n", __FILE__, __LINE__);
}

//______________________________________________________________________
//
inline bool
OnDemandDataWarehouse::hasGetAccess( const Task            * runningTask
                                   , const VarLabel        * label
                                   ,       int               matlIndex
                                   , const Patch           * patch
                                   ,       IntVector         lowOffset
                                   ,       IntVector         highOffset
                                   ,       RunningTaskInfo * info
                                   )
{
  return runningTask->hasRequires( label, matlIndex, patch, lowOffset, highOffset, getWhichDW(info) );
}

//______________________________________________________________________
//
inline bool
OnDemandDataWarehouse::hasPutAccess( const Task     * runningTask
                                   , const VarLabel * label
                                   ,       int        matlIndex
                                   , const Patch    * patch
                                   )
{
  return runningTask->hasComputes( label, matlIndex, patch );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::pushRunningTask( const Task                               * task
                                      ,       std::vector<OnDemandDataWarehouseP>* dws
                                      )
{
  std::lock_guard<Uintah::MasterLock> push_lock(g_running_tasks_lock);

  ASSERT(task);

  // true if the element was inserted, false if already exists
  bool inserted = m_running_tasks.insert(std::make_pair(std::this_thread::get_id(), RunningTaskInfo(task, dws))).second;

  DOUT(g_check_accesses, "Rank-" << Parallel::getMPIRank() << " TID-" << std::this_thread::get_id() << "  Task: " << task->getName() << ((inserted) ? " was pushed for access check." : " not pushed, element exists."));

}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::popRunningTask()
{
  std::lock_guard<Uintah::MasterLock> pop_lock(g_running_tasks_lock);

  auto iter = m_running_tasks.find(std::this_thread::get_id());
  if (iter != m_running_tasks.end()) {
    size_t num_erased = m_running_tasks.erase(std::this_thread::get_id());
    DOUT(g_check_accesses, "Rank-" << Parallel::getMPIRank() << " TID-" << std::this_thread::get_id()
                                   << "  Task: " << iter->second.m_task->getName() << " removed ("
                                   << num_erased << " total element(s))");
  }
}

//______________________________________________________________________
//
inline std::map<std::thread::id, OnDemandDataWarehouse::RunningTaskInfo>*
OnDemandDataWarehouse::getRunningTasksInfo()
{
  std::lock_guard<Uintah::MasterLock> get_running_task_lock(g_running_tasks_lock);


  if (m_running_tasks.empty()) {
    return nullptr;
  }
  else {
    return &m_running_tasks;
  }
}

//______________________________________________________________________
//
inline bool
OnDemandDataWarehouse::hasRunningTask()
{
  std::lock_guard<Uintah::MasterLock> has_running_task_lock(g_running_tasks_lock);

  return (m_running_tasks.find(std::this_thread::get_id()) != m_running_tasks.end());
}

//______________________________________________________________________
//
inline OnDemandDataWarehouse::RunningTaskInfo*
OnDemandDataWarehouse::getCurrentTaskInfo()
{
  std::lock_guard<Uintah::MasterLock> get_running_task_lock(g_running_tasks_lock);

  auto iter = m_running_tasks.find(std::this_thread::get_id());

  if (iter == m_running_tasks.end()) {
    return nullptr;
  }
  else {
    return &(m_running_tasks.find(std::this_thread::get_id())->second);
  }
}

//______________________________________________________________________
//
DataWarehouse*
OnDemandDataWarehouse::getOtherDataWarehouse( Task::WhichDW     dw
                                            , RunningTaskInfo * info
                                            )
{
  int dwindex = info->m_task->mapDataWarehouse( dw );
  DataWarehouse* result = (*info->m_dws)[dwindex].get_rep();
  return result;
}

//______________________________________________________________________
//
DataWarehouse*
OnDemandDataWarehouse::getOtherDataWarehouse( Task::WhichDW dw )
{
  RunningTaskInfo* info = getCurrentTaskInfo();
  int dwindex = info->m_task->mapDataWarehouse( dw );
  DataWarehouse* result = (*info->m_dws)[dwindex].get_rep();
  return result;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::checkTasksAccesses( const PatchSubset    * patches
                                         , const MaterialSubset * matls
                                         )
{
#if 0

#if SCI_ASSERTION_LEVEL >= 1

  task_access_monitor access_lock{ task_access_monitor::READER };

  RunningTaskInfo* currentTaskInfo = getCurrentTaskInfo();
  ASSERT(currentTaskInfo != nullptr);

  const Task* currentTask = currentTaskInfo->m_task;
  ASSERT(currentTask != nullptr);

  if (isFinalized()) {
    checkAccesses(currentTaskInfo, currentTask->getRequires(), GetAccess, patches, matls);
  }
  else {
    checkAccesses(currentTaskInfo, currentTask->getRequires(), GetAccess   , patches, matls);
    checkAccesses(currentTaskInfo, currentTask->getComputes(), PutAccess   , patches, matls);
    checkAccesses(currentTaskInfo, currentTask->getModifies(), ModifyAccess, patches, matls);
  }


#endif

#endif
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::checkAccesses(       RunningTaskInfo  * currentTaskInfo
                                    , const Task::Dependency * dep
                                    ,       AccessType         accessType
                                    , const PatchSubset      * domainPatches
                                    , const MaterialSubset   * domainMatls
                                    )
{
  ASSERT(currentTaskInfo != nullptr);
  const Task* currentTask = currentTaskInfo->m_task;
  if (currentTask->isReductionTask()) {
    return;  // no need to check reduction tasks.
  }

  VarAccessMap& currentTaskAccesses = currentTaskInfo->m_accesses;

  Handle<PatchSubset> default_patches  = scinew PatchSubset();
  Handle<MaterialSubset> default_matls = scinew MaterialSubset();
  default_patches->add(0);
  default_matls->add(-1);

  for (; dep != nullptr; dep = dep->m_next) {

#if 0
    if ((isFinalized() && dep->m_whichdw == Task::NewDW) || (!isFinalized() && dep->m_whichdw == Task::OldDW)) {
      continue;
    }
#endif

    const VarLabel* label = dep->m_var;
    IntVector lowOffset, highOffset;
    Patch::getGhostOffsets(label->typeDescription()->getType(), dep->m_gtype, dep->m_num_ghost_cells, lowOffset, highOffset);

    constHandle<PatchSubset> patches = dep->getPatchesUnderDomain(domainPatches);
    constHandle<MaterialSubset> matls = dep->getMaterialsUnderDomain(domainMatls);

    if (label->typeDescription() && label->typeDescription()->isReductionVariable()) {
      patches = default_patches.get_rep();
    }
    else if (patches == nullptr) {
      patches = default_patches.get_rep();
    }
    if (matls == nullptr) {
      matls = default_matls.get_rep();
    }

    if (currentTask->getName() == "Relocate::relocateParticles") {
      continue;
    }

    for (int m = 0; m < matls->size(); ++m) {
      int matl = matls->get(m);

      for (int p = 0; p < patches->size(); ++p) {
        const Patch* patch = patches->get(p);

        VarLabelMatl<Patch> key(label, matl, patch);
        auto find_iter = currentTaskAccesses.find(key);
        if (find_iter == currentTaskAccesses.end() || (*find_iter).second.accessType != accessType) {
          // If you require with ghost cells and modify, it can get into this situation
          if ((*find_iter).second.accessType == ModifyAccess && accessType == GetAccess) {
            continue;
          }

#if 1
// THIS OLD HACK PERHAPS CAN GO AWAY
          if (lowOffset == IntVector(0, 0, 0) && highOffset == IntVector(0, 0, 0)) {
            // In checkGetAccess(), this case does not record the fact
            // that the var was accessed, so don't throw exception here.
            continue;
          }
#endif
          if (find_iter == currentTaskAccesses.end()) {
            std::cout << "Error: did not find " << label->getName() << "\n";
            std::cout << "Mtl: " << m << ", Patch: " << *patch << "\n";
          }
          else {
            std::cout << "Error: accessType is not GetAccess for " << label->getName() << "\n";
          }
          std::cout << "For Task:\n";
          currentTask->displayAll(std::cout);

          // Makes request that is never followed through.
          std::string has, needs;
          if (accessType == GetAccess) {
            has = "task requires";
            if (isFinalized()) {
              needs = "get from the old datawarehouse";
            }
            else {
              needs = "get from the new datawarehouse";
            }
          }
          else if (accessType == PutAccess) {
            has = "task computes";
            needs = "datawarehouse put";
          }
          else {
            has = "task modifies";
            needs = "datawarehouse modify";
          }
          SCI_THROW(DependencyException(currentTask, label, matl, patch, has, needs, __FILE__, __LINE__));
        }
        // Can == ModifyAccess when you require with ghost cells and modify
        else if (((*find_iter).second.lowOffset != lowOffset || (*find_iter).second.highOffset != highOffset) && accessType != ModifyAccess ) {
          // Makes request for ghost cells that are never gotten.
          AccessInfo accessInfo = (*find_iter).second;
          ASSERT(accessType == GetAccess);

          // Assert that the request was greater than what was asked for
          // because the other cases (where it asked for more than the request)
          // should have been caught in checkGetAccess().
          ASSERT(Max((*find_iter).second.lowOffset, lowOffset)   == lowOffset);
          ASSERT(Max((*find_iter).second.highOffset, highOffset) == highOffset);

          std::string has, needs;
          has = "task requires";
          std::ostringstream ghost_str;
          ghost_str << " requesting " << dep->m_num_ghost_cells << " layer";
          if (dep->m_num_ghost_cells > 1) {
            ghost_str << "s";
          }
          ghost_str << " of ghosts around " << Ghost::getGhostTypeName(dep->m_gtype);
          has += ghost_str.str();

          if (isFinalized()) {
            needs = "get from the old datawarehouse";
          }
          else {
            needs = "get from the new datawarehouse";
          }
          needs += " that includes these ghosts";

          SCI_THROW(DependencyException(currentTask, label, matl, patch, has, needs, __FILE__, __LINE__));
        }
      }
    }
  }
}

//______________________________________________________________________
//
// For timestep abort/recomute
bool
OnDemandDataWarehouse::abortTimeStep()
{
  // BJW - time step aborting does not work with MPI - disabling.
  if( d_myworld->nRanks() == 0 ) {
    Patch * patch = nullptr;

    if (exists(VarLabel::find(abortTimeStep_name), -1, patch)) {
      bool_or_vartype ats_var;
      get( ats_var, VarLabel::find(abortTimeStep_name) );
      return bool(ats_var);
    }
    else
      return false;
  }
  else
    return false;
}

//__________________________________
//
bool
OnDemandDataWarehouse::recomputeTimeStep()
{
  Patch * patch = nullptr;

  if (exists(VarLabel::find(recomputeTimeStep_name), -1, patch)) {
    bool_or_vartype rts_var;
    get( rts_var, VarLabel::find(recomputeTimeStep_name) );
    return bool(rts_var);
  }
  else
    return false;
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::getVarLabelMatlLevelTriples( std::vector<VarLabelMatl<Level> > & vars ) const
{
  m_level_DB.getVarLabelMatlTriples( vars );
}

//______________________________________________________________________
//
void
OnDemandDataWarehouse::print()
{
  std::ostringstream mesg;
  mesg << d_myworld->myRank() << " VARIABLES in DW " << getID()
       << "\n" << d_myworld->myRank() << " Variable Patch Material\n"
       << "  -----------------------";
  DOUT(true, mesg.str());

  m_var_DB.print( d_myworld->myRank());
  m_level_DB.print( d_myworld->myRank());
}

//______________________________________________________________________
//  print debugging information
void
OnDemandDataWarehouse::printDebuggingPutInfo( const VarLabel * label
                                            , int              matlIndex
                                            , const Patch    * patch
                                            , int              line
                                            )
{
  if (g_dw_get_put_dbg.active()) {
    int L_indx = patch->getLevel()->getIndex();
    std::ostringstream mesg;
    mesg << " Putting (line: " << line << ") ";
    mesg << std::left;
    mesg.width(20);
    mesg << *label << " MI: " << matlIndex << " L-" << L_indx << " " << *patch << " \tinto DW: " << d_generation;
    DOUTR(true, mesg.str());
  }
}

//______________________________________________________________________
//  print debugging information
void
OnDemandDataWarehouse::printDebuggingPutInfo( const VarLabel * label
                                            , int              matlIndex
                                            , const Level    * level
                                            , int              line
                                            )
{
  if (g_dw_get_put_dbg.active()) {
    int L_indx = 0;
    if( level ){
      L_indx = level->getIndex();
    }

    std::ostringstream mesg;
    mesg << " Putting (line: "<<line<< ") ";
    mesg << std::left;
    mesg.width( 20 );
    mesg << *label << " MI: " << matlIndex << " L-"<< L_indx <<" " << " \tinto DW: " << d_generation;
    DOUTR(true, mesg.str());
  }
}

//______________________________________________________________________
//
// DS: 01042020: fix for OnDemandDW race condition
//bool
//OnDemandDataWarehouse::compareAndSwapAllocateOnCPU(char const* label, const int patchID, const int matlIndx, const int levelIndx)
//{
  //assuming varLock will be already secured in allocate method

//  bool allocated = false;
//  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
//  atomicDataStatus* status = nullptr;
//
//  std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
//    if (it != atomicStatusInHostMemory.end()) {
//        printf("ERROR:OnDemandDataWarehouse::compareAndSwapAllocate( ) already allocated. Possible race condition or duplicate allocation.\n");
//        varLock->unlock();
//        exit(-1);
//    } else {
//      //insert here
//      atomicDataStatus newVarStatus = ALLOCATED;
//      atomicStatusInHostMemory.insert( std::map<labelPatchMatlLevel, atomicDataStatus>::value_type( lpml, newVarStatus ) );
//      varLock->unlock();
//      return true;
//    }
//}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::isValidOnCPU(char const* label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  if (atomicStatusInHostMemory.find(lpml) != atomicStatusInHostMemory.end()) {
    bool retVal = ((__sync_fetch_and_or(&(atomicStatusInHostMemory.at(lpml)), 0) & VALID) == VALID);
    varLock->unlock();
    return retVal;
  } else {
    varLock->unlock();
    return false;
  }
}

//______________________________________________________________________
//
// TODO: This needs to be turned into a compare and swap operation
bool
OnDemandDataWarehouse::compareAndSwapSetValidOnCPU(char const* const label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
    if (it != atomicStatusInHostMemory.end()) {
      atomicDataStatus *status = &(it->second);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID) == VALID) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~COPYING_IN;
        newVarStatus = newVarStatus | VALID;
        settingValid = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      atomicDataStatus newVarStatus = VALID | ALLOCATED;
      atomicStatusInHostMemory.insert( std::map<labelPatchMatlLevel, atomicDataStatus>::value_type( lpml, newVarStatus ) );
      varLock->unlock();
      return true;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::compareAndSwapSetInvalidOnCPU(char const* const label, const int patchID, const int matlIndx, const int levelIndx)
{
  varLock->lock();
  bool settingValid = false;
  while (!settingValid) {
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
    if (it != atomicStatusInHostMemory.end()) {
      atomicDataStatus *status = &(it->second);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID) != VALID) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~VALID;
        settingValid = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      atomicDataStatus newVarStatus = ALLOCATED;
      atomicStatusInHostMemory.insert( std::map<labelPatchMatlLevel, atomicDataStatus>::value_type( lpml, newVarStatus ) );
      varLock->unlock();
      return true;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
//
// returns false if something else already claimed to copy or has copied data into the CPU.
// returns true if we are the ones to manage this variable's ghost data.
bool
OnDemandDataWarehouse::compareAndSwapCopyingIntoCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{

  atomicDataStatus* status = nullptr;

  // get the status
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  varLock->lock();
  std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
  if (it != atomicStatusInHostMemory.end()) {
    status = &(it->second);
  } else {
    //insert here??
    atomicDataStatus newVarStatus = COPYING_IN;
    atomicStatusInHostMemory.insert( std::map<labelPatchMatlLevel, atomicDataStatus>::value_type( lpml, newVarStatus ) );
    varLock->unlock();
    return true;
  }
  varLock->unlock();

  bool copyingin = false;
  while (!copyingin) {
    // get the address
    atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
    if (((oldVarStatus & COPYING_IN) == COPYING_IN) ||
       ((oldVarStatus & VALID) == VALID) ||
       ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
        // Something else already took care of it.  So this task won't manage it.
        return false;
      } else {
      //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
      atomicDataStatus newVarStatus = oldVarStatus | COPYING_IN;
      newVarStatus = newVarStatus & ~UNKNOWN;
      copyingin = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
    }
  }
  return true;
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::compareAndSwapAwaitingGhostDataOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  bool allocating = false;

  varLock->lock();
  while (!allocating) {
    //get the address
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
    if (it != atomicStatusInHostMemory.end()) {
      atomicDataStatus *status = &(it->second);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if (((oldVarStatus & AWAITING_GHOST_COPY) == AWAITING_GHOST_COPY) || ((oldVarStatus & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS)) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus | AWAITING_GHOST_COPY;
        allocating = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      printf("ERROR:OnDemandDataWarehouse::compareAndSwapAwaitingGhostDataOnCPU( )  Variable %s not found.\n", label);
      exit(-1);
      return false;
    }
  }
  varLock->unlock();
  return true;
}

//______________________________________________________________________
//
bool
OnDemandDataWarehouse::isValidWithGhostsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
  if (it != atomicStatusInHostMemory.end()) {
    bool retVal = ((__sync_fetch_and_or(&(it->second), 0) & VALID_WITH_GHOSTS) == VALID_WITH_GHOSTS);
    varLock->unlock();
    return retVal;
  } else {
    varLock->unlock();
    printf("var not found\n");
    return false;
  }
}

//______________________________________________________________________
//
// TODO: This needs to be turned into a compare and swap operation
void
OnDemandDataWarehouse::setValidWithGhostsOnCPU(char const* label, int patchID, int matlIndx, int levelIndx)
{
  varLock->lock();
  labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
  std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
  if (it != atomicStatusInHostMemory.end()) {
    //UNKNOWN
    //make sure the valid is still turned on
    __sync_or_and_fetch(&(it->second ), VALID);

    //turn off AWAITING_GHOST_COPY
    __sync_and_and_fetch(&(it->second ), ~AWAITING_GHOST_COPY);

    //turn on VALID_WITH_GHOSTS
    __sync_or_and_fetch(&(it->second ), VALID_WITH_GHOSTS);

    varLock->unlock();
  } else {
    varLock->unlock();
    exit(-1);
  }
}

//______________________________________________________________________
//
// returns false if something else already changed a valid variable to valid awaiting ghost data
// returns true if we are the ones to manage this variable's ghost data.
bool
OnDemandDataWarehouse::compareAndSwapSetInvalidWithGhostsOnCPU( char const* label, int patchID, int matlIndx, int levelIndx)
{
  bool allocating = false;

  varLock->lock();
  while (!allocating) {
    //get the address
    labelPatchMatlLevel lpml(label, patchID, matlIndx, levelIndx);
    std::map<labelPatchMatlLevel, atomicDataStatus>::iterator it = atomicStatusInHostMemory.find(lpml);
    if (it != atomicStatusInHostMemory.end()) {
      atomicDataStatus *status = &(it->second);
      atomicDataStatus oldVarStatus  = __sync_or_and_fetch(status, 0);
      if ((oldVarStatus & VALID_WITH_GHOSTS) == 0) {
        //Something else already took care of it.  So this task won't manage it.
        varLock->unlock();
        return false;
      } else {
        //Attempt to claim we'll manage the ghost cells for this variable.  If the claim fails go back into our loop and recheck
        atomicDataStatus newVarStatus = oldVarStatus & ~VALID_WITH_GHOSTS;
        allocating = __sync_bool_compare_and_swap(status, oldVarStatus, newVarStatus);
      }
    } else {
      varLock->unlock();
      atomicDataStatus newVarStatus = ALLOCATED;
      atomicStatusInHostMemory.insert( std::map<labelPatchMatlLevel, atomicDataStatus>::value_type( lpml, newVarStatus ) );
      varLock->unlock();
      return true;
    }
  }
  varLock->unlock();
  return true;
}
