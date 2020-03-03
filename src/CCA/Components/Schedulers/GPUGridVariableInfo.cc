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

#include <CCA/Components/Schedulers/GPUGridVariableInfo.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>

#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DebugStream.h>

extern Uintah::MasterLock cerrLock;

namespace Uintah {
  extern DebugStream gpu_stats;
}

using LabelPatchMatlLevelDW  = GpuUtilities::LabelPatchMatlLevelDw;
using TupleVariableMap       = std::map<LabelPatchMatlLevelDW, DeviceGridVariableInfo>;
using TupleVariableMultiMap  = std::multimap<LabelPatchMatlLevelDW, DeviceGridVariableInfo>;

//______________________________________________________________________
//
DeviceGridVariableInfo::DeviceGridVariableInfo(       Variable          * var
                                              ,       DeviceVarDest       dest
                                              ,       bool                staging
                                              ,       IntVector           sizeVector
                                              ,       size_t              sizeOfDataType
                                              ,       size_t              varMemSize
                                              ,       IntVector           offset
                                              ,       int                 matlIndx
                                              ,       int                 levelIndx
                                              , const Patch             * patchPointer
                                              , const Task::Dependency  * dep
                                              ,       Ghost::GhostType    gtype
                                              ,       int                 numGhostCells
                                              ,       unsigned int        whichGPU
                                              )
  : m_var{var}
  , m_dest{dest}
  , m_staging{staging}
  , m_sizeVector{sizeVector}
  , m_sizeOfDataType{sizeOfDataType}
  , m_varMemSize{varMemSize}
  , m_offset{offset}
  , m_matlIndx{matlIndx}
  , m_levelIndx{levelIndx}
  , m_patchPointer{patchPointer}
  , m_dep{dep}
  , m_gtype{gtype}
  , m_numGhostCells{numGhostCells}
  , m_whichGPU{whichGPU}
{}


//______________________________________________________________________
//
DeviceGridVariableInfo::DeviceGridVariableInfo(       Variable         * var
                                              ,       DeviceVarDest      dest
                                              ,       bool               staging
                                              ,       size_t             sizeOfDataType
                                              ,       size_t             varMemSize
                                              ,       int                matlIndx
                                              ,       int                levelIndx
                                              , const Patch            * patchPointer
                                              , const Task::Dependency * dep
                                              ,       unsigned int       whichGPU
                                              )
  : m_var{var}
  , m_dest{dest}
  , m_staging{staging}
  , m_sizeOfDataType{sizeOfDataType}
  , m_varMemSize{varMemSize}
  , m_matlIndx{matlIndx}
  , m_levelIndx{levelIndx}
  , m_patchPointer{patchPointer}
  , m_dep{dep}
  , m_numGhostCells{0}
  , m_whichGPU{whichGPU}
//  , m_dest{GpuUtilities::sameDeviceSameMpiRank}
{}


//______________________________________________________________________
//
void
DeviceGridVariables::clear() {
  vars.clear();
  deviceInfoMap.clear();
}


//______________________________________________________________________
//
void
DeviceGridVariables::add( const Patch            * patchPointer
                        ,       int                matlIndx
                        ,       int                levelIndx
                        ,       bool               staging
                        ,       IntVector          sizeVector
                        ,       size_t             varMemSize
                        ,       size_t             sizeOfDataType
                        ,       IntVector          offset
                        , const Task::Dependency * dep
                        ,       Ghost::GhostType   gtype
                        ,       int                numGhostCells
                        ,       unsigned int       whichGPU
                        ,       Variable         * var
                        ,       DeviceVarDest      dest
                        )
{

  LabelPatchMatlLevelDW lpmld(dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  DeviceGridVariableInfo tmp(var, dest, staging, sizeVector, sizeOfDataType, varMemSize, offset, matlIndx, levelIndx, patchPointer, dep, gtype, numGhostCells, whichGPU);

  std::pair <TupleVariableMultiMap::iterator, TupleVariableMultiMap::iterator> ret = vars.equal_range(lpmld);
  for (auto it = ret.first; it != ret.second; ++it) {
    if (it->second == tmp) {
      if (staging) {
        if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                << " DeviceGridVariables::add() - "
                << " This preparation queue for a host-side GPU datawarehouse already added this exact staging variable.  No need to add it again.  For label "
                << dep->m_var->getName()
                << " patch " << patchPointer->getID()
                << " matl " << matlIndx
                << " level " << levelIndx
                << " offset (" << offset.x() << ", " << offset.y() << ", " << offset.z() << ")"
                << " size (" << sizeVector.x() << ", " << sizeVector.y() << ", " << sizeVector.z() << ")"
                << std::endl;
          }
          cerrLock.unlock();
        }

        return;

      } else {
        // Don't add the same device var twice.
        printf("DeviceGridVariables::add() - ERROR:\n %s DeviceGridVariables::add() - This preparation queue for a host-side GPU datawarehouse already added this exact non-staging variable for label %s patch %d matl %d level %d staging %s offset(%d, %d, %d) size (%d, %d, %d) dw %d.  The found label was %s patch %d offset was (%d, %d, %d).\n",
            UnifiedScheduler::myRankThread().c_str(),
            dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx,
            staging ? "true" : "false",
            offset.x(), offset.y(), offset.z(),
            sizeVector.x(), sizeVector.y(), sizeVector.z(),
            dep->mapDataWarehouse(),
            it->first.m_label.c_str(),
            it->first.m_patchID,
            it->second.m_offset.x(), it->second.m_offset.y(), it->second.m_offset.z());

        SCI_THROW(InternalError("DeviceGridVariables::add() - Preparation queue already contained same exact variable for: -" + dep->m_var->getName(), __FILE__, __LINE__));
      }
    }
  }

  unsigned int totalVars;
  unsigned int entrySize = ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  if (deviceInfoMap.find(whichGPU) == deviceInfoMap.end() ) {
    DeviceInfo di;
    di.totalVars[dep->mapDataWarehouse()] = 1;
    totalVars = di.totalVars[dep->mapDataWarehouse()];

    // contiguous array calculations
    di.totalSize = entrySize;
    di.totalSizeForDataWarehouse[dep->mapDataWarehouse()] = entrySize;
    deviceInfoMap.insert(std::pair<unsigned int, DeviceInfo>(whichGPU, di));

  } else {
    DeviceInfo & di = deviceInfoMap[whichGPU];
    di.totalVars[dep->mapDataWarehouse()] += 1;
    totalVars = di.totalVars[dep->mapDataWarehouse()];

    // contiguous array calculations
    di.totalSize += entrySize;
    di.totalSizeForDataWarehouse[dep->mapDataWarehouse()] += entrySize;

  }

  // contiguous array calculations
  vars.insert( TupleVariableMap::value_type( lpmld, tmp ) );
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " DeviceGridVariables::add() - "
          << "Added into preparation queue for a host-side GPU datawarehouse a variable for label "
          << dep->m_var->getName()
          << " patch " << patchPointer->getID()
          << " matl " << matlIndx
          << " level " << levelIndx
          << " staging " << std::boolalpha << staging
          << " offset (" << offset.x() << ", " << offset.y() << ", " << offset.z() << ")"
          << " size (" << sizeVector.x() << ", " << sizeVector.y() << ", " << sizeVector.z() << ")"
          << " totalVars is: " << totalVars << std::endl;
    }
    cerrLock.unlock();
  }

  // TODO: Do we bother refining it if one copy is wholly inside another one?
}


//______________________________________________________________________
// For adding perPach vars, they don't have ghost cells.
// They also don't need to indicate the region they are valid (the patch handles that).
void
DeviceGridVariables::add( const Patch            * patchPointer
                        ,       int                matlIndx
                        ,       int                levelIndx
                        ,       size_t             varMemSize
                        ,       size_t             sizeOfDataType
                        , const Task::Dependency * dep
                        ,       unsigned int       whichGPU
                        ,       Variable         * var
                        ,       DeviceVarDest      dest
                        )
{
  //unlike grid variables, we should only have one instance of label/patch/matl/level/dw for patch variables.
  LabelPatchMatlLevelDW lpmld(dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  if (vars.find(lpmld) == vars.end()) {

    unsigned int entrySize = ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding)
                              % UnifiedScheduler::bufferPadding) + varMemSize;
    if (deviceInfoMap.find(whichGPU) == deviceInfoMap.end()) {
      DeviceInfo di;
      di.totalVars[dep->mapDataWarehouse()] = 1;
      //contiguous array calculations
      di.totalSize = entrySize;
      di.totalSizeForDataWarehouse[dep->mapDataWarehouse()] = entrySize;
      deviceInfoMap.insert(std::pair<unsigned int, DeviceInfo>(whichGPU, di));

    }
    else {
      DeviceInfo & di = deviceInfoMap[whichGPU];
      di.totalVars[dep->mapDataWarehouse()] += 1;

      //contiguous array calculations
      di.totalSize += entrySize;
      di.totalSizeForDataWarehouse[dep->mapDataWarehouse()] += entrySize;

    }

    DeviceGridVariableInfo tmp(var, dest, false, sizeOfDataType, varMemSize, matlIndx, levelIndx, patchPointer, dep, whichGPU);
    vars.insert(TupleVariableMap::value_type(lpmld, tmp));
  }
  else {
    //Don't add the same device var twice.
    printf("ERROR:\n This preparation queue already added this exact variable for label %s patch %d matl %d level %d dw %d\n",
           dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
    SCI_THROW(InternalError("Preparation queue already contained same exact variable for: -" + dep->m_var->getName(), __FILE__, __LINE__));
  }
}


//______________________________________________________________________
//
bool
DeviceGridVariables::varAlreadyExists( const VarLabel * label
                                     , const Patch    * patchPointer
                                     ,       int        matlIndx
                                     ,       int        levelIndx
                                     ,       int        dataWarehouse
                                     )
{


  LabelPatchMatlLevelDW lpmld(label->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dataWarehouse);
  std::pair <TupleVariableMultiMap::iterator, TupleVariableMultiMap::iterator> ret = vars.equal_range(lpmld);
  for (auto it = ret.first; it != ret.second; ++it) {
    if (it->second.m_staging == false) {
      if (gpu_stats.active()) {
          cerrLock.lock();
          {
            gpu_stats << UnifiedScheduler::myRankThread()
                << " DeviceGridVariables::varAlreadyExists() - "
                << " The var already existed for label " << label->getName().c_str()
                << " patch " << patchPointer->getID()
                << " matl " << matlIndx
                << " level " << levelIndx
                << " dataWarehouse " << dataWarehouse
                << std::endl;
          }
          cerrLock.unlock();
       }
      return true;
    }
  }
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " DeviceGridVariables::varAlreadyExists() - "
          << "The var did not yet exist for label " << label->getName().c_str()
          << " patch " << patchPointer->getID()
          << " matl " << matlIndx
          << " level " << levelIndx
          << " dataWarehouse " << dataWarehouse
          << std::endl;
    }
    cerrLock.unlock();
  }
  return false;
}


//______________________________________________________________________
//
bool
DeviceGridVariables::stagingVarAlreadyExists( const VarLabel * label
                                            , const Patch    * patchPointer
                                            ,       int        matlIndx
                                            ,       int        levelIndx
                                            ,       IntVector  low
                                            ,       IntVector  size
                                            ,       int        dataWarehouse
                                            )
{
  LabelPatchMatlLevelDW lpmld(label->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dataWarehouse);
  std::pair <TupleVariableMultiMap::iterator, TupleVariableMultiMap::iterator> ret = vars.equal_range(lpmld);
  for (auto it = ret.first; it != ret.second; ++it) {
    if (it->second.m_staging == true && it->second.m_offset == low && it->second.m_sizeVector == size) {
      return true;
    }
  }
  return false;
}


//______________________________________________________________________
// For adding taskVars, which are snapshots of the host-side GPU DW.
// This is the normal scenario, no "staging" variables.
void
DeviceGridVariables::addTaskGpuDWVar( const Patch            * patchPointer
                                    ,       int                matlIndx
                                    ,       int                levelIndx
                                    ,       size_t             sizeOfDataType
                                    , const Task::Dependency * dep
                                    ,       unsigned int       whichGPU
                                    )
{
  LabelPatchMatlLevelDW lpmld(dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  std::pair <TupleVariableMultiMap::iterator, TupleVariableMultiMap::iterator> ret = vars.equal_range(lpmld);
  for (auto it = ret.first; it != ret.second; ++it) {
    if (it->second.m_staging == false) {
      // Don't add the same device var twice.
      printf("ERROR:\n ::addTaskGpuDWVar() - This preparation queue for a task datawarehouse already added this exact variable for label %s patch %d matl %d level %d dw %d\n",dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
      SCI_THROW(InternalError("Preparation queue for a task datawarehouse already contained same exact variable for: -" + dep->m_var->getName(), __FILE__, __LINE__));
    }
  }

  unsigned int totalVars;
  if (deviceInfoMap.find(whichGPU) == deviceInfoMap.end() ) {
     DeviceInfo di;
     di.totalVars[dep->mapDataWarehouse()] = 1;
     totalVars = 1;
     deviceInfoMap.insert(std::pair<unsigned int, DeviceInfo>(whichGPU, di));

   } else {
     DeviceInfo & di = deviceInfoMap[whichGPU];
     di.totalVars[dep->mapDataWarehouse()] += 1;
     totalVars = di.totalVars[dep->mapDataWarehouse()];
   }


  // TODO: The Task DW doesn't hold any pointers.  So what does that mean about contiguous arrays?
  // Should contiguous arrays be organized by task???
  DeviceGridVariableInfo tmp(nullptr, GpuUtilities::unknown, false, sizeOfDataType, 0, matlIndx, levelIndx, patchPointer, dep, whichGPU);
  vars.insert( TupleVariableMap::value_type( lpmld, tmp ) );
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " DeviceGridVariables::addTaskGpuDWVar() - "
          << "Added into preparation queue for a task datawarehouse for a variable for label "
          << dep->m_var->getName()
          << " patch " << patchPointer->getID()
          << " matl " << matlIndx
          << " level " << levelIndx
          << " staging false"
          << " totalVars is: " << totalVars << std::endl;
    }
    cerrLock.unlock();
  }
}


//______________________________________________________________________
// For adding staging taskVars, which are snapshots of the host-side GPU DW.
void
DeviceGridVariables::addTaskGpuDWStagingVar( const Patch            * patchPointer
                                           ,       int                matlIndx
                                           ,       int                levelIndx
                                           ,       IntVector          offset
                                           ,       IntVector          sizeVector
                                           ,       size_t             sizeOfDataType
                                           , const Task::Dependency * dep
                                           ,       unsigned int       whichGPU
                                           )
{

  //Since this is a queue, we aren't likely to have the parent variable of the staging var,
  //as that likely went into the regular host-side gpudw as a computes in the last timestep.
  //Just make sure we haven't already added this exact staging variable.
  LabelPatchMatlLevelDW lpmld(dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  TupleVariableMap::iterator it = vars.find(lpmld);
  std::pair <TupleVariableMultiMap::iterator, TupleVariableMultiMap::iterator> ret = vars.equal_range(lpmld);
  for (auto it = ret.first; it != ret.second; ++it) {
    if (it->second.m_staging == true && it->second.m_sizeVector == sizeVector && it->second.m_offset == offset) {
      //Don't add the same device var twice.
      if (gpu_stats.active()) {
        cerrLock.lock();
        {
          gpu_stats << UnifiedScheduler::myRankThread()
              << " DeviceGridVariables::addTaskGpuDWStagingVar() - "
              << " This preparation queue for a task datawarehouse already added this exact staging variable.  No need to add it again.  For label "
              << dep->m_var->getName()
              << " patch " << patchPointer->getID()
              << " matl " << matlIndx
              << " level " << levelIndx
              << " offset (" << offset.x() << ", " << offset.y() << ", " << offset.z() << ")"
              << " size (" << sizeVector.x() << ", " << sizeVector.y() << ", " << sizeVector.z() << ")"
              << std::endl;
        }
        cerrLock.unlock();
      }
      return;
    }
  }

  unsigned int totalVars;
  if (deviceInfoMap.find(whichGPU) == deviceInfoMap.end() ) {
     DeviceInfo di;
     di.totalVars[dep->mapDataWarehouse()] = 1;
     totalVars = 1;
     deviceInfoMap.insert(std::pair<unsigned int, DeviceInfo>(whichGPU, di));

   } else {
     DeviceInfo & di = deviceInfoMap[whichGPU];
     di.totalVars[dep->mapDataWarehouse()] += 1;
     totalVars = di.totalVars[dep->mapDataWarehouse()];
   }

  size_t varMemSize = sizeVector.x() * sizeVector.y() * sizeVector.z() * sizeOfDataType;
  DeviceGridVariableInfo tmp(nullptr, GpuUtilities::unknown, true, sizeVector, sizeOfDataType, varMemSize , offset, matlIndx, levelIndx, patchPointer, dep, Ghost::None, 0, whichGPU);
  vars.insert( TupleVariableMap::value_type( lpmld, tmp ) );
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " DeviceGridVariables::addTaskGpuDWStagingVar() - "
          << "Added into preparation queue for a task datawarehouse for a variable for label "
          << dep->m_var->getName()
          << " patch " << patchPointer->getID()
          << " matl " << matlIndx
          << " level " << levelIndx
          << " staging true"
          << " offset (" << offset.x() << ", " << offset.y() << ", " << offset.z() << ")"
          << " size (" << sizeVector.x() << ", " << sizeVector.y() << ", " << sizeVector.z() << ")"
          << " totalVars is: " << totalVars << std::endl;
    }
    cerrLock.unlock();
  }
}


//______________________________________________________________________
// For adding taskVars, which are snapshots of the host-side GPU DW
// This is the normal scenario, no "staging" variables.
void
DeviceGridVariables::addVarToBeGhostReady( const std::string      & taskName
                                         , const Patch            * patchPointer
                                         ,       int                matlIndx
                                         ,       int                levelIndx
                                         , const Task::Dependency * dep
                                         ,       unsigned int       whichGPU
                                         )
{
  LabelPatchMatlLevelDW lpmld(dep->m_var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  std::pair<TupleVariableMultiMap::iterator, TupleVariableMultiMap::iterator> ret = vars.equal_range(lpmld);
  if (ret.first == ret.second) {
    DeviceGridVariableInfo tmp(nullptr, GpuUtilities::unknown, false, 0, 0, matlIndx, levelIndx, patchPointer, dep, whichGPU);
    vars.insert(TupleVariableMap::value_type(lpmld, tmp));
    if (gpu_stats.active()) {
      cerrLock.lock();
      {
        gpu_stats << UnifiedScheduler::myRankThread()
                  << " DeviceGridVariables::addVarToBeGhostReady() - " << "For task " << taskName
                  << " added to the listing of vars for which it is managing ghost cells" << dep->m_var->getName()
                  << " patch " << patchPointer->getID()
                  << " matl " << matlIndx
                  << " level " << levelIndx
                  << std::endl;
      }
      cerrLock.unlock();
    }
  }
}


//______________________________________________________________________
//
DeviceGridVariableInfo
DeviceGridVariables::getStagingItem( const std::string & label
                                   , const Patch       * patch
                                   , const int           matlIndx
                                   , const int           levelIndx
                                   , const IntVector     low
                                   , const IntVector     size
                                   , const int           dataWarehouseIndex
                                   ) const
{
  LabelPatchMatlLevelDW lpmld(label.c_str(), patch->getID(), matlIndx, levelIndx, dataWarehouseIndex);
  std::pair<TupleVariableMultiMap::const_iterator, TupleVariableMultiMap::const_iterator> ret = vars.equal_range(lpmld);
  TupleVariableMultiMap::const_iterator it;
  for (it = ret.first; it != ret.second; ++it) {
    if (it->second.m_staging == true && it->second.m_offset == low && it->second.m_sizeVector == size) {
      return it->second;
    }
  }
  printf("Error: DeviceGridVariables::getStagingItem(), item not found for offset (%d, %d, %d) size (%d, %d, %d).\n", low.x(),
         low.y(), low.z(), size.x(), size.y(), size.z());
  SCI_THROW(InternalError("Error: DeviceGridVariables::getStagingItem(), item not found for: -" + label, __FILE__, __LINE__));
}


//______________________________________________________________________
//
size_t
DeviceGridVariables::getTotalSize( const unsigned int whichGPU )
{
  return deviceInfoMap[whichGPU].totalSize;
}


//______________________________________________________________________
//
size_t DeviceGridVariables::getSizeForDataWarehouse( const unsigned int whichGPU, const int dwIndex )
{
  return deviceInfoMap[whichGPU].totalSizeForDataWarehouse[dwIndex];
}


//______________________________________________________________________
//
unsigned int
DeviceGridVariables::getTotalVars( const unsigned int whichGPU, const int DWIndex ) const
{
  std::map<unsigned int, DeviceInfo>::const_iterator it = deviceInfoMap.find(whichGPU);
  if (it != deviceInfoMap.end()) {
    return it->second.totalVars[DWIndex];
  }
  return 0;
}


//______________________________________________________________________
//
unsigned int
DeviceGridVariables::getTotalMaterials( const unsigned int whichGPU, const int DWIndex ) const
{
  std::map<unsigned int, DeviceInfo>::const_iterator it = deviceInfoMap.find(whichGPU);
  if (it != deviceInfoMap.end()) {
    return it->second.totalMaterials[DWIndex];
  }
  return 0;
}


//______________________________________________________________________
//
unsigned int
DeviceGridVariables::getTotalLevels( const unsigned int whichGPU, const int DWIndex ) const
{
  std::map<unsigned int, DeviceInfo>::const_iterator it = deviceInfoMap.find(whichGPU);
  if (it != deviceInfoMap.end()) {
    return it->second.totalLevels[DWIndex];
  }
  return 0;
}


//-----------------------------------------------------------------------------__________________________________________________________________________
//GpuUtilities methods
//-----------------------------------------------------------------------------


//______________________________________________________________________
//
void
GpuUtilities::assignPatchesToGpus(const GridP& grid){

  unsigned int currentAcceleratorCounter = 0;
  int numDevices = OnDemandDataWarehouse::getNumDevices();
  if (numDevices > 0) {
    std::map<const Patch *, int>::iterator it;
    for (int i = 0; i < grid->numLevels(); i++) {
      LevelP level = grid->getLevel(i);
      for (auto iter = level->patchesBegin(); iter != level->patchesEnd(); ++iter){
        // TODO: Clean up so that instead of assigning round robin, it assigns in blocks.
        const Patch* patch = *iter;
        it = patchAcceleratorLocation.find(patch);
        if (it == patchAcceleratorLocation.end()) {
          // This patch has not been assigned, so assign it to a GPU in a round robin fashion.
          patchAcceleratorLocation.insert(std::pair<const Patch *,int>(patch,currentAcceleratorCounter));
          currentAcceleratorCounter++;
          currentAcceleratorCounter %= numDevices;
        }
      }
    }
  }
}


//______________________________________________________________________
//
int
GpuUtilities::getGpuIndexForPatch(const Patch* patch)
{
  auto it = patchAcceleratorLocation.find(patch);
  if (it != patchAcceleratorLocation.end()) {
    return it->second;
  }
  return -1;
}
