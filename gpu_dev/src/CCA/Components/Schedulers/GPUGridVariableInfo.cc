/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
extern DebugStream gpu_stats;

extern SCIRun::Mutex cerrLock;

DeviceGridVariableInfo::DeviceGridVariableInfo(Variable* var,
            GpuUtilities::DeviceVarDestination dest,
            bool staging,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            Ghost::GhostType gtype,
            int numGhostCells,
            int whichGPU) {
  this->var = var;
  this->dest = dest;
  this->staging = staging;
  this->sizeVector = sizeVector;
  this->sizeOfDataType = sizeOfDataType;
  this->varMemSize = varMemSize;
  this->offset = offset;
  this->matlIndx = matlIndx;
  this->levelIndx = levelIndx;
  this->patchPointer = patchPointer;
  this->dep = dep;
  this->gtype = gtype;
  this->numGhostCells = numGhostCells;
  this->whichGPU = whichGPU;
}

DeviceGridVariableInfo::DeviceGridVariableInfo(Variable* var,
            GpuUtilities::DeviceVarDestination dest,
            bool staging,
            size_t sizeOfDataType,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            int whichGPU) {
  this->var = var;
  this->dest = dest;
  this->sizeOfDataType = sizeOfDataType;
  this->varMemSize = 0;
  this->matlIndx = matlIndx;
  this->levelIndx = levelIndx;
  this->staging = staging;
  this->patchPointer = patchPointer;
  this->dep = dep;
  this->whichGPU = whichGPU;
  this->numGhostCells = 0;
  this->dest = GpuUtilities::sameDeviceSameMpiRank;
}


DeviceGridVariables::DeviceGridVariables() {
  totalSize = 0;
  for (int i = 0; i < Task::TotalDWs; i++) {
    totalSizeForDataWarehouse[i] = 0;
    totalVars[i] = 0;
    totalMaterials[i] = 0;
    totalLevels[i] = 0;

  }
}
void DeviceGridVariables::add(const Patch* patchPointer,
          int matlIndx,
          int levelIndx,
          bool staging,
          IntVector sizeVector,
          size_t varMemSize,
          size_t sizeOfDataType,
          IntVector offset,
          const Task::Dependency* dep,
          Ghost::GhostType gtype,
          int numGhostCells,
          int whichGPU,
          Variable* var,
          GpuUtilities::DeviceVarDestination dest) {

  GpuUtilities::LabelPatchMatlLevelDw lpmld(dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  DeviceGridVariableInfo tmp(var, dest, staging, sizeVector, sizeOfDataType, varMemSize, offset, matlIndx, levelIndx, patchPointer, dep, gtype, numGhostCells, whichGPU);

  std::pair <std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator, std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator> ret;
  ret = vars.equal_range(lpmld);
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it=ret.first; it!=ret.second; ++it) {

    if (it->second == tmp) {
      //Don't add the same device var twice.
      printf("ERROR:\n %s DeviceGridVariables::add() - This preparation queue for a host-side GPU datawarehouse already added this exact variable for label %s patch %d matl %d level %d staging %s offset(%d, %d, %d) size (%d, %d, %d) dw %d.  The found label was %s patch %d offset was (%d, %d, %d).\n",
          UnifiedScheduler::myRankThread().c_str(),
          dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx,
          staging ? "true" : "false",
          offset.x(), offset.y(), offset.z(),
          sizeVector.x(), sizeVector.y(), sizeVector.z(),
          dep->mapDataWarehouse(),
          it->first.label.c_str(),
          it->first.patchID,
          it->second.offset.x(), it->second.offset.y(), it->second.offset.z());

      SCI_THROW(InternalError("Preparation queue already contained same exact variable for: -" + dep->var->getName(), __FILE__, __LINE__));
    }
  }

  totalVars[dep->mapDataWarehouse()] += 1;

  //contiguous array calculations
  totalSize += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  totalSizeForDataWarehouse[dep->mapDataWarehouse()] += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  vars.insert( std::map<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::value_type( lpmld, tmp ) );
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " DeviceGridVariables::add() - "
          << "Added into preparation queue for a host-side GPU datawarehouse a variable for label "
          << dep->var->getName()
          << " patch " << patchPointer->getID()
          << " matl " << matlIndx
          << " level " << levelIndx
          << " staging " << std::boolalpha << staging
          << " offset (" << offset.x() << ", " << offset.y() << ", " << offset.z() << ")"
          << " size (" << sizeVector.x() << ", " << sizeVector.y() << ", " << sizeVector.z() << ")"
          << " totalVars is: " << totalVars[dep->mapDataWarehouse()] << endl;
    }
    cerrLock.unlock();
  }

  //TODO: Do we bother refining it if one copy is wholly inside another one?
}

//For adding perPach vars, they don't have ghost cells.
//They also don't need to indicate the region they are valid (the patch
//handles that).
void DeviceGridVariables::add(const Patch* patchPointer,
          int matlIndx,
          int levelIndx,
          size_t varMemSize,
          size_t sizeOfDataType,
          const Task::Dependency* dep,
          int whichGPU,
          Variable* var,
          GpuUtilities::DeviceVarDestination dest) {

  //unlike grid variables, we should only have one instance of label/patch/matl/level/dw for patch variables.
  GpuUtilities::LabelPatchMatlLevelDw lpmld(dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  if (vars.find(lpmld) == vars.end()) {
    totalVars[dep->mapDataWarehouse()] += 1;

    //contiguous array calculations
    totalSize += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
    totalSizeForDataWarehouse[dep->mapDataWarehouse()] += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
    DeviceGridVariableInfo tmp(var, dest, false, sizeOfDataType, matlIndx, levelIndx, patchPointer, dep, whichGPU);
    vars.insert( std::map<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::value_type( lpmld, tmp ) );
  } else {
    //Don't add the same device var twice.
    printf("ERROR:\n This preparation queue already added this exact variable for label %s patch %d matl %d level %d dw %d\n",dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
    SCI_THROW(InternalError("Preparation queue already contained same exact variable for: -" + dep->var->getName(), __FILE__, __LINE__));
  }
}


bool DeviceGridVariables::varAlreadyExists(const VarLabel* label,
                                       const Patch* patchPointer,
                                       int matlIndx,
                                       int levelIndx,
                                       int dataWarehouse) {

  GpuUtilities::LabelPatchMatlLevelDw lpmld(label->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dataWarehouse);
  std::pair <std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator, std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator> ret;
  ret = vars.equal_range(lpmld);
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it=ret.first; it!=ret.second; ++it) {
    if (it->second.staging == false) {
      printf("Found a copy for %s\n", it->first.label.c_str());
      return true;
    }
  }
  return false;

}

bool DeviceGridVariables::stagingVarAlreadyExists(const VarLabel* label,
                                       const Patch* patchPointer,
                                       int matlIndx,
                                       int levelIndx,
                                       IntVector low,
                                       IntVector size,
                                       int dataWarehouse) {

  GpuUtilities::LabelPatchMatlLevelDw lpmld(label->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dataWarehouse);
  std::pair <std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator, std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator> ret;
  ret = vars.equal_range(lpmld);
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it=ret.first; it!=ret.second; ++it) {
    if (it->second.staging == true && it->second.offset == low && it->second.sizeVector == size) {
      return true;
    }
  }
  return false;

}

//For adding taskVars, which are snapshots of the host-side GPU DW
//This is the normal scenario, no "staging" variables.
void DeviceGridVariables::addTaskGpuDWVar(const Patch* patchPointer,
          int matlIndx,
          int levelIndx,
          size_t sizeOfDataType,
          const Task::Dependency* dep,
          int whichGPU) {
  GpuUtilities::LabelPatchMatlLevelDw lpmld(dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  std::pair <std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator, std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator> ret;
  ret = vars.equal_range(lpmld);
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it=ret.first; it!=ret.second; ++it) {
    if (it->second.staging == false) {
      //Don't add the same device var twice.
      printf("ERROR:\n ::addTaskGpuDWVar() - This preparation queue for a task datawarehouse already added this exact variable for label %s patch %d matl %d level %d dw %d\n",dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
      SCI_THROW(InternalError("Preparation queue for a task datawarehouse already contained same exact variable for: -" + dep->var->getName(), __FILE__, __LINE__));
    }
  }
  totalVars[dep->mapDataWarehouse()] += 1;
  //TODO: The Task DW doesn't hold any pointers.  So what does that mean about contiguous arrays?
  //Should contiguous arrays be organized by task???
  DeviceGridVariableInfo tmp(NULL, GpuUtilities::unknown, false, sizeOfDataType, matlIndx, levelIndx, patchPointer, dep, whichGPU);
  vars.insert( std::map<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::value_type( lpmld, tmp ) );
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " DeviceGridVariables::addTaskGpuDWVar() - "
          << "Added into preparation queue for a task datawarehouse for a variable for label "
          << dep->var->getName()
          << " patch " << patchPointer->getID()
          << " matl " << matlIndx
          << " level " << levelIndx
          << " staging false"
          << " totalVars is: " << totalVars[dep->mapDataWarehouse()] << endl;
    }
    cerrLock.unlock();
  }


  //printf("addTaskGpuDWVar() - Added into preparation queue a for task datawarehouse a variable for label %s patch %d matl %d level %d staging: false, totalVars is: %d\n",
  //        dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, totalVars[dep->mapDataWarehouse()]);
}

//For adding staging taskVars, which are snapshots of the host-side GPU DW.
void DeviceGridVariables::addTaskGpuDWStagingVar(const Patch* patchPointer,
          int matlIndx,
          int levelIndx,
          IntVector offset,
          IntVector sizeVector,
          size_t sizeOfDataType,
          const Task::Dependency* dep,
          int whichGPU) {

  //Since this is a queue, we aren't likely to have the parent variable of the staging var,
  //as that likely went into the regular host-side gpudw as a computes in the last timestep.

  //Just make sure we haven't already added this exact staging variable.


  GpuUtilities::LabelPatchMatlLevelDw lpmld(dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  map<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it = vars.find(lpmld);
  std::pair <std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator, std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator> ret;
  ret = vars.equal_range(lpmld);
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::iterator it=ret.first; it!=ret.second; ++it) {
    if (it->second.staging == true && it->second.sizeVector == sizeVector && it->second.offset == offset) {
      //Don't add the same device var twice.
      printf("ERROR:\n addTaskGpuDWStagingVar() - This preparation queue for a task datawarehouse already added this exact variable for label %s patch %d matl %d level %d dw %d\n",dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
      SCI_THROW(InternalError("addTaskGpuDWStagingVar() - Preparation queue for a task datawarehouse already contained same exact variable for: " + dep->var->getName(), __FILE__, __LINE__));
    }
  }

  totalVars[dep->mapDataWarehouse()] += 1;

  size_t varMemSize = sizeVector.x() * sizeVector.y() * sizeVector.z() * sizeOfDataType;
  DeviceGridVariableInfo tmp(NULL, GpuUtilities::unknown, true, sizeVector, sizeOfDataType, varMemSize , offset, matlIndx, levelIndx, patchPointer, dep, Ghost::None, 0, whichGPU);
  vars.insert( std::map<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::value_type( lpmld, tmp ) );
  if (gpu_stats.active()) {
    cerrLock.lock();
    {
      gpu_stats << UnifiedScheduler::myRankThread()
          << " DeviceGridVariables::addTaskGpuDWStagingVar() - "
          << "Added into preparation queue for a task datawarehouse for a variable for label "
          << dep->var->getName()
          << " patch " << patchPointer->getID()
          << " matl " << matlIndx
          << " level " << levelIndx
          << " staging true"
          << " offset (" << offset.x() << ", " << offset.y() << ", " << offset.z() << ")"
          << " size (" << sizeVector.x() << ", " << sizeVector.y() << ", " << sizeVector.z() << ")"
          << " totalVars is: " << totalVars[dep->mapDataWarehouse()] << endl;
    }
    cerrLock.unlock();
  }
}



DeviceGridVariableInfo DeviceGridVariables::getStagingItem(
    const string& label,
    const Patch* patch,
    const int matlIndx,
    const int levelIndx,
    const IntVector low,
    const IntVector size,
    const int dataWarehouseIndex) const {
  GpuUtilities::LabelPatchMatlLevelDw lpmld(label.c_str(), patch->getID(), matlIndx, levelIndx, dataWarehouseIndex);
  std::pair <std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::const_iterator, std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::const_iterator> ret;
  ret = vars.equal_range(lpmld);
  for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::const_iterator it=ret.first; it!=ret.second; ++it) {
   if (it->second.staging == true && it->second.offset == low && it->second.sizeVector == size) {
     return it->second;
   }
  }

  printf("Error: DeviceGridVariables::getStagingItem(), item not found for offset (%d, %d, %d) size (%d, %d, %d).\n",
      low.x(), low.y(), low.z(), size.x(), size.y(), size.z());
  SCI_THROW(InternalError("Error: DeviceGridVariables::getStagingItem(), item not found for: -" + label, __FILE__, __LINE__));
}

size_t DeviceGridVariables::getTotalSize() {
  return totalSize;
}

size_t DeviceGridVariables::getSizeForDataWarehouse(int dwIndex) {
  return totalSizeForDataWarehouse[dwIndex];
}

unsigned int DeviceGridVariables::numItems() {
  return vars.size();
}

unsigned int DeviceGridVariables::getTotalVars(int DWIndex) const {
  return totalVars[DWIndex];
}

unsigned int DeviceGridVariables::getTotalMaterials(int DWIndex) const {
  return totalMaterials[DWIndex];
}

unsigned int DeviceGridVariables::getTotalLevels(int DWIndex) const {
  return totalLevels[DWIndex];
}


//__________________________________________________________________________
//GpuUtilities methods
//__________________________________________________________________________
void GpuUtilities::assignPatchesToGpus(const GridP& grid){

  currentAcceleratorCounter = 0;
  int numDevices = OnDemandDataWarehouse::getNumDevices();
  if (numDevices > 0) {
    std::map<const Patch *, int>::iterator it;
    for (int i = 0; i < grid->numLevels(); i++) {
      LevelP level = grid->getLevel(i);
      for (Level::patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); ++iter){
        //TODO: Clean up so that instead of assigning round robin, it assigns in blocks.
        const Patch* patch = *iter;
        it = patchAcceleratorLocation.find(patch);
        if (it == patchAcceleratorLocation.end()) {
          //this patch has not been assigned, so assign it.
          //assign it to a gpu in a round robin fashion.
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
int GpuUtilities::getGpuIndexForPatch(const Patch* patch) {

  std::map<const Patch *, int>::iterator it;
  it = patchAcceleratorLocation.find(patch);
  if (it != patchAcceleratorLocation.end()) {
    return it->second;
  }
  return -1;

}
