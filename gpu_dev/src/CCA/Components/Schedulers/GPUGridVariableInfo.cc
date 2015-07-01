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

DeviceGridVariableInfo::DeviceGridVariableInfo(Variable* var,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            bool validOnDevice,
            Ghost::GhostType gtype,
            int numGhostCells,
            int whichGPU) {
  this->var = var;
  this->sizeVector = sizeVector;
  this->sizeOfDataType = sizeOfDataType;
  this->varMemSize = varMemSize;
  this->offset = offset;
  this->matlIndx = matlIndx;
  this->levelIndx = levelIndx;
  this->patchPointer = patchPointer;
  this->dep = dep;
  this->validOnDevice = validOnDevice;
  this->gtype = gtype;
  this->numGhostCells = numGhostCells;
  this->whichGPU = whichGPU;
}

DeviceGridVariableInfo::DeviceGridVariableInfo(Variable* var,
            size_t sizeOfDataType,
            size_t varMemSize,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            bool validOnDevice,
            int whichGPU) {
  this->var = var;
  this->sizeOfDataType = sizeOfDataType;
  this->varMemSize = varMemSize;
  this->matlIndx = matlIndx;
  this->levelIndx = levelIndx;
  this->patchPointer = patchPointer;
  this->dep = dep;
  this->validOnDevice = validOnDevice;
  this->whichGPU = whichGPU;
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
          IntVector sizeVector,
          size_t varMemSize,
          size_t sizeOfDataType,
          IntVector offset,
          Variable* var,
          const Task::Dependency* dep,
          bool validOnDevice,
          Ghost::GhostType gtype,
          int numGhostCells,
          int whichGPU) {
  totalSize += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  totalSizeForDataWarehouse[dep->mapDataWarehouse()] += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  totalVars[dep->mapDataWarehouse()] += 1;
  DeviceGridVariableInfo tmp(var, sizeVector, sizeOfDataType, varMemSize, offset, matlIndx, levelIndx, patchPointer, dep, validOnDevice, gtype, numGhostCells, whichGPU);
  vars.push_back(tmp);
}

void DeviceGridVariables::add(const Patch* patchPointer,
          int matlIndx,
          int levelIndx,
          size_t varMemSize,
          size_t sizeOfDataType,
          Variable* var,
          const Task::Dependency* dep,
          bool validOnDevice,
          int whichGPU) {

  totalSize += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  totalSizeForDataWarehouse[dep->mapDataWarehouse()] += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  totalVars[dep->mapDataWarehouse()] += 1;
  DeviceGridVariableInfo tmp(var, sizeOfDataType, varMemSize, matlIndx, levelIndx, patchPointer, dep, validOnDevice,  whichGPU);
  vars.push_back(tmp);
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


int DeviceGridVariables::getMatlIndx(int index) {
  return vars.at(index).matlIndx;
}

int DeviceGridVariables::getLevelIndx(int index) {
  return vars.at(index).levelIndx;
}

const Patch* DeviceGridVariables::getPatchPointer(int index) {
  return vars.at(index).patchPointer;
}
IntVector DeviceGridVariables::getSizeVector(int index) {
  return vars.at(index).sizeVector;
}
IntVector DeviceGridVariables::getOffset(int index) {
    return vars.at(index).offset;
}

Variable* DeviceGridVariables::getVar(int index) {
  return vars.at(index).var;
}
const Task::Dependency* DeviceGridVariables::getDependency(int index) {
  return vars.at(index).dep;
}

size_t DeviceGridVariables::getSizeOfDataType(int index) {
  return vars.at(index).sizeOfDataType;
}

size_t DeviceGridVariables::getVarMemSize(int index) {
  return vars.at(index).varMemSize;
}

Ghost::GhostType DeviceGridVariables::getGhostType(int index) {
  return vars.at(index).gtype;
}

int DeviceGridVariables::getNumGhostCells(int index) {
  return vars.at(index).numGhostCells;
}

int DeviceGridVariables::getWhichGPU(int index) {
  return vars.at(index).whichGPU;
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
