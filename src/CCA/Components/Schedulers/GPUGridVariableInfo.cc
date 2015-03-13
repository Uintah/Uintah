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

deviceGridVariableInfo::deviceGridVariableInfo(Variable* var,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            int materialIndex,
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
  this->materialIndex = materialIndex;
  this->patchPointer = patchPointer;
  this->dep = dep;
  this->validOnDevice = validOnDevice;
  this->gtype = gtype;
  this->numGhostCells = numGhostCells;
  this->whichGPU = whichGPU;
}

deviceGridVariableInfo::deviceGridVariableInfo(Variable* var,
            size_t sizeOfDataType,
            size_t varMemSize,
            int materialIndex,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            bool validOnDevice,
            int whichGPU) {
  this->var = var;
  this->sizeOfDataType = sizeOfDataType;
  this->varMemSize = varMemSize;
  this->materialIndex = materialIndex;
  this->patchPointer = patchPointer;
  this->dep = dep;
  this->validOnDevice = validOnDevice;
  this->whichGPU = whichGPU;
}


deviceGridVariables::deviceGridVariables() {
  totalSize = 0;
  for (int i = 0; i < Task::TotalDWs; i++) {
    totalSizeForDataWarehouse[i] = 0;
  }
}
void deviceGridVariables::add(const Patch* patchPointer,
          int materialIndex,
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
  deviceGridVariableInfo tmp(var, sizeVector, sizeOfDataType, varMemSize, offset, materialIndex, patchPointer, dep, validOnDevice, gtype, numGhostCells, whichGPU);
  vars.push_back(tmp);
}

void deviceGridVariables::add(const Patch* patchPointer,
          int materialIndex,
          size_t varMemSize,
          size_t sizeOfDataType,
          Variable* var,
          const Task::Dependency* dep,
          bool validOnDevice,
          int whichGPU) {

  totalSize += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  totalSizeForDataWarehouse[dep->mapDataWarehouse()] += ((UnifiedScheduler::bufferPadding - varMemSize % UnifiedScheduler::bufferPadding) % UnifiedScheduler::bufferPadding) + varMemSize;
  deviceGridVariableInfo tmp(var, sizeOfDataType, varMemSize, materialIndex, patchPointer, dep, validOnDevice,  whichGPU);
  vars.push_back(tmp);
}



size_t deviceGridVariables::getTotalSize() {
  return totalSize;
}

size_t deviceGridVariables::getSizeForDataWarehouse(int dwIndex) {
  return totalSizeForDataWarehouse[dwIndex];
}

unsigned int deviceGridVariables::numItems() {
  return vars.size();
}


int deviceGridVariables::getMaterialIndex(int index) {
  return vars.at(index).materialIndex;
}
const Patch* deviceGridVariables::getPatchPointer(int index) {
  return vars.at(index).patchPointer;
}
IntVector deviceGridVariables::getSizeVector(int index) {
  return vars.at(index).sizeVector;
}
IntVector deviceGridVariables::getOffset(int index) {
    return vars.at(index).offset;
}

Variable* deviceGridVariables::getVar(int index) {
  return vars.at(index).var;
}
const Task::Dependency* deviceGridVariables::getDependency(int index) {
  return vars.at(index).dep;
}

size_t deviceGridVariables::getSizeOfDataType(int index) {
  return vars.at(index).sizeOfDataType;
}

size_t deviceGridVariables::getVarMemSize(int index) {
  return vars.at(index).varMemSize;
}

Ghost::GhostType deviceGridVariables::getGhostType(int index) {
  return vars.at(index).gtype;
}

int deviceGridVariables::getNumGhostCells(int index) {
  return vars.at(index).numGhostCells;
}

int deviceGridVariables::getWhichGPU(int index) {
  return vars.at(index).whichGPU;
}
