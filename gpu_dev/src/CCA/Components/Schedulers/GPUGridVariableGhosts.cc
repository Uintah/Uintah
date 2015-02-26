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
#include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>

deviceGhostCellsInfo::deviceGhostCellsInfo(GridVariableBase* gridVar,
          const Patch* sourcePatchPointer,
          int sourceDeviceNum,
          const Patch* destPatchPointer,
          int destDeviceNum,
          int materialIndex,
          IntVector low,
          IntVector high,
          const Task::Dependency* dep) {
  this->gridVar = gridVar;
  this->sourcePatchPointer = sourcePatchPointer;
  this->sourceDeviceNum = sourceDeviceNum;
  this->destPatchPointer = destPatchPointer;
  this->destDeviceNum = destDeviceNum;
  this->materialIndex = materialIndex;
  this->low = low;
  this->high = high;
  this->dep = dep;

}

void deviceGhostCells::add(GridVariableBase* gridVar,
          const Patch* sourcePatchPointer,
          int sourceDeviceNum,
          const Patch* destPatchPointer,
          int destDeviceNum,
          int materialIndex,
          IntVector low,
          IntVector high,
          const Task::Dependency* dep) {
  deviceGhostCellsInfo tmp(gridVar, sourcePatchPointer, sourceDeviceNum, destPatchPointer, destDeviceNum, materialIndex,  low, high, dep);
  vars.push_back(tmp);
}

unsigned int deviceGhostCells::numItems() {
  return vars.size();
}

int deviceGhostCells::getMaterialIndex(int index) {
  return vars.at(index).materialIndex;
}

const Patch* deviceGhostCells::getSourcePatchPointer(int index) {
  return vars.at(index).sourcePatchPointer;
}

int deviceGhostCells::getSourceDeviceNum(int index) {
  return vars.at(index).sourceDeviceNum;
}

const Patch* deviceGhostCells::getDestPatchPointer(int index) {
  return vars.at(index).destPatchPointer;
}

int deviceGhostCells::getDestDeviceNum(int index) {
  return vars.at(index).destDeviceNum;
}

IntVector deviceGhostCells::getLow(int index) {
  return vars.at(index).low;
}
IntVector deviceGhostCells::getHigh(int index) {
  return vars.at(index).high;
}

GridVariableBase* deviceGhostCells::getGridVar(int index) {
  return vars.at(index).gridVar;
}

const Task::Dependency* deviceGhostCells::getDependency(int index) {
  return vars.at(index).dep;
}


