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

DeviceGhostCellsInfo::DeviceGhostCellsInfo(char const* label,
    const Patch* sourcePatchPointer,
    const Patch* destPatchPointer,
    int matlIndx,
    int levelIndx,
    IntVector low,
    IntVector high,
    int xstride,
    IntVector virtualOffset,
    int sourceDeviceNum,
    int destDeviceNum,
    int fromResource,  //fromNode
    int toResource,    //toNode, needed when preparing contiguous arrays to send off host for MPI
    Task::WhichDW dwIndex,
    DeviceGhostCells::Destination dest) {
  this->label = label;
  this->sourcePatchPointer = sourcePatchPointer;
  this->destPatchPointer = destPatchPointer;
  this->matlIndx = matlIndx;
  this->levelIndx = levelIndx;
  this->low = low;
  this->high = high;
  this->xstride = xstride;
  this->virtualOffset = virtualOffset;
  this->sourceDeviceNum = sourceDeviceNum;
  this->destDeviceNum = destDeviceNum;
  this->fromResource = fromResource;
  this->toResource = toResource;
  this->dwIndex = dwIndex;
  this->dest = dest;
}

DeviceGhostCells::DeviceGhostCells() {
  for (int i = 0; i < Task::TotalDWs; i++) {
    totalGhostCellCopies[i] = 0;
  }
}

void DeviceGhostCells::add(char const* label,
          const Patch* sourcePatchPointer,
          const Patch* destPatchPointer,
          int matlIndx,
          int levelIndx,
          IntVector low,
          IntVector high,
          int xstride,
          IntVector virtualOffset,
          int sourceDeviceNum,
          int destDeviceNum,
          int fromResource,  //fromNode
          int toResource,
          Task::WhichDW dwIndex,
          Destination dest) {   //toNode, needed when preparing contiguous arrays to send off host for MPI
  DeviceGhostCellsInfo tmp(label, sourcePatchPointer, destPatchPointer, matlIndx, levelIndx,
                           low, high, xstride, virtualOffset, sourceDeviceNum, destDeviceNum,
                           fromResource, toResource, dwIndex, dest);
  totalGhostCellCopies[dwIndex] += 1;
  vars.push_back(tmp);
}


unsigned int DeviceGhostCells::numItems() {
  return vars.size();
}

char const * DeviceGhostCells::getLabelName(int index) {
  return vars.at(index).label;
}

int DeviceGhostCells::getMatlIndx(int index) {
  return vars.at(index).matlIndx;
}

int DeviceGhostCells::getLevelIndx(int index) {
  return vars.at(index).levelIndx;
}

const Patch* DeviceGhostCells::getSourcePatchPointer(int index) {
  return vars.at(index).sourcePatchPointer;
}

int DeviceGhostCells::getSourceDeviceNum(int index) {
  return vars.at(index).sourceDeviceNum;
}

const Patch* DeviceGhostCells::getDestPatchPointer(int index) {
  return vars.at(index).destPatchPointer;
}

int DeviceGhostCells::getDestDeviceNum(int index) {
  return vars.at(index).destDeviceNum;
}

IntVector DeviceGhostCells::getLow(int index) {
  return vars.at(index).low;
}
IntVector DeviceGhostCells::getHigh(int index) {
  return vars.at(index).high;
}

IntVector DeviceGhostCells::getVirtualOffset(int index) {
  return vars.at(index).virtualOffset;
}

Task::WhichDW DeviceGhostCells::getDwIndex(int index) {
  return vars.at(index).dwIndex;
}

unsigned int DeviceGhostCells::getNumGhostCellCopies(Task::WhichDW dwIndex) const {
  return totalGhostCellCopies[dwIndex];
}

