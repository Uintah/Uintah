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

DeviceGhostCellsInfo::DeviceGhostCellsInfo(const VarLabel* label,
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

void DeviceGhostCells::add(const VarLabel* label,
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

  //DeviceGhostCellsInfo::LabelPatchMatlLevelDw lpmld(label->getName().c_str(), destPatchPointer->getID(), matlIndx, levelIndx, dwIndex);
  //if (vars.find(lpmld) == vars.end()) {
      totalGhostCellCopies[dwIndex] += 1;
      DeviceGhostCellsInfo tmp(label, sourcePatchPointer, destPatchPointer, matlIndx, levelIndx,
                                 low, high, xstride, virtualOffset, sourceDeviceNum, destDeviceNum,
                                 fromResource, toResource, dwIndex, dest);
  //    vars.insert( std::map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGhostCellsInfo>::value_type( lpmld, tmp ) );
  //} else {

    //TODO: merge sizeVector into what we've got.
    //Don't add the same device var twice.
  //  printf("ERROR:\n This destination is already receiving a ghost cell copy.   queue already added a variable for label %s patch %d matl %d level %d dw %d\n",dep->var->getName().c_str(), patchPointer->getID(), matlIndx, levelIndx, dep->mapDataWarehouse());
  //  exit(-1);
  //}


  int deviceID = GpuUtilities::getGpuIndexForPatch(destPatchPointer);
  if (destinationDevices.find(deviceID) == destinationDevices.end()) {
    destinationDevices.insert(deviceID);
  }

  vars.push_back(tmp);

}

set<int>& DeviceGhostCells::getDestinationDevices() {
  return destinationDevices;
}

unsigned int DeviceGhostCells::numItems() const {
  return vars.size();
}

const VarLabel* DeviceGhostCells::getLabel(int index) const {
  return vars.at(index).label;
}
char const * DeviceGhostCells::getLabelName(int index) const {
  return vars.at(index).label->getName().c_str();
}

int DeviceGhostCells::getMatlIndx(int index) const {
  return vars.at(index).matlIndx;
}

int DeviceGhostCells::getLevelIndx(int index) const {
  return vars.at(index).levelIndx;
}

const Patch* DeviceGhostCells::getSourcePatchPointer(int index) const {
  return vars.at(index).sourcePatchPointer;
}

int DeviceGhostCells::getSourceDeviceNum(int index) const {
  return vars.at(index).sourceDeviceNum;
}

const Patch* DeviceGhostCells::getDestPatchPointer(int index) const {
  return vars.at(index).destPatchPointer;
}

int DeviceGhostCells::getDestDeviceNum(int index) const {
  return vars.at(index).destDeviceNum;
}

IntVector DeviceGhostCells::getLow(int index) const {
  return vars.at(index).low;
}
IntVector DeviceGhostCells::getHigh(int index) const {
  return vars.at(index).high;
}

IntVector DeviceGhostCells::getVirtualOffset(int index) const {
  return vars.at(index).virtualOffset;
}

Task::WhichDW DeviceGhostCells::getDwIndex(int index) const {
  return vars.at(index).dwIndex;
}

unsigned int DeviceGhostCells::getNumGhostCellCopies(Task::WhichDW dwIndex) const {
  return totalGhostCellCopies[dwIndex];
}

DeviceGhostCells::Destination DeviceGhostCells::getDestination(int index) const {
  return vars.at(index).dest;
}


