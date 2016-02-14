/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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
#include <stdio.h>

DeviceGhostCellsInfo::DeviceGhostCellsInfo(const VarLabel* label,
    const Patch* sourcePatchPointer,
    const Patch* destPatchPointer,
    int matlIndx,
    int levelIndx,
    bool sourceStaging,
    bool destStaging,
    IntVector varOffset,
    IntVector varSize,
    IntVector low,
    IntVector high,
    int xstride,
    IntVector virtualOffset,
    int sourceDeviceNum,
    int destDeviceNum,
    int fromResource,  //fromNode
    int toResource,    //toNode, needed when preparing contiguous arrays to send off host for MPI
    Task::WhichDW dwIndex,
    GpuUtilities::DeviceVarDestination dest) {
  this->label = label;
  this->sourcePatchPointer = sourcePatchPointer;
  this->destPatchPointer = destPatchPointer;
  this->matlIndx = matlIndx;
  this->levelIndx = levelIndx;
  this->sourceStaging = sourceStaging,
  this->destStaging = destStaging,
  this->varOffset = varOffset,
  this->varSize = varSize,
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

DeviceGhostCells::DeviceGhostCells() {}

void DeviceGhostCells::add(const VarLabel* label,
          const Patch* sourcePatchPointer,
          const Patch* destPatchPointer,
          int matlIndx,
          int levelIndx,
          bool sourceStaging,
          bool destStaging,
          IntVector varOffset,
          IntVector varSize,
          IntVector low,
          IntVector high,
          int xstride,
          IntVector virtualOffset,
          int sourceDeviceNum,
          int destDeviceNum,
          int fromResource,  //fromNode
          int toResource,
          Task::WhichDW dwIndex,
          GpuUtilities::DeviceVarDestination dest) {   //toNode, needed when preparing contiguous arrays to send off host for MPI

  //unlike grid variables, we should only have one instance of label/patch/matl/level/dw for patch variables.
  GpuUtilities::GhostVarsTuple gvt(label->getName(), matlIndx, levelIndx, sourcePatchPointer->getID(), destPatchPointer->getID(), (int)dwIndex, low, high);
  if (ghostVars.find(gvt) == ghostVars.end()) {

    if (totalGhostCellCopies.find(sourceDeviceNum) == totalGhostCellCopies.end() ) {

      //initialize the array for number of copies per GPU datawarehouse.
      DatawarehouseIds item;

      for (int i = 0; i < Task::TotalDWs; i++) {
        item.DwIds[i] = 0;
      }

      item.DwIds[dwIndex] = 1;

      totalGhostCellCopies.insert(std::pair<unsigned int, DatawarehouseIds>(sourceDeviceNum, item));

     } else {

       totalGhostCellCopies[sourceDeviceNum].DwIds[dwIndex] += 1;

     }

    int deviceID = GpuUtilities::getGpuIndexForPatch(destPatchPointer);
    if (destinationDevices.find(deviceID) == destinationDevices.end()) {
      destinationDevices.insert(deviceID);
    }
    DeviceGhostCellsInfo tmp(label, sourcePatchPointer, destPatchPointer, matlIndx, levelIndx, sourceStaging, destStaging,
                               varOffset, varSize, low, high, xstride, virtualOffset, sourceDeviceNum, destDeviceNum,
                               fromResource, toResource, dwIndex, dest);
    ghostVars.insert( std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::value_type( gvt, tmp ) );
  } else {
    //Don't add the same device var twice.
    printf("ERROR:\n This preparation queue for ghost cell copies already added this exact copy for label %s matl %d level %d dw %d\n",label->getName().c_str(), matlIndx, levelIndx, (int)dwIndex);
    SCI_THROW(InternalError("Preparation queue already contained same exact variable for: -" + label->getName(), __FILE__, __LINE__));
  }




  /*
  //DeviceGhostCellsInfo::LabelPatchMatlLevelDw lpmld(label->getName().c_str(), destPatchPointer->getID(), matlIndx, levelIndx, dwIndex);
  //if (vars.find(lpmld) == vars.end()) {
      totalGhostCellCopies[dwIndex] += 1;
      DeviceGhostCellsInfo tmp(label, sourcePatchPointer, destPatchPointer, matlIndx, levelIndx, destStaging,
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
  */
}
/*
DeviceGhostCellsInfo DeviceGhostCells::getItem(
    const VarLabel* label,
    const Patch* patch,
    const int matlIndx,
    const int levelIndx,
    const IntVector low,
    const IntVector size,
    const int dataWarehouseIndex) const {
  GpuUtilities::LabelPatchMatlLevelDw lpmld(label->getName().c_str(), patch->getID(), matlIndx, levelIndx, dataWarehouseIndex);
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>::const_iterator it = vars.find(lpmld);
  while (it != vars.end()) {
     if (it->second.staging == true && it->second.offset == low && it->second.sizeVector == size) {
       return it->second;
     }
     ++it;
   }

  printf("Error: DeviceGridVariables::getStagingItem(), item not found for offset (%d, %d, %d) size (%d, %d, %d).\n",
      low.x(), low.y(), low.z(), size.x(), size.y(), size.z());
  SCI_THROW(InternalError("Error: DeviceGridVariables::getStagingItem(), item not found for: -" + label->getName(), __FILE__, __LINE__));
}
*/
set<unsigned int>& DeviceGhostCells::getDestinationDevices() {
  return destinationDevices;
}

unsigned int DeviceGhostCells::numItems() const {
  return ghostVars.size();
}
unsigned int DeviceGhostCells::getNumGhostCellCopies(const unsigned int whichDevice, Task::WhichDW dwIndex) const {
  std::map<unsigned int, DatawarehouseIds>::const_iterator it;
  it = totalGhostCellCopies.find(whichDevice);
  if(it != totalGhostCellCopies.end()) {
      return it->second.DwIds[(unsigned int)dwIndex];
  }
  return 0;

}



