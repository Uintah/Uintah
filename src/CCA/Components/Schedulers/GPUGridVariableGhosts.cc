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

#include <CCA/Components/Schedulers/GPUGridVariableGhosts.h>

#include <cstdio>
#include <map>
#include <set>

//______________________________________________________________________
//
void
DeviceGhostCells::clear()
{
  ghostVars.clear();
  totalGhostCellCopies.clear();
}


//______________________________________________________________________
//
std::set<unsigned int>&
DeviceGhostCells::getDestinationDevices()
{
  return destinationDevices;
}


//______________________________________________________________________
//
unsigned int
DeviceGhostCells::numItems() const
{
  return ghostVars.size();
}


//______________________________________________________________________
//
unsigned int
DeviceGhostCells::getNumGhostCellCopies( const unsigned int  whichDevice
                                       ,       Task::WhichDW dwIndex
                                       ) const
{
  std::map<unsigned int, DatawarehouseIds>::const_iterator it = totalGhostCellCopies.find(whichDevice);
  if (it != totalGhostCellCopies.end()) {
    return it->second.DwIds[(unsigned int)dwIndex];
  }
  return 0;
}


//______________________________________________________________________
//
const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>&
DeviceGhostCells::getMap() const
{
  return ghostVars;
}


//______________________________________________________________________
//
void
DeviceGhostCells::add( const VarLabel              * label
                     , const Patch                 * sourcePatchPointer
                     , const Patch                 * destPatchPointer
                     ,       int                     matlIndx
                     ,       int                     levelIndx
                     ,       bool                    sourceStaging
                     ,       bool                    destStaging
                     ,       IntVector               varOffset
                     ,       IntVector               varSize
                     ,       IntVector               low
                     ,       IntVector               high
                     ,       int                     xstride
                     ,       TypeDescription::Type   datatype
                     ,       IntVector               virtualOffset
                     ,       int                     sourceDeviceNum
                     ,       int                     destDeviceNum
                     ,       int                     fromResource
                     ,       int                     toResource
                     ,       Task::WhichDW           dwIndex
                     ,       DeviceVarDest           dest
                     )
{

  // unlike grid variables, we should only have one instance of label/patch/matl/level/dw for patch variables.
  GpuUtilities::GhostVarsTuple gvt(label->getName(), matlIndx, levelIndx, sourcePatchPointer->getID(), destPatchPointer->getID(),
                                   (int)dwIndex, low, high);
  if (ghostVars.find(gvt) == ghostVars.end()) {

    if (totalGhostCellCopies.find(sourceDeviceNum) == totalGhostCellCopies.end()) {

      // initialize the array for number of copies per GPU datawarehouse.
      DatawarehouseIds item;

      for (int i = 0; i < Task::TotalDWs; i++) {
        item.DwIds[i] = 0;
      }
      item.DwIds[dwIndex] = 1;
      totalGhostCellCopies.insert(std::pair<unsigned int, DatawarehouseIds>(sourceDeviceNum, item));
    }
    else {
      totalGhostCellCopies[sourceDeviceNum].DwIds[dwIndex] += 1;
    }

    int deviceID = GpuUtilities::getGpuIndexForPatch(destPatchPointer);
    if (destinationDevices.find(deviceID) == destinationDevices.end()) {
      destinationDevices.insert(deviceID);
    }
    DeviceGhostCellsInfo tmp(label, sourcePatchPointer, destPatchPointer, matlIndx, levelIndx, sourceStaging, destStaging,
                             varOffset, varSize, low, high, xstride, datatype, virtualOffset, sourceDeviceNum, destDeviceNum,
                             fromResource, toResource, dwIndex, dest);
    ghostVars.insert(std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>::value_type(gvt, tmp));
  }
  else {
    // don't add the same device var twice.
    printf("DeviceGhostCells::add() - ERROR:\n This preparation queue for ghost cell copies already added this exact copy for label %s matl %d level %d dw %d\n",
           label->getName().c_str(), matlIndx, levelIndx, (int)dwIndex);
    SCI_THROW(InternalError("DeviceGhostCells::add() - Preparation queue already contained same exact variable for: -" + label->getName(), __FILE__, __LINE__));
  }
}


//______________________________________________________________________
//
DeviceGhostCellsInfo::DeviceGhostCellsInfo( const VarLabel              * label
                                          , const Patch                 * sourcePatchPointer
                                          , const Patch                 * destPatchPointer
                                          ,       int                     matlIndx
                                          ,       int                     levelIndx
                                          ,       bool                    sourceStaging
                                          ,       bool                    destStaging
                                          ,       IntVector               varOffset
                                          ,       IntVector               varSize
                                          ,       IntVector               low
                                          ,       IntVector               high
                                          ,       int                     xstride
                                          ,       TypeDescription::Type   datatype
                                          ,       IntVector               virtualOffset
                                          ,       int                     sourceDeviceNum
                                          ,       int                     destDeviceNum
                                          ,       int                     fromResource  // from-node
                                          ,       int                     toResource    // to-node, needed when preparing contiguous arrays to send off host for MPI
                                          ,       Task::WhichDW           dwIndex
                                          ,       DeviceVarDest           dest
                                          )
  : m_label{label}
  , m_sourcePatchPointer{sourcePatchPointer}
  , m_destPatchPointer{destPatchPointer}
  , m_matlIndx{matlIndx}
  , m_levelIndx{levelIndx}
  , m_sourceStaging{sourceStaging}
  , m_destStaging{destStaging}
  , m_varOffset{varOffset}
  , m_varSize{varSize}
  , m_low{low}
  , m_high{high}
  , m_xstride{xstride}
  , m_datatype{datatype}
  , m_virtualOffset{virtualOffset}
  , m_sourceDeviceNum{sourceDeviceNum}
  , m_destDeviceNum{destDeviceNum}
  , m_fromResource{fromResource}
  , m_toResource{toResource}
  , m_dwIndex{dwIndex}
  , m_dest{dest}
{}

