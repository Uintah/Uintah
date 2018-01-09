/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H
#define CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H

#include <CCA/Components/Schedulers/GPUGridVariableInfo.h> //For GpuUtilities

#include <Core/Datatypes/TypeName.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Disclosure/TypeDescription.h>

#include <map>
#include <set>

using namespace Uintah;

class DeviceGhostCells;
class DeviceGhostCellsInfo;

class DeviceGhostCells {

using DeviceVarDest = GpuUtilities::DeviceVarDestination;

public:

  DeviceGhostCells() = default;

  ~DeviceGhostCells(){};

  void clear();

  std::set<unsigned int>& getDestinationDevices();

  unsigned int numItems() const;

  unsigned int getNumGhostCellCopies(const unsigned int whichDevice, Task::WhichDW dwIndex) const;

  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>& getMap() const;

  void add( const VarLabel              * label
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
          ,       int                     fromResource   // from-node
          ,       int                     toResource
          ,       Task::WhichDW           dwIndex
          ,       DeviceVarDest           dest           // to-node, needed when preparing contiguous arrays to send off host for MPI
          );


private:

  struct DatawarehouseIds {
    unsigned int DwIds[Task::TotalDWs];
  };

  std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> ghostVars;

  std::set<unsigned int > destinationDevices;                      // which devices.

  std::map <unsigned int, DatawarehouseIds> totalGhostCellCopies;  // total per device.

};


class DeviceGhostCellsInfo {

using DeviceVarDest = GpuUtilities::DeviceVarDestination;

public:

  DeviceGhostCellsInfo( const VarLabel              * label
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
                      ,       DeviceVarDest           des
                      );

  const VarLabel             * m_label;
  const Patch                * m_sourcePatchPointer;
  const Patch                * m_destPatchPointer;
        int                    m_matlIndx;
        int                    m_levelIndx;
        bool                   m_sourceStaging;
        bool                   m_destStaging;
        IntVector              m_varOffset;        // the low coordinate of the actual variable
        IntVector              m_varSize;          // the size of the actual variable
        IntVector              m_low;              // the low coordinate within the variable we're copying
        IntVector              m_high;             // the high coordinate within the region of the variable we're copying
        int                    m_xstride;
        TypeDescription::Type  m_datatype;
        IntVector              m_virtualOffset;
        int                    m_sourceDeviceNum;
        int                    m_destDeviceNum;
        int                    m_fromResource;     // from-node
        int                    m_toResource;       // to-node, needed when preparing contiguous arrays to send off host for MPI
        Task::WhichDW          m_dwIndex;
        DeviceVarDest          m_dest;
};


#endif // End CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H
