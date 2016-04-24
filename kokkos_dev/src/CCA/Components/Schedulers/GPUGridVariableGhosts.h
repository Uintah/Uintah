#ifndef CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H
#define CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H

#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <CCA/Components/Schedulers/GPUGridVariableInfo.h> //For GpuUtilities
#include <Core/Datatypes/TypeName.h>
#include <Core/Grid/Task.h>
#include <vector>

using namespace std;
using namespace Uintah;

class DeviceGhostCells;
class DeviceGhostCellsInfo;

struct DatawarehouseIds {
  unsigned int DwIds[Task::TotalDWs];
};

class DeviceGhostCells {
public:

  DeviceGhostCells();

  void clear();

  void add(const VarLabel* label,
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
      GpuUtilities::DeviceVarDestination dest);    //toNode, needed when preparing contiguous arrays to send off host for MPI

  set<unsigned int>& getDestinationDevices();

  unsigned int numItems() const;

  unsigned int getNumGhostCellCopies(const unsigned int whichDevice, Task::WhichDW dwIndex) const;

  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>& getMap() const {
    return ghostVars;
  }

private:

  map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> ghostVars;

  std::set< unsigned int > destinationDevices;  //Which devices.

  std::map <unsigned int, DatawarehouseIds> totalGhostCellCopies;  //Total per device.

};

class DeviceGhostCellsInfo {
public:
  DeviceGhostCellsInfo(const VarLabel* label,
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
      GpuUtilities::DeviceVarDestination dest);

  const VarLabel* label;
  const Patch* sourcePatchPointer;
  const Patch* destPatchPointer;
  int matlIndx;
  int levelIndx;
  bool sourceStaging;
  bool destStaging;
  IntVector varOffset;  //The low coordinate of the actual variable
  IntVector varSize;    //The size of the actual variable
  IntVector low;        //The low coordinate within the variable we're copying
  IntVector high;       //the high coordinate within the region of the variable we're copying
  int xstride;
  IntVector virtualOffset;
  int sourceDeviceNum;
  int destDeviceNum;
  int fromResource;  //fromNode
  int toResource;    //toNode, needed when preparing contiguous arrays to send off host for MPI
  Task::WhichDW dwIndex;
  GpuUtilities::DeviceVarDestination dest;
};


#endif // End CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H
