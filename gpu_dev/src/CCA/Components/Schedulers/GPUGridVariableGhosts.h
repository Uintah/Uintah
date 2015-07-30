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

class DeviceGhostCells {
public:

  DeviceGhostCells();

  void add(const VarLabel* label,
      const Patch* sourcePatchPointer,
      const Patch* destPatchPointer,
      int matlIndx,
      int levelIndx,
      bool destStaging,
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

  set<int>& getDestinationDevices();

  unsigned int numItems() const;
/*
  const VarLabel* getLabel(int index) const;

  char const * getLabelName(int index) const;

  int getMatlIndx(int index) const;

  int getLevelIndx(int index) const;

  bool getdestStaging(int index) const;

  const Patch* getSourcePatchPointer(int index) const;

  int getSourceDeviceNum(int index) const;

  const Patch* getDestPatchPointer(int index) const;

  int getDestDeviceNum(int index) const;

  IntVector getLow(int index) const;

  IntVector getHigh(int index) const;

  IntVector getVirtualOffset(int index) const;

  Task::WhichDW getDwIndex(int index) const;
*/
  unsigned int getNumGhostCellCopies(Task::WhichDW dwIndex) const;

  //GpuUtilities::DeviceVarDestination getDestination(int index) const;
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>& getMap() const {
    return ghostVars;
  }

private:
  //std::map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGridVariableInfo> vars;

  //vector< DeviceGhostCellsInfo > vars;
  map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo> ghostVars;
  std::set< int > destinationDevices;
  unsigned int totalGhostCellCopies[Task::TotalDWs];
};

class DeviceGhostCellsInfo {
public:
  DeviceGhostCellsInfo(const VarLabel* label,
      const Patch* sourcePatchPointer,
      const Patch* destPatchPointer,
      int matlIndx,
      int levelIndx,
      bool destStaging,
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
  bool destStaging;
  IntVector low;
  IntVector high;
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
