#ifndef CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H
#define CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H

#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Grid/Task.h>
#include <vector>

using namespace std;
using namespace Uintah;

class DeviceGhostCells;
class DeviceGhostCellsInfo;

class DeviceGhostCells {
public:

  enum Destination {
    sameDeviceSameNode = 0,
    anotherDeviceSameNode = 1,
    anotherNode = 2
  };

  DeviceGhostCells();

  void add(char const* label,
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
      Destination dest);    //toNode, needed when preparing contiguous arrays to send off host for MPI

  unsigned int numItems();

  char const * getLabelName(int index);

  int getMatlIndx(int index);

  int getLevelIndx(int index);

  const Patch* getSourcePatchPointer(int index);

  int getSourceDeviceNum(int index);

  const Patch* getDestPatchPointer(int index);

  int getDestDeviceNum(int index);

  IntVector getLow(int index);

  IntVector getHigh(int index);

  IntVector getVirtualOffset(int index);

  Task::WhichDW getDwIndex(int index);

  unsigned int getNumGhostCellCopies(Task::WhichDW dwIndex) const;

private:
  vector< DeviceGhostCellsInfo > vars;
  unsigned int totalGhostCellCopies[Task::TotalDWs];
};

class DeviceGhostCellsInfo {
public:
  DeviceGhostCellsInfo(char const* label,
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
      DeviceGhostCells::Destination dest);
  char const* label;
  const Patch* sourcePatchPointer;
  const Patch* destPatchPointer;
  int matlIndx;
  int levelIndx;
  IntVector low;
  IntVector high;
  int xstride;
  IntVector virtualOffset;
  int sourceDeviceNum;
  int destDeviceNum;
  int fromResource;  //fromNode
  int toResource;    //toNode, needed when preparing contiguous arrays to send off host for MPI
  Task::WhichDW dwIndex;
  DeviceGhostCells::Destination dest;
};


#endif // End CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEGHOSTS_H
