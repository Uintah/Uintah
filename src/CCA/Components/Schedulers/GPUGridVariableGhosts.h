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

  enum Destination {
    sameDeviceSameMpiRank = 0,
    anotherDeviceSameMpiRank = 1,
    anotherMpiRank = 2
  };

  struct LabelPatchMatlLevelDw {
    std::string     label;
    int        patchID;
    int        matlIndx;
    int        levelIndx;
    int        dataWarehouse;
    LabelPatchMatlLevelDw(const char * label, int patchID, int matlIndx, int levelIndx, int dataWarehouse) {
      this->label = label;
      this->patchID = patchID;
      this->matlIndx = matlIndx;
      this->levelIndx = levelIndx;
      this->dataWarehouse = dataWarehouse;
    }
    //This so it can be used in an STL map
    bool operator<(const LabelPatchMatlLevelDw& right) const {
      if (this->label < right.label) {
        return true;
      } else if (this->label == right.label && (this->patchID < right.patchID)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx < right.matlIndx)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx == right.matlIndx) && (this->levelIndx < right.levelIndx)) {
        return true;
      } else if (this->label == right.label && (this->patchID == right.patchID) && (this->matlIndx == right.matlIndx) && (this->levelIndx == right.levelIndx) && (this->dataWarehouse < right.dataWarehouse)) {
        return true;
      } else {
        return false;
      }

    }
  };

  DeviceGhostCells();

  void add(const VarLabel* label,
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

  set<int>& getDestinationDevices();

  unsigned int numItems() const;

  const VarLabel* getLabel(int index) const;

  char const * getLabelName(int index) const;

  int getMatlIndx(int index) const;

  int getLevelIndx(int index) const;

  const Patch* getSourcePatchPointer(int index) const;

  int getSourceDeviceNum(int index) const;

  const Patch* getDestPatchPointer(int index) const;

  int getDestDeviceNum(int index) const;

  IntVector getLow(int index) const;

  IntVector getHigh(int index) const;

  IntVector getVirtualOffset(int index) const;

  Task::WhichDW getDwIndex(int index) const;

  unsigned int getNumGhostCellCopies(Task::WhichDW dwIndex) const;

  Destination getDestination(int index) const;

private:
  //std::map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGridVariableInfo> vars;

  vector< DeviceGhostCellsInfo > vars;
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
  const VarLabel* label;
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
