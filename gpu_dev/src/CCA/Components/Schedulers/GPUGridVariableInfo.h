#ifndef CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H
#define CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H

#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/Schedulers/GPUDataWarehouse.h>
#include <vector>
#include <map>

using namespace std;
using namespace Uintah;


class DeviceGridVariableInfo {
public:

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

  DeviceGridVariableInfo(Variable* var,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            Ghost::GhostType gtype,
            int numGhostCells,
            int whichGPU);

  //For PerPatch vars
  DeviceGridVariableInfo(Variable* var,
            size_t sizeOfDataType,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            int whichGPU);

  Variable* var;
  IntVector sizeVector;
  size_t sizeOfDataType;
  size_t varMemSize;
  IntVector offset;
  int matlIndx;
  int levelIndx;
  const Patch* patchPointer;
  const Task::Dependency* dep;
  Ghost::GhostType gtype;
  int numGhostCells;
  int whichGPU;
};


class DeviceGridVariables {
public:
  DeviceGridVariables();

  //For grid vars
  void add(const Patch* patchPointer,
            int matlIndx,
            int levelIndx,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            Variable* var,
            const Task::Dependency* dep,
            Ghost::GhostType gtype,
            int numGhostCells,
            int whichGPU);

  //For PerPatch vars.  They don't use ghost cells
  void add(const Patch* patchPointer,
            int matlIndx,
            int levelIndx,
            size_t varMemSize,
            size_t sizeOfDataType,
            Variable* var,
            const Task::Dependency* dep,
            int whichGPU);

  //For task vars.
  void addTaskGpuDWVar(const Patch* patchPointer,
              int matlIndx,
              int levelIndx,
              size_t varMemSize,
              const Task::Dependency* dep,
              int whichGPU);

  Variable* getVariable(const VarLabel* label,
          const Patch* patch,
          const int matlIndx,
          const int levelIndx,
          const int dataWarehouseIndex) const;

  DeviceGridVariableInfo getItem(const VarLabel* label,
          const Patch* patch,
          const int matlIndx,
          const int levelIndx,
          const int dataWarehouseIndex) const;

  size_t getTotalSize();

  size_t getSizeForDataWarehouse(int dwIndex);

  unsigned int numItems();

  unsigned int getTotalVars(int DWIndex) const;
  unsigned int getTotalMaterials(int DWIndex) const;
  unsigned int getTotalLevels(int DWIndex) const;

  std::map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGridVariableInfo>& getMap() {
    return vars;
  }



private:
  size_t totalSize;
  size_t totalSizeForDataWarehouse[Task::TotalDWs];

  std::map<DeviceGridVariableInfo::LabelPatchMatlLevelDw, DeviceGridVariableInfo> vars; //This map acts essentially contains objects
                        //which are first queued up, and then processed in a group.  These DeviceGridVariableInfo objects
                        //can 1) Tell the host-side GPU DW what variables need to be created on the GPU and what copies need
                        //to be made host to deivce.  2) Tell a task GPU DW which variables it needs to know about from
                        //the host-side GPU DW (this task GPU DW gets sent into the GPU).  Or 3) Tells a task GPU DW
                        //the ghost cell copies that need to occur within a GPU.
                        //For #2/#3, it is possible in corner cases or periodic boundary conditions that a source
                        //variable will be used multiple times to extract ghost cell info from.  So we need to verify
                        //if that variable is already in our map prior to inserting it.

  unsigned int totalVars[Task::TotalDWs];
  unsigned int totalMaterials[Task::TotalDWs];
  unsigned int totalLevels[Task::TotalDWs];

};

class GpuUtilities {
public:


  static void assignPatchesToGpus(const GridP& grid);
  static int getGpuIndexForPatch(const Patch* patch);

};

static std::map<const Patch *, int> patchAcceleratorLocation;
static unsigned int currentAcceleratorCounter;

#endif // End CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H
