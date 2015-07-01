#ifndef CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H
#define CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H

#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Grid/Task.h>
#include <vector>

using namespace std;
using namespace Uintah;


class DeviceGridVariableInfo {
public:
  DeviceGridVariableInfo(Variable* var,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            bool validOnDevice,
            Ghost::GhostType gtype,
            int numGhostCells,
            int whichGPU);

  DeviceGridVariableInfo(Variable* var,
            size_t sizeOfDataType,
            size_t varMemSize,
            int matlIndx,
            int levelIndx,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            bool validOnDevice,
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
  bool validOnDevice;
  Ghost::GhostType gtype;
  int numGhostCells;
  int whichGPU;
};


class DeviceGridVariables {
public:
  DeviceGridVariables();
  void add(const Patch* patchPointer,
            int matlIndx,
            int levelIndx,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            Variable* var,
            const Task::Dependency* dep,
            bool validOnDevice,
            Ghost::GhostType gtype,
            int numGhostCells,
            int whichGPU);

  void add(const Patch* patchPointer,
              int matlIndx,
              int levelIndx,
              size_t sizeOfDataType,
              size_t varMemSize,
              Variable* var,
              const Task::Dependency* dep,
              bool validOnDevice,
              int whichGPU);



  size_t getTotalSize();

  size_t getSizeForDataWarehouse(int dwIndex);

  unsigned int numItems();

  int getMatlIndx(int index);

  int getLevelIndx(int index);

  const Patch* getPatchPointer(int index);

  IntVector getSizeVector(int index);

  IntVector getOffset(int index);

  Variable* getVar(int index);

  const Task::Dependency* getDependency(int index);

  size_t getSizeOfDataType(int index);

  size_t getVarMemSize(int index);

  Ghost::GhostType getGhostType(int index);

  int getNumGhostCells(int index);

  int getWhichGPU(int index);

  unsigned int getTotalVars(int DWIndex) const;
  unsigned int getTotalMaterials(int DWIndex) const;
  unsigned int getTotalLevels(int DWIndex) const;


private:
  size_t totalSize;
  size_t totalSizeForDataWarehouse[Task::TotalDWs];
  vector< DeviceGridVariableInfo > vars;
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
