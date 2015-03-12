
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Grid/Task.h>
#include <vector>

using namespace std;
using namespace Uintah;


class deviceGridVariableInfo {
public:
  deviceGridVariableInfo(Variable* var,
            IntVector sizeVector,
            size_t sizeOfDataType,
            size_t varMemSize,
            IntVector offset,
            int materialIndex,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            bool validOnDevice,
            Ghost::GhostType gtype,
            int numGhostCells,
            int whichGPU);

  deviceGridVariableInfo(Variable* var,
            size_t sizeOfDataType,
            size_t varMemSize,
            int materialIndex,
            const Patch* patchPointer,
            const Task::Dependency* dep,
            bool validOnDevice,
            int whichGPU);

  Variable* var;
  IntVector sizeVector;
  size_t sizeOfDataType;
  size_t varMemSize;
  IntVector offset;
  int materialIndex;
  const Patch* patchPointer;
  const Task::Dependency* dep;
  bool validOnDevice;
  Ghost::GhostType gtype;
  int numGhostCells;
  int whichGPU;
};


class deviceGridVariables {
public:
  deviceGridVariables();
  void add(const Patch* patchPointer,
            int materialIndex,
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
              int materialIndex,
              size_t sizeOfDataType,
              size_t varMemSize,
              Variable* var,
              const Task::Dependency* dep,
              bool validOnDevice,
              int whichGPU);

  size_t getTotalSize();

  size_t getSizeForDataWarehouse(int dwIndex);

  unsigned int numItems();

  int getMaterialIndex(int index);

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


private:
  size_t totalSize;
  size_t totalSizeForDataWarehouse[Task::TotalDWs];
  vector< deviceGridVariableInfo > vars;
};

