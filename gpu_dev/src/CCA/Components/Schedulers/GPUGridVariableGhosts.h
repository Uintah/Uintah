#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/GridVariableBase.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Grid/Task.h>
#include <vector>

using namespace std;
using namespace Uintah;


class deviceGhostCellsInfo {
public:
  deviceGhostCellsInfo(GridVariableBase* gridVar,
      const Patch* sourcePatchPointer,
      int sourceDeviceNum,
      const Patch* destPatchPointer,
      int destDeviceNum,
      int materialIndex,
      IntVector low,
      IntVector high,
      const Task::Dependency* dep,
      IntVector virtualOffset);
  GridVariableBase* gridVar;
  const Patch* sourcePatchPointer;
  int sourceDeviceNum;
  const Patch* destPatchPointer;
  int destDeviceNum;
  int materialIndex;
  IntVector low;
  IntVector high;
  IntVector virtualOffset;
  const Task::Dependency* dep;
};


class deviceGhostCells {
public:
  void add(GridVariableBase* gridVar,
      const Patch* sourcePatchPointer,
      int sourceDeviceNum,
      const Patch* destPatchPointer,
      int destDeviceNum,
      int materialIndex,
      IntVector low,
      IntVector high,
      const Task::Dependency* dep,
      IntVector virtualOffset);

  unsigned int numItems();

  int getMaterialIndex(int index);

  const Patch* getSourcePatchPointer(int index);

  int getSourceDeviceNum(int index);

  const Patch* getDestPatchPointer(int index);

  int getDestDeviceNum(int index);

  IntVector getLow(int index);

  IntVector getHigh(int index);

  GridVariableBase* getGridVar(int index);

  const Task::Dependency* getDependency(int index);

  IntVector getVirtualOffset(int index);

private:
  vector< deviceGhostCellsInfo > vars;
};

