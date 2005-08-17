
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ConstantThermo.h>

using namespace Uintah;

ConstantThermo::ConstantThermo(ProblemSpecP& ps)
{
  ps->require("thermal_conductivity",d_thermalConductivity);
  ps->require("specific_heat",d_specificHeat);
  ps->require("gamma",d_gamma);
}

ConstantThermo::~ConstantThermo()
{
}

void ConstantThermo::addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                            int numGhostCells)
{
  // No additional requirements
}

void ConstantThermo::addTaskDependencies_Cp(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  // No additional requirements
}

void ConstantThermo::addTaskDependencies_Cv(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  // No additional requirements
}

void ConstantThermo::addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  // No additional requirements
}

void ConstantThermo::addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                           int numGhostCells)
{
  // No additional requirements
}
