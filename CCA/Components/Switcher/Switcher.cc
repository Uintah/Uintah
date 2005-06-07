#include <Packages/Uintah/CCA/Components/Switcher/Switcher.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/SimpleMaterial.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

Switcher::Switcher(const ProcessorGroup* myworld)
  : UintahParallelComponent(myworld)
{
}

Switcher::~Switcher()
{
}

void Switcher::problemSetup(const ProblemSpecP& params, GridP& grid,
                            SimulationStateP& sharedState)
{
}
 
void Switcher::scheduleInitialize(const LevelP& level,
                                  SchedulerP& sched)
{
  SimulationInterface* sim = 
    dynamic_cast<SimulationInterface*>(getPort("sim",0));
  if (sim)
    sim->scheduleInitialize(level,sched);
}
 
void Switcher::scheduleComputeStableTimestep(const LevelP& level,
                                             SchedulerP& sched)
{
  SimulationInterface* sim = 
    dynamic_cast<SimulationInterface*>(getPort("sim",0));
  if (sim)
    sim->scheduleComputeStableTimestep(level,sched);
}

void
Switcher::scheduleTimeAdvance(const LevelP& level, SchedulerP& sched,
                              int, int )
{
}

void Switcher::computeStableTimestep(const ProcessorGroup* pg,
				     const PatchSubset* /*patches*/,
				     const MaterialSubset* /*matls*/,
				     DataWarehouse*,
				     DataWarehouse* new_dw)
{
}

void Switcher::initialize(const ProcessorGroup* pg,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw, DataWarehouse* new_dw)
{
}


void Switcher::timeAdvance(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw, DataWarehouse* new_dw)
{

}
