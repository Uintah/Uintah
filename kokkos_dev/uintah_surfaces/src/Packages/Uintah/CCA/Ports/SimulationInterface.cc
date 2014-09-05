
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCIRun;

SimulationInterface::SimulationInterface()
{
}

SimulationInterface::~SimulationInterface()
{
}

void SimulationInterface::scheduleRefine(const PatchSet*, 
					 SchedulerP&)
{
  throw InternalError("scheduleRefine not implemented for this component\n");
}

void SimulationInterface::scheduleRefineInterface(const LevelP&, 
						  SchedulerP&,
						  int, int)
{
  throw InternalError("scheduleRefineInterface not implemented for this component\n");
}

void SimulationInterface::scheduleCoarsen(const LevelP&, 
					  SchedulerP&)
{
  throw InternalError("scheduleCoarsen not implemented for this component\n");
}

void SimulationInterface::scheduleTimeAdvance(const LevelP&,
					      SchedulerP&,
					      int, int)
{
  throw InternalError("no simulation implemented?");
}

void SimulationInterface::scheduleErrorEstimate(const LevelP&,
						SchedulerP&)
{
  throw InternalError("scheduleErrorEstimate not implemented for this component");
}

void SimulationInterface::scheduleInitialErrorEstimate(const LevelP& coarseLevel,
                                                       SchedulerP& sched)
{
  throw InternalError("scheduleInitialErrorEstimate not implemented for this component");
}

double SimulationInterface::recomputeTimestep(double)
{
  throw InternalError("recomputeTimestep not implemented for this component");
}

bool SimulationInterface::restartableTimesteps()
{
  return false;
}

void SimulationInterface::addMaterial(const ProblemSpecP& params, GridP& grid,
                                      SimulationStateP& state)
{
  throw InternalError("addMaterial not implemented for this component");
}

void SimulationInterface::scheduleInitializeAddedMaterial(const LevelP&
                                                                coarseLevel,
                                                          SchedulerP& sched)
{
  throw InternalError("scheduleInitializeAddedMaterial not implemented for this component");
}
