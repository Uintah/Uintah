
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

void SimulationInterface::scheduleRefine(const LevelP& fineLevel, 
					 SchedulerP& scheduler)
{
  throw InternalError("scheduleRefine not implemented for this component\n");
}

void SimulationInterface::scheduleRefineInterface(const LevelP& fineLevel, 
						  SchedulerP& scheduler,
						  int step, int nsteps)
{
  throw InternalError("scheduleRefineInterface not implemented for this component\n");
}

void SimulationInterface::scheduleCoarsen(const LevelP& coarseLevel, 
					  SchedulerP& scheduler)
{
  throw InternalError("scheduleCoarsen not implemented for this component\n");
}

void SimulationInterface::scheduleTimeAdvance(const LevelP& level,
					      SchedulerP& sched,
					      int step, int nsteps)
{
  throw InternalError("no simulation implemented?");
}

void SimulationInterface::scheduleErrorEstimate(const LevelP& level,
						SchedulerP& sched)
{
  throw InternalError("scheduleErrorEstimate not implemented for this component\n");
}
