
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

void SimulationInterface::scheduleRefine(const LevelP&, 
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
  throw InternalError("scheduleErrorEstimate not implemented for this component\n");
}
