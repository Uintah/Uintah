
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/Core/Grid/Task.h>

using namespace Uintah;
using namespace SCIRun;

SimulationInterface::SimulationInterface()
{
}

SimulationInterface::~SimulationInterface()
{
}

void
SimulationInterface::scheduleRefine(const PatchSet*, 
                                    SchedulerP&)
{
  throw InternalError("scheduleRefine not implemented for this component\n", __FILE__, __LINE__);
}

void
SimulationInterface::scheduleRefineInterface(const LevelP&, 
                                             SchedulerP&,
                                             bool, bool)
{
  throw InternalError("scheduleRefineInterface not implemented for this component\n",
                      __FILE__, __LINE__);
}

void
SimulationInterface::scheduleCoarsen(const LevelP&, 
                                     SchedulerP&)
{
  throw InternalError("scheduleCoarsen not implemented for this component\n", __FILE__, __LINE__);
}

void
SimulationInterface::scheduleTimeAdvance(const LevelP&,
                                         SchedulerP&)
{
  throw InternalError("no simulation implemented?", __FILE__, __LINE__);
}

void
SimulationInterface::scheduleErrorEstimate(const LevelP&,
						SchedulerP&)
{
  throw InternalError("scheduleErrorEstimate not implemented for this component",
                      __FILE__, __LINE__);
}

void
SimulationInterface::scheduleInitialErrorEstimate(const LevelP& /*coarseLevel*/,
                                                  SchedulerP& /*sched*/)
{
  throw InternalError("scheduleInitialErrorEstimate not implemented for this component",
                      __FILE__, __LINE__);
}

double
SimulationInterface::recomputeTimestep(double)
{
  throw InternalError("recomputeTimestep not implemented for this component", __FILE__, __LINE__);
}

bool
SimulationInterface::restartableTimesteps()
{
  return false;
}

void
SimulationInterface::addMaterial(const ProblemSpecP& /*params*/, GridP& /*grid*/,
                                 SimulationStateP& /*state*/)
{
  throw InternalError("addMaterial not implemented for this component", __FILE__, __LINE__);
}

void
SimulationInterface::scheduleInitializeAddedMaterial(const LevelP&
                                                     coarseLevel,
                                                     SchedulerP& /*sched*/)
{
  throw InternalError("scheduleInitializeAddedMaterial not implemented for this component",
                      __FILE__, __LINE__);
}

double
SimulationInterface::getSubCycleProgress(DataWarehouse* fineDW)
{
  // DWs are always created in order of time.
  int fineID = fineDW->getID();  
  int coarseNewID = fineDW->getOtherDataWarehouse(Task::CoarseNewDW)->getID();
  // need to do this check, on init timestep, old DW is NULL, and getOtherDW will throw exception
  if (fineID == coarseNewID)
    return 1.0; 
  int coarseOldID = fineDW->getOtherDataWarehouse(Task::CoarseOldDW)->getID();
  
  return ((double)fineID-coarseOldID) / (coarseNewID-coarseOldID);
}
