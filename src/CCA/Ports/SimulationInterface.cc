/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Ports/SimulationInterface.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Task.h>

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
