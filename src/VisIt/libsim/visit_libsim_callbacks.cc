/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include "VisItControlInterface_V2.h"

#include <CCA/Components/SimulationController/AMRSimulationController.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Regridder.h>

#include <Core/Grid/Grid.h>

namespace Uintah {

#define VISIT_COMMAND_PROCESS 0
#define VISIT_COMMAND_SUCCESS 1
#define VISIT_COMMAND_FAILURE 2

//---------------------------------------------------------------------
// visit_BroadcastIntCallback
//     Callback for processing integers
//---------------------------------------------------------------------
int visit_BroadcastIntCallback(int *value, int sender)
{
  if( Parallel::usingMPI() )
    return MPI_Bcast(value, 1, MPI_INT, sender, MPI_COMM_WORLD);
  else
    return 0;
}


//---------------------------------------------------------------------
// visit_BroadcastStringCallback
//     Callback for processing strings
//---------------------------------------------------------------------
int visit_BroadcastStringCallback(char *str, int len, int sender)
{
  if( Parallel::usingMPI() )
    return MPI_Bcast(str, len, MPI_CHAR, sender, MPI_COMM_WORLD);
  else
    return 0;
}


//---------------------------------------------------------------------
// visit_BroadcastSlaveCommand
//     Helper function for ProcessVisItCommand
//---------------------------------------------------------------------
void visit_BroadcastSlaveCommand(int *command)
{
  if( Parallel::usingMPI() )
    MPI_Bcast(command, 1, MPI_INT, 0, MPI_COMM_WORLD);
}


//---------------------------------------------------------------------
// visit_SlaveProcessCallback
//     Callback involved in command communication.
//---------------------------------------------------------------------
void visit_SlaveProcessCallback()
{
  int command = VISIT_COMMAND_PROCESS;
  visit_BroadcastSlaveCommand(&command);
}


//---------------------------------------------------------------------
// visit_ControlCommandCallback
//     Process user commands from the viewer on all processors.
//---------------------------------------------------------------------
void
visit_ControlCommandCallback(const char *cmd, const char *args, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  if(strcmp(cmd, "StepCycle") == 0 && sim->simMode != VISIT_SIMMODE_FINISHED)
  {
    std::stringstream msg;	  
    msg << "Visit libsim - step cycle value" << args;
    VisItUI_setValueS("SIMULATION_STATUS", msg.str().c_str(), 1);
    VisItUI_setValueS("SIMULATION_STATUS", " ", 1);
  }
  else if(strcmp(cmd, "Stop") == 0 && sim->simMode != VISIT_SIMMODE_FINISHED)
  {
    sim->runMode = VISIT_SIMMODE_STOPPED;
  }
  else if(strcmp(cmd, "Step") == 0 && sim->simMode != VISIT_SIMMODE_FINISHED)
  {
    sim->runMode = VISIT_SIMMODE_STEP;
  }
  else if(strcmp(cmd, "Run") == 0 && sim->simMode != VISIT_SIMMODE_FINISHED)
  {
    sim->runMode = VISIT_SIMMODE_RUNNING;
  }
  else if(strcmp(cmd, "Stats") == 0)
  {
  }
  else if(strcmp(cmd, "Regrid") == 0 && sim->simMode != VISIT_SIMMODE_FINISHED)
  {
    sim->AMRSimController->getRegridder()->setForceRegridding(true);
    sim->runMode = VISIT_SIMMODE_STEP;
  }
  else if(strcmp(cmd, "Save") == 0)
  {
    GridP gridP = sim->gridP;
    Output *output = sim->AMRSimController->getOutput();
    SchedulerP schedulerP = sim->AMRSimController->getSchedulerP();

    if (output)
    {
      output->finalizeTimestep( sim->time, sim->delt, gridP, schedulerP, 0 );
      output->sched_allOutputTasks( sim->delt, gridP, schedulerP, true );

      output->findNext_OutputCheckPoint_Timestep( sim->delt, gridP );
      output->writeto_xml_files( sim->delt, gridP );
    }
  }

  // Only allow the runMode to finish if the simulation is finished.
  else if(strcmp(cmd, "Finish") == 0)
  {
    if(sim->simMode == VISIT_SIMMODE_FINISHED)
      sim->runMode = VISIT_SIMMODE_FINISHED;
    else
      sim->runMode = VISIT_SIMMODE_RUNNING;
  }
  else if(strcmp(cmd, "Terminate") == 0)
  {
    sim->runMode = VISIT_SIMMODE_RUNNING;
    sim->simMode = VISIT_SIMMODE_TERMINATED;

    if(sim->isProc0)
    {
      std::stringstream msg;	  
      msg << "Visit libsim - Terminating the simulation";
      VisItUI_setValueS("SIMULATION_STATUS", msg.str().c_str(), 1);
      VisItUI_setValueS("SIMULATION_STATUS", " ", 1);
    }
  }
  else if(strcmp(cmd, "Abort") == 0)
  {
    if(sim->isProc0)
    {
      std::stringstream msg;	  
      msg << "Visit libsim - Aborting the simulation";
      VisItUI_setValueS("SIMULATION_STATUS", msg.str().c_str(), 1);
      VisItUI_setValueS("SIMULATION_STATUS", " ", 1);
    }

    exit( 0 );
  }

  if( sim->runMode == VISIT_SIMMODE_RUNNING &&
      sim->simMode == VISIT_SIMMODE_RUNNING )
  {
    if(sim->isProc0)
    {
      VisItUI_setValueS("SIMULATION_MODE", "Running", 1);

      std::stringstream msg;	  
      msg << "Visit libsim - Continuing the simulation";
      VisItUI_setValueS("SIMULATION_STATUS", msg.str().c_str(), 1);
    }
  }
}


//---------------------------------------------------------------------
// ProcessVisItCommand
//     Process commands from the viewer on all processors.
//---------------------------------------------------------------------
int visit_ProcessVisItCommand( visit_simulation_data *sim )
{
  if( Parallel::usingMPI() )
  {
    int command;
    
    if(sim->isProc0)
    {
      int success = VisItProcessEngineCommand();
      if(success)
      {
        command = VISIT_COMMAND_SUCCESS;
        visit_BroadcastSlaveCommand(&command);
        return 1;
      }
      else
      {
        command = VISIT_COMMAND_FAILURE;
        visit_BroadcastSlaveCommand(&command);
        return 0;
      }
    }
    else
    {
      /* Note: only through the SlaveProcessCallback callback
       * above can the rank 0 process send a VISIT_COMMAND_PROCESS
       * instruction to the non-rank 0 processes. */
      while(1)
      {
        visit_BroadcastSlaveCommand(&command);
        
        switch(command)
        {
        case VISIT_COMMAND_PROCESS:
          VisItProcessEngineCommand();
          break;
        case VISIT_COMMAND_SUCCESS:
          return 1;
        case VISIT_COMMAND_FAILURE:
          return 0;
        }
      }
    }
  }
  else
  {
    return VisItProcessEngineCommand();
  }

  return 1;
}

} // End namespace Uintah
