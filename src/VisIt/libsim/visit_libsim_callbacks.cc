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

#include "visit_libsim_customUI.h"

#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Components/DataArchiver/DataArchiver.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Util/DebugStream.h>

#include <stdio.h>

#include "visit_libsim.h"

namespace Uintah {

static SCIRun::DebugStream visitdbg( "VisItLibSim", true );

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

  if(strcmp(cmd, "Stop") == 0 && sim->simMode != VISIT_SIMMODE_FINISHED)
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
  else if(strcmp(cmd, "Save") == 0)
  {
    // Do not call unless the simulation is stopped or finished as it
    // will interfer with the task graph.
    if(sim->simMode == VISIT_SIMMODE_STOPPED ||
       sim->simMode == VISIT_SIMMODE_FINISHED)
    {
      VisItUI_setValueS("SIMULATION_ENABLE_BUTTON", "Save", 0);

      Output *output = sim->simController->getOutput();
      SchedulerP schedulerP = sim->simController->getSchedulerP();
      
      ((DataArchiver *)output)->outputTimestep( sim->time,
                                                sim->delt,
                                                sim->gridP,
                                                schedulerP );
    }
    else
      VisItUI_setValueS("SIMULATION_MESSAGE_BOX",
                        "Can not save a timestep unless the simulation is stopped", 0);
  }

  else if(strcmp(cmd, "Checkpoint") == 0)
  {
    // Do not call unless the simulation is stopped or finished as it
    // will interfer with the task graph.
    if(sim->simMode == VISIT_SIMMODE_STOPPED ||
       sim->simMode == VISIT_SIMMODE_FINISHED)
    {
      VisItUI_setValueS("SIMULATION_ENABLE_BUTTON", "Checkpoint", 0);

      Output *output = sim->simController->getOutput();
      SchedulerP schedulerP = sim->simController->getSchedulerP();
      
      ((DataArchiver *)output)->checkpointTimestep( sim->time,
                                                    sim->delt,
                                                    sim->gridP,
                                                    schedulerP );
    }
    else
      VisItUI_setValueS("SIMULATION_MESSAGE_BOX",
                        "Can not save a checkpoint unless the simulation is stopped", 0);
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
      VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
      VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);

      visitdbg << msg.str().c_str() << std::endl;
      visitdbg.flush();
    }
  }
  else if(strcmp(cmd, "Abort") == 0)
  {
    if(sim->isProc0)
    {
      std::stringstream msg;      
      msg << "Visit libsim - Aborting the simulation";
      VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
      VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);

      visitdbg << msg.str().c_str() << std::endl;
      visitdbg.flush();
    }

    exit( 0 );
  }
  else if(strcmp(cmd, "TimeLimitsEnabled") == 0 )
  {
    std::string varStr = std::string(args);
    size_t found = varStr.find_last_of(";");
    varStr = varStr.substr(found + 1);

    sim->timeRange = atoi( varStr.c_str() );
  }
  else if(strcmp(cmd, "StartCycle") == 0 )
  {
    std::string varStr = std::string(args);
    size_t found = varStr.find_last_of(";");
    varStr = varStr.substr(found + 1);

    sim->timeStart = atoi( varStr.c_str() );
  }
  else if(strcmp(cmd, "StepCycle") == 0 )
  {
    std::string varStr = std::string(args);
    size_t found = varStr.find_last_of(";");
    varStr = varStr.substr(found + 1);

    sim->timeStep = atoi( varStr.c_str() );
  }
  else if(strcmp(cmd, "StopCycle") == 0 )
  {
    std::string varStr = std::string(args);
    size_t found = varStr.find_last_of(";");
    varStr = varStr.substr(found + 1);

    sim->timeStop = atoi( varStr.c_str() );
  }
  else
  {
    std::stringstream msg;
    msg << "Visit libsim - ignoring command " << cmd << "  args " << args;
    VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
    VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
  }
  
  if( sim->runMode == VISIT_SIMMODE_RUNNING &&
      sim->simMode == VISIT_SIMMODE_RUNNING )
  {
    if(sim->isProc0)
    {
      VisItUI_setValueS("SIMULATION_MODE", "Running", 1);

      std::stringstream msg;      
      msg << "Visit libsim - Continuing the simulation";
      VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
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

//---------------------------------------------------------------------
// MaxTimeStepCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_MaxTimeStepCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->maxTimestep = atoi(val);
}

//---------------------------------------------------------------------
// MaxTimeCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_MaxTimeCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->maxTime = atof(val);
}

//---------------------------------------------------------------------
// DeltaTCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_DeltaTCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  DataWarehouse          *dw = sim->simController->getSchedulerP()->getLastDW();

  double delt = atof(val);

  simStateP->adjustDelT( false );
  dw->override(delt_vartype(delt), simStateP->get_delt_label());
}

//---------------------------------------------------------------------
// DeltaTMinCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_DeltaTMinCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->delt_min = atof(val);
}

//---------------------------------------------------------------------
// DeltaTMaxCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_DeltaTMaxCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->delt_max = atof(val);
}

//---------------------------------------------------------------------
// DeltaTFactorCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_DeltaTFactorCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->delt_factor = atof(val);
}

//---------------------------------------------------------------------
// MaxWallTimeCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_MaxWallTimeCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->max_wall_time = atof(val);
}

//---------------------------------------------------------------------
// UPSVariableTableCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_UPSVariableTableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  int row, column;
  char str[128];

  sscanf (val, "%d | %d | %s", &row, &column, str);

  std::vector< SimulationState::interactiveVar > &vars =
    simStateP->d_interactiveVars;
      
  SimulationState::interactiveVar &var = vars[row];

  switch( var.type )
  {       
    case Uintah::TypeDescription::int_type:
    {
      int value;
      sscanf (val, "%d | %d | %d", &row, &column, &value);
      *(var.Ivalue) = value;
      break;
    }
    
    case Uintah::TypeDescription::double_type:
    {
      double value;
      sscanf (val, "%d | %d | %lf", &row, &column, &value);
      *(var.Dvalue) = value;
      break;
    }
    
    case Uintah::TypeDescription::Vector:
    {
      double x, y, z;
      sscanf (val, "%d | %d | %lf,%lf,%lf", &row, &column, &x, &y, &z);
      *(var.Vvalue) = Vector(x, y, z);
      break;
    }
    default:
      throw InternalError(" invalid data type", __FILE__, __LINE__); 
  }

  // Set the modified flag to true so the component knows the variable
  // was modified.
  var.modified = true;

  // Changing this variable may require recompiling the task graph.
  if( var.recompile )
    simStateP->setRecompileTaskGraph( true );
}

//---------------------------------------------------------------------
// OutputIntervalVariableTableCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_OutputIntervalVariableTableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  Output *output = sim->simController->getOutput();

  int row, column;
  double value;

  sscanf (val, "%d | %d | %lf", &row, &column, &value);

  // Output interval based on time.
  if( row == OutputIntervalRow )
  {
    if( output->getOutputInterval() > 0 )
      output->updateOutputInterval( value );
    // Output interval based on timestep.
    else
      output->updateOutputTimestepInterval( value );
  }
  
  // Checkpoint interval based on times.
  else if( row == CheckpointIntervalRow )
  {
    if( output->getCheckpointInterval() > 0 )
      output->updateCheckpointInterval( value );
    // Checkpoint interval based on timestep.
    else
      output->updateCheckpointTimestepInterval( value );
  }
}

//---------------------------------------------------------------------
// ImageCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_ImageGenerateCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageGenerate = val;
}

//---------------------------------------------------------------------
// ImageFilenameCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_ImageFilenameCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageFilename = std::string(val);
}

//---------------------------------------------------------------------
// ImageHeightCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_ImageHeightCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageHeight = atoi(val);
}

//---------------------------------------------------------------------
// ImageWidthCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_ImageWidthCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageWidth = atoi(val);
}

//---------------------------------------------------------------------
// ImageFormatCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_ImageFormatCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageFormat = val;
}

//---------------------------------------------------------------------
// StopAtTimestepCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_StopAtTimestepCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->stopAtTimestep = atoi(val);
}

  //---------------------------------------------------------------------
// StopAtLastTimestepCallback
//     Custom UI callback
//---------------------------------------------------------------------
void visit_StopAtLastTimestepCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->stopAtLastTimestep = val;
}

} // End namespace Uintah

