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

#include "visit_libsim.h"
#include "visit_libsim_database.h"
#include "visit_libsim_callbacks.h"
#include "visit_libsim_customUI.h"

#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Components/DataArchiver/DataArchiver.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Util/DebugStream.h>

#include <fstream>

namespace Uintah {

static Uintah::DebugStream visitdbg( "VisItLibSim", true );

#define VISIT_COMMAND_PROCESS 0
#define VISIT_COMMAND_SUCCESS 1
#define VISIT_COMMAND_FAILURE 2

//---------------------------------------------------------------------
// VarModifiedMessage
//  This method reports to the user when a variable is modified.
//---------------------------------------------------------------------
template< class varType >
void visit_VarModifiedMessage( visit_simulation_data *sim,
                               std::string name,
                               varType value )
{
  // Depending on the GUI widget the reporting might be on a key
  // stroke key stroke basis or after a return is sent.
  if( sim->isProc0 )
  {
    std::stringstream msg;
    msg << "Visit libsim - At time step " << sim->cycle << " "
        << "the user modified the variable " << name << " "
        << "to be " << value << ". ";
      
    VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
    VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
    
    // visitdbg << msg.str().c_str() << std::endl;
    // visitdbg.flush();
  }

  // Using a map - update the value so it can be recorded by Uintah.
  std::stringstream tmpstr;
  tmpstr << value;
  
  sim->modifiedVars[ name ] = tmpstr.str();
}


//---------------------------------------------------------------------
// getNextString
//   This method is called to get the next string value and return
//   the remaining part.
//
//   NOTE: it is assumed that there is a space on both sides of the delimiter
//   EXAMPLE: string | string string | string
//---------------------------------------------------------------------
std::string getNextString( std::string &cmd, const std::string delimiter )
{
  size_t delim = cmd.find_first_of( delimiter );

  std::string str = cmd;

  if( delim != std::string::npos)
  {
    // str.erase(delim-1, std::string::npos);  
    // cmd.erase(0, delim+2);
    str.erase(delim, std::string::npos);  
    cmd.erase(0, delim+delimiter.length());
  }
  
  return str;
}

//---------------------------------------------------------------------
// getTableCMD
//   This method is called to parse the table cmd to get the
//   row, column, and double value.
//---------------------------------------------------------------------
void getTableCMD( char *cmd,
                  unsigned int &row, unsigned int &column, double &value )
{
  std::string strcmd(cmd);

  std::string str = getNextString( strcmd, " | " );
  row = atoi( str.c_str() );

  str = getNextString( strcmd, " | " );
  column = atoi( str.c_str() );

  str = getNextString( strcmd, " | " );
  value = atof( str.c_str() );
}


//---------------------------------------------------------------------
// getTableCMD
//   This method is called to parse the table cmd to get the
//   row, column, and name.
//---------------------------------------------------------------------
void getTableCMD( char *cmd,
                  unsigned int &row, unsigned int &column, char *name )
{
  std::string strcmd(cmd);

  std::string str = getNextString( strcmd, " | " );
  row = atoi( str.c_str() );

  str = getNextString( strcmd, " | " );
  column = atoi( str.c_str() );

  str = getNextString( strcmd, " | " );
  strcpy( name, str.c_str() );
}
  

//---------------------------------------------------------------------
// visit_BroadcastIntCallback
//     Callback for processing integers
//---------------------------------------------------------------------
int visit_BroadcastIntCallback(int *value, int sender)
{
  if( Parallel::usingMPI() )
    return Uintah::MPI::Bcast(value, 1, MPI_INT, sender, MPI_COMM_WORLD);
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
    return Uintah::MPI::Bcast(str, len, MPI_CHAR, sender, MPI_COMM_WORLD);
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
    Uintah::MPI::Bcast(command, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
  else if(strcmp(cmd, "Unused") == 0 && sim->simMode != VISIT_SIMMODE_FINISHED)
  {
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
                        "Can not save a timestep unless the simulation has "
			"run for at least one time step and is stopped.", 0);
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
                        "Can not save a checkpoint unless the simulation has "
			"run for at least one time step and is stopped.", 0);
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
  else if(strcmp(cmd, "ActivateCustomUI") == 0 )
  {
    visit_SimGetCustomUIData(cbdata);    
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
    // These messages are really only helpful when debugging. 
    // std::stringstream msg;
    // msg << "Visit libsim - ignoring command " << cmd << "  args " << args;
    // VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
    // VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
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
//     Custom UI callback for a line edit box
//---------------------------------------------------------------------
void visit_MaxTimeStepCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->maxTimestep = atoi(val);

  visit_VarModifiedMessage( sim, "MaxTimeStep", val);
}


//---------------------------------------------------------------------
// MaxTimeCallback
//     Custom UI callback for a line edit box
//---------------------------------------------------------------------
void visit_MaxTimeCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->simController->getSimulationTime()->maxTime = atof(val);

  visit_VarModifiedMessage( sim, "MaxTime", val);
}


//---------------------------------------------------------------------
// DeltaTVariableCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_DeltaTVariableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;
  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  DataWarehouse          *dw = sim->simController->getSchedulerP()->getLastDW();

  unsigned int row, column;
  double value;
  
  getTableCMD( val, row, column, value);

  switch( row )
  {
  case 1:
    dw->override(delt_vartype(value), simStateP->get_delt_label());
    visit_VarModifiedMessage( sim, "DeltaTNext", value );
    break;
  case 2:
    sim->simController->getSimulationTime()->delt_factor = value;
    visit_VarModifiedMessage( sim, "DeltaTFactor", value );
    break;
  case 3:
    sim->simController->getSimulationTime()->delt_min = value;
    visit_VarModifiedMessage( sim, "DeltaTMin", value );
    break;
  case 4:
    sim->simController->getSimulationTime()->delt_max = value;
    visit_VarModifiedMessage( sim, "DeltaTMax", value );
    break;
  }
}

//---------------------------------------------------------------------
// WallTimesVariableCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_WallTimesVariableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  unsigned int row, column;
  double value;

  getTableCMD( val, row, column, value);

  if( row == 5 )
  {
    sim->simController->getSimulationTime()->max_wall_time = value;
    visit_VarModifiedMessage( sim, "MaxWallTime", value );
  }
}

//---------------------------------------------------------------------
// UPSVariableCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_UPSVariableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  unsigned int row, column;
  char str[128];

  getTableCMD( val, row, column, str);

  std::vector< SimulationState::interactiveVar > &vars = simStateP->d_UPSVars;
      
  SimulationState::interactiveVar &var = vars[row];

  switch( var.type )
  {       
    case Uintah::TypeDescription::bool_type:
    {
      bool *val = (bool*) var.value;
      *val = atoi( str );
      break;
    }
    
    case Uintah::TypeDescription::int_type:
    {
      int *val = (int*) var.value;
      *val = atoi( str );
      break;
    }
    
    case Uintah::TypeDescription::double_type:
    {
      double *val = (double*) var.value;
      *val = atof( str );
      break;
    }
    
    case Uintah::TypeDescription::Vector:
    {
      double x, y, z;
      sscanf(str, "%lf,%lf,%lf", &x, &y, &z);

      Vector *val = (Vector*) var.value;
      *val = Vector(x, y, z);
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

  if( sim->isProc0 )
  {
    std::stringstream msg;
    msg << "Visit libsim - At time step " << sim->cycle << " "
        << "the user modified the variable " << var.name << " "
        << "to be " << str << ". ";

    if( var.recompile )
      msg << "The task graph will be recompiled.";
      
    VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
    VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
    
    visitdbg << msg.str().c_str() << std::endl;
    visitdbg.flush();
  }
}

//---------------------------------------------------------------------
// OutputIntervalVariableCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_OutputIntervalVariableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  Output *output = sim->simController->getOutput();

  unsigned int row, column;
  double value;

  getTableCMD( val, row, column, value);

  // Output interval.
  if( row == OutputIntervalRow )
  {
    // Output interval based on time.
    if( output->getOutputInterval() > 0 )
      output->updateOutputInterval( value );
    // Output interval based on timestep.
    else
      output->updateOutputTimestepInterval( value );
  }
  
  // Checkpoint interval.
  else if( row == CheckpointIntervalRow )
  {
    // Checkpoint interval based on times.
    if( output->getCheckpointInterval() > 0 )
      output->updateCheckpointInterval( value );
    // Checkpoint interval based on timestep.
    else
      output->updateCheckpointTimestepInterval( value );
  }
}

//---------------------------------------------------------------------
// ImageCallback
//     Custom UI callback for a check box
//---------------------------------------------------------------------
void visit_ImageGenerateCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageGenerate = val;
}

//---------------------------------------------------------------------
// ImageFilenameCallback
//     Custom UI callback for a line edit
//---------------------------------------------------------------------
void visit_ImageFilenameCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageFilename = std::string(val);
}

//---------------------------------------------------------------------
// ImageHeightCallback
//     Custom UI callback for a line edit
//---------------------------------------------------------------------
void visit_ImageHeightCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageHeight = atoi(val);
}

//---------------------------------------------------------------------
// ImageWidthCallback
//     Custom UI callback for a line edit
//---------------------------------------------------------------------
void visit_ImageWidthCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageWidth = atoi(val);
}

//---------------------------------------------------------------------
// ImageFormatCallback
//     Custom UI callback for a drop down menu
//---------------------------------------------------------------------
void visit_ImageFormatCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->imageFormat = val;
}

//---------------------------------------------------------------------
// StopAtTimeStepCallback
//     Custom UI callback for a line edit
//---------------------------------------------------------------------
void visit_StopAtTimeStepCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->stopAtTimeStep = atoi(val);
}

//---------------------------------------------------------------------
// StopAtLastTimeStepCallback
//     Custom UI callback for a check box
//---------------------------------------------------------------------
void visit_StopAtLastTimeStepCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->stopAtLastTimeStep = val;
}

//---------------------------------------------------------------------
// StateVariableCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_StateVariableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  unsigned int row, column;
  char str[128];

  getTableCMD( val, row, column, str);

  std::vector< SimulationState::interactiveVar > &vars =
    simStateP->d_stateVars;
      
  SimulationState::interactiveVar &var = vars[row];

  switch( var.type )
  {       
    case Uintah::TypeDescription::bool_type:
    {
      bool *val = (bool*) var.value;
      *val = atoi( str );
      break;
    }
    
    case Uintah::TypeDescription::int_type:
    {
      int *val = (int*) var.value;
      *val = atoi( str );
      break;
    }
    
    case Uintah::TypeDescription::double_type:
    {
      double *val = (double*) var.value;
      *val = atof( str );
      break;
    }
    
    case Uintah::TypeDescription::Vector:
    {
      double x, y, z;
      sscanf(str, "%lf,%lf,%lf", &x, &y, &z);

      Vector *val = (Vector*) var.value;
      *val = Vector(x, y, z);
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

  if( sim->isProc0 )
  {
    std::stringstream msg;
    msg << "Visit libsim - At time step " << sim->cycle << " "
        << "the user modified the variable " << var.name << " "
        << "to be " << str << ". ";

    if( var.recompile )
      msg << "The task graph will be recompiled.";
      
    VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
    VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
    
    visitdbg << msg.str().c_str() << std::endl;
    visitdbg.flush();
  }
}

//---------------------------------------------------------------------
// StripChartCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_StripChartCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  unsigned int chart, index;
  char name[128];

  // Note the row/column are flipped from the actual table.
  getTableCMD( val, index, chart, name);

  sim->stripChartNames[chart][index] = std::string(name);
}

//---------------------------------------------------------------------
// DebugStreamCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_DebugStreamCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  unsigned int row, column;
  char value[128];

  getTableCMD( val, row, column, value);

  if( column == 1 )
  {
    std::string active( value );
    
    simStateP->d_debugStreams[row]->setActive( active == "TRUE" ||
					       active == "True" ||
					       active == "true" );
    
    if( simStateP->d_debugStreams[row]->outstream == nullptr )
    {
      simStateP->d_debugStreams[row]->setFilename( "cout" );
      simStateP->d_debugStreams[row]->outstream = &std::cout;
    }
  }
  else if( column == 2 )
  {
    std::string filename( value );

    if( filename.find("cerr") != std::string::npos )
    {
      simStateP->d_debugStreams[row]->setFilename( "cerr" );
      simStateP->d_debugStreams[row]->outstream = &std::cerr;
    }
    else if( filename.find("cout") != std::string::npos )
    {
      simStateP->d_debugStreams[row]->setFilename( "cout" );
      simStateP->d_debugStreams[row]->outstream = &std::cout;
    }
    else
    {
      simStateP->d_debugStreams[row]->setFilename( filename );

      if( simStateP->d_debugStreams[row]->outstream &&
	  simStateP->d_debugStreams[row]->outstream != &std::cerr &&
	  simStateP->d_debugStreams[row]->outstream != &std::cout )
	delete simStateP->d_debugStreams[row]->outstream;

      simStateP->d_debugStreams[row]->outstream = new std::ofstream(filename);
    }
  }
}

} // End namespace Uintah
