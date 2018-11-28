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
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

#include <fstream>

namespace Uintah {

extern Uintah::Dout visitdbg;

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
                               varType oldValue, varType newValue )
{
  // Depending on the GUI widget the reporting might be on a key
  // stroke key stroke basis or after a return is sent.
  if( sim->isProc0 )
  {
    std::stringstream msg;
    msg << "Visit libsim - At time step " << sim->cycle << " "
        << "the user modified the variable " << name << " "
        << "from " << oldValue << " " << "to " << newValue << ". ";
      
    VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
    VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
    
    // DOUT( visitdbg, msg.str().c_str() );
  }

  // Using a map - update the value so it can be recorded by Uintah.
  std::stringstream oldStr, newStr;

  // See if the value haas been recorded previously. 
  std::map< std::string, std::pair<std::string, std::string> >::iterator it =
    sim->modifiedVars.find( name);

  // If receorded previouosly get the original oldValue so to preserve
  // it.
  if (it != sim->modifiedVars.end())
    oldStr << it->second.first;
  // Otherwise use the current oldValue
  else
    oldStr << oldValue;
  
  newStr << newValue;

  // Store the old and new values.
  sim->modifiedVars[ name ] =
    std::pair<std::string, std::string>(oldStr.str(), newStr.str());

  sim->simController->getApplicationInterface()->haveModifiedVars( true );
}

//---------------------------------------------------------------------
// getNextString
//   This method is called to get the next string value and return
//   the remaining part.
//
//---------------------------------------------------------------------
std::string getNextString( std::string &cmd, const std::string delimiter )
{
  size_t delim = cmd.find_first_of( delimiter );

  std::string str = cmd;

  if( delim != std::string::npos)
  {
    str.erase(delim, std::string::npos);  
    cmd.erase(0, delim+delimiter.length());
  }
  
  return str;
}

//---------------------------------------------------------------------
// Method: QvisSimulationWindow::parseCompositeCMD
//
// Purpose:
//   This method is called to parse a composite cmd to get the
//   index and name.
//---------------------------------------------------------------------
void
parseCompositeCMD( const char *cmd,
                   unsigned int &index,
                   std::string &text )
{
  std::string strcmd(cmd);

  std::string str = getNextString( strcmd, " | " );
  index = atoi( str.c_str() );

  text = getNextString( strcmd, " | " );
}

//---------------------------------------------------------------------
// Method: QvisSimulationWindow::parseCompositeCMD
//
// Purpose:
//   This method is called to parse a composite cmd to get the
//   row, column, and name.
//---------------------------------------------------------------------
void
parseCompositeCMD( const char *cmd,
                   unsigned int &row,
                   unsigned int &column,
                   std::string &text )
{
  std::string strcmd(cmd);

  std::string str = getNextString( strcmd, " | " );
  row = atoi( str.c_str() );

  str = getNextString( strcmd, " | " );
  column = atoi( str.c_str() );

  text = getNextString( strcmd, " | " );
}

//---------------------------------------------------------------------
// Method: QvisSimulationWindow::parseCompositeCMD
//
// Purpose:
//   This method is called to parse a composite cmd to get the
//   row, column, x, and y values.
//---------------------------------------------------------------------
void
parseCompositeCMD( const char *cmd,
                   unsigned int &row,
                   unsigned int &column,
                   double &x, double &y )
{
  std::string strcmd(cmd);

  std::string str = getNextString( strcmd, " | " );
  row = atof( str.c_str() );

  str = getNextString( strcmd, " | " );
  column = atof( str.c_str() );

  str = getNextString( strcmd, " | " );
  x = atof( str.c_str() );

  str = getNextString( strcmd, " | " );
  y = atof( str.c_str() );
}

//---------------------------------------------------------------------
// Method: QvisSimulationWindow::parseCompositeCMD
//
// Purpose:
//   This method is called to parse a composite cmd to get the
//   row, column, x, and y values.
//---------------------------------------------------------------------
void
parseCompositeCMD( const char *cmd,
                   unsigned int &row,
                   unsigned int &column,
                   double &val)
{
  std::string strcmd(cmd);

  std::string str = getNextString( strcmd, " | " );
  row = atof( str.c_str() );

  str = getNextString( strcmd, " | " );
  column = atof( str.c_str() );

  str = getNextString( strcmd, " | " );
  val = atof( str.c_str() );
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

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

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
    // visit_SimGetMetaData(cbdata);
    // visit_SimGetDomainList(nullptr, cbdata);
  }
  else if(strncmp(cmd, "Output", 6) == 0)
  {
    // Do not call unless the simulation is stopped or finished as it
    // will interfer with the task graph.
    if(sim->simMode == VISIT_SIMMODE_STOPPED ||
       sim->simMode == VISIT_SIMMODE_FINISHED)
    {
      // Disable the button so to prevent outputing the same time step
      VisItUI_setValueS("SIMULATION_ENABLE_BUTTON", cmd, 0);

      // Do not allow out of order output files.
      if( strlen(cmd) == 6 )
        VisItUI_setValueS("SIMULATION_ENABLE_BUTTON", "Output Previous", 0);
        
      Output *output = sim->simController->getOutput();
      SchedulerP schedulerP = sim->simController->getSchedulerP();
      
      output->outputTimeStep( sim->gridP, schedulerP, strlen(cmd) > 6 );
    }
    else
      VisItUI_setValueS("SIMULATION_MESSAGE_BOX",
                        "Can not save a timestep unless the simulation has "
                        "run for at least one time step and is stopped.", 0);
  }
  else if(strncmp(cmd, "Checkpoint", 10) == 0)
  {
    // Do not call unless the simulation is stopped or finished as it
    // will interfer with the task graph.
    if(sim->simMode == VISIT_SIMMODE_STOPPED ||
       sim->simMode == VISIT_SIMMODE_FINISHED)
    {
      // Disable the button so to prevent checkpointing the same time step
      VisItUI_setValueS("SIMULATION_ENABLE_BUTTON", cmd, 0);

      // Do not allow out of order checkpoint files.
      if( strlen(cmd) == 10 )
        VisItUI_setValueS("SIMULATION_ENABLE_BUTTON", "Checkpoint Previous", 0);
        
      Output *output = sim->simController->getOutput();
      SchedulerP schedulerP = sim->simController->getSchedulerP();
      
      output->checkpointTimeStep( sim->gridP, schedulerP, strlen(cmd) > 10 );
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
  else if(strcmp(cmd, "Terminate") == 0 ||
	  strcmp(cmd, "Abort") == 0)
  {
    sim->runMode = VISIT_SIMMODE_RUNNING;
    sim->simMode = VISIT_SIMMODE_TERMINATED;

    if(sim->isProc0)
    {
      std::stringstream msg;      
      msg << "Visit libsim - Terminating the simulation";
      VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
      VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);

      DOUT( visitdbg, msg.str().c_str() );

      VisItUI_setValueS("STRIP_CHART_CLEAR_ALL", "NoOp", 1);
      VisItUI_setValueS("SIMULATION_MODE", "Not connected", 1);
    }

    if(strcmp(cmd, "Abort") == 0)
    {
      VisItDisconnect();

      exit( 0 );
    }
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
  else if(strcmp(cmd, "StripChartVar") == 0 )
  {
    std::string strcmd = std::string(args);
    size_t pos = strcmd.find_last_of(";");
    strcmd = strcmd.substr(pos + 1);

    std::string str = getNextString( strcmd, " | " );
    unsigned int chart = atoi( str.c_str() );

    str = getNextString( strcmd, " | " );
    unsigned int curve = atoi( str.c_str() );

    std::string var = getNextString( strcmd, " | " );

    sim->stripChartNames[chart][curve] = var;
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
void visit_TimeStepsMaxCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  int oldValue = appInterface->getTimeStepsMax();
  int newValue = atoi(val);
  
  if( newValue <= sim->cycle )
  {
    std::stringstream msg;
    msg << "Visit libsim - the value (" << newValue << ") for "
        << "the time step maximum is before the current time step. "
        << "Resetting the value.";
    VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);

    VisItUI_setValueI("MaxTimeSteps", appInterface->getTimeStepsMax(), 1);
  }
  else
  {  
    appInterface->setTimeStepsMax( newValue );
    
    visit_VarModifiedMessage( sim, "TimeStepsMax", oldValue, newValue);
  }
}


//---------------------------------------------------------------------
// MaxTimeCallback
//     Custom UI callback for a line edit box
//---------------------------------------------------------------------
void visit_SimTimeMaxCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  float oldValue = appInterface->getSimTimeMax();
  float newValue = atof(val);

  if( newValue <= appInterface->getSimTime() )
  {
    std::stringstream msg;
    msg << "Visit libsim - the value (" << newValue << ") for "
        << "the simulation maximum time is before the current time. "
        << "Resetting the value.";
    VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);

    VisItUI_setValueD("SimTimeMax", appInterface->getSimTimeMax(), 1);
  }
  else
  {  
    appInterface->setSimTimeMax( newValue );

    visit_VarModifiedMessage( sim, "SimTimeMax", oldValue, newValue);
  }
}


//---------------------------------------------------------------------
// EndOnMaxTimeCallback
//     Custom UI callback for a check box
//---------------------------------------------------------------------
void visit_SimTimeEndAtMaxCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  appInterface->setSimTimeEndAtMax( val );
}


//---------------------------------------------------------------------
// DeltaTVariableCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_DeltaTVariableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  DataWarehouse *dw = sim->simController->getSchedulerP()->getLastDW();

  unsigned int row, column;
  double oldValue, newValue;
  
  parseCompositeCMD(val, row, column, newValue);

  switch( row )
  {
  case 1:  // DeltaTNext
    {
      double minValue = appInterface->getDelTMin();
      double maxValue = appInterface->getDelTMax();
      oldValue = appInterface->getNextDelT();
      
      if( newValue < minValue || maxValue < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for DeltaTNext "
            << "is outside the range [" << minValue << ", " << maxValue << "]. "
            << "Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetDeltaTValues( sim );
        return;
      }

      appInterface->setNextDelT( newValue );
    }
    break;
  case 2:  // DeltaTFactor
    {
      double minValue = 1.0e-4;
      double maxValue = 1.0e+4;
      oldValue = appInterface->getDelTMultiplier();
      
      if( newValue < minValue || maxValue < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue
            << ") for the DeltaTMutlipler "
            << "is outside the range [" << minValue << ", " << maxValue << "]. "
            << "Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetDeltaTValues( sim );
        return;
      }

      appInterface->setDelTMultiplier( newValue );
      visit_VarModifiedMessage( sim, "DeltaTMultiplier", oldValue, newValue );
    }
    break;
  case 3:  // MaxDeltaIncrease
    {
      double minValue = 0;
      double maxValue = 1.e99;
      oldValue = appInterface->getDelTMaxIncrease();
      
      if( newValue < minValue || maxValue < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for DeltaTMaxIncrease "
            << "is outside the range [" << minValue << ", " << maxValue << "]. "
            << "Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetDeltaTValues( sim );
        return;
      }

      appInterface->setDelTMaxIncrease( newValue );
      visit_VarModifiedMessage( sim, "DeltaTMaxIncrease", oldValue, newValue );
    }
    break;
  case 4:   // MinDeltaT
    {
      double minValue = 0;
      double maxValue = appInterface->getDelTMax();
      oldValue = appInterface->getDelTMin();
      
      if( newValue < minValue || maxValue < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for DeltaTMin "
            << "is outside the range [" << minValue << ", " << maxValue << "]. "
            << "Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetDeltaTValues( sim );
        return;
      }

      appInterface->setDelTMin( newValue );
      visit_VarModifiedMessage( sim, "DeltaTMin", oldValue, newValue );
    }
    break;
  case 5:  // MaxDeltaT
    {
      double minValue = appInterface->getDelTMin();
      double maxValue = 1.0e9;
      oldValue = appInterface->getDelTMax();
      
      if( newValue < minValue || maxValue < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for DeltaTMax "
            << "is outside the range [" << minValue << ", " << maxValue << "]. "
            << "Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetDeltaTValues( sim );
        return;
      }

      appInterface->setDelTMax( newValue );
      visit_VarModifiedMessage( sim, "DeltaTMax", oldValue, newValue );
    }
    break;
  case 6:  // MaxInitialDelta
    {
      double minValue = 1.0e-99;
      double maxValue = DBL_MAX;
      oldValue = appInterface->getDelTInitialMax();
      
      if( newValue < minValue || maxValue < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for DeltaTInitialMax "
            << "is outside the range [" << minValue << ", " << maxValue << "]. "
            << "Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetDeltaTValues( sim );
        return;
      }

      appInterface->setDelTInitialMax( newValue );
      visit_VarModifiedMessage( sim, "DeltaTInitialMax", oldValue, newValue );
    }
    break;
  case 7:  // InitialDeltaRange
    {
      double minValue = 0;
      double maxValue = 1.0e99;
      oldValue = appInterface->getDelTInitialRange();
      
      if( newValue < minValue || maxValue < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for DeltaTInitialRange "
            << "is outside the range [" << minValue << ", " << maxValue << "]. "
            << "Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetDeltaTValues( sim );
        return;
      }

      appInterface->setDelTInitialRange( newValue );
      visit_VarModifiedMessage( sim, "DeltaTInitialRange", oldValue, newValue );
    }
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

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  unsigned int row, column;
  double oldValue, newValue;

  parseCompositeCMD( val, row, column, newValue);

  if( row == 5 )
  {
    oldValue = appInterface->getWallTimeMax();
    appInterface->setWallTimeMax( newValue );
    visit_VarModifiedMessage( sim, "WallTimeMax", oldValue, newValue );
  }
}

//---------------------------------------------------------------------
// UPSVariableCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_UPSVariableCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  unsigned int row, column;
  std::string text;

  parseCompositeCMD( val, row, column, text);

  std::vector< ApplicationInterface::interactiveVar > &vars =
    appInterface->getUPSVars();
      
  ApplicationInterface::interactiveVar &var = vars[row];

  switch( var.type )
  {       
    case Uintah::TypeDescription::bool_type:
    {
      bool *val = (bool*) var.value;
      bool oldValue = *val;
      int  newValue = atoi( text.c_str() );

      if( newValue != (bool) var.range[0] && newValue != (bool) var.range[1] )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetUPSVars( sim );
        return;
      }
      
      *val = (bool) newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, (bool) newValue );
      break;
    }
    
    case Uintah::TypeDescription::int_type:
    {
      int *val = (int*) var.value;
      int oldValue = *val;
      int newValue = atoi( text.c_str() );

      if( newValue < (int) var.range[0] || (int) var.range[1] < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetUPSVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
      break;
    }
    
    case Uintah::TypeDescription::double_type:
    {
      double *val = (double*) var.value;
      double oldValue = *val;
      double newValue = atof( text.c_str() );

      if( newValue < (double) var.range[0] || (double) var.range[1] < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << "is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetUPSVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
      break;
    }
    
    case Uintah::TypeDescription::Point:
    {
      double x, y, z;
      sscanf(text.c_str(), "%lf,%lf,%lf", &x, &y, &z);

      Point *val = (Point*) var.value;
      Point oldValue = *val;
      Point newValue = Point(x, y, z);

      if( newValue.x() < (double) var.range[0] ||
          (double) var.range[1] < newValue.x() ||
          newValue.y() < (double) var.range[0] ||
          (double) var.range[1] < newValue.y() ||
          newValue.z() < (double) var.range[0] ||
          (double) var.range[1] < newValue.z() )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetUPSVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
      break;
    }
    case Uintah::TypeDescription::Vector:
    {
      double x, y, z;
      sscanf(text.c_str(), "%lf,%lf,%lf", &x, &y, &z);

      Vector *val = (Vector*) var.value;
      Vector oldValue = *val;
      Vector newValue = Vector(x, y, z);

      if( newValue.length() < (double) var.range[0] ||
          (double) var.range[1] < newValue.length() )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetUPSVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
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
    sim->simController->setRecompileTaskGraph( true );
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

  parseCompositeCMD( val, row, column, value);

  // Output interval.
  if( row == OutputIntervalRow )
  {
    // Output interval based on time.
    if( output->getOutputInterval() > 0 )
      output->setOutputInterval( value );
    // Output interval based on timestep.
    else // if( output->getOutputTimeStepInterval() > 0 )
      output->setOutputTimeStepInterval( value );
  }
 
  // Checkpoint interval.
  else if( row == CheckpointIntervalRow )
  {
    // Checkpoint interval based on times.
    if( output->getCheckpointInterval() > 0 )
      output->setCheckpointInterval( value );
    // Checkpoint interval based on the wall time.
    else if( output->getCheckpointWallTimeInterval() > 0 )
      output->setCheckpointWallTimeInterval( value );
    // Checkpoint interval based on the time step.
    else // if( output->getCheckpointTimeStepInterval() > 0 )
      output->setCheckpointTimeStepInterval( value );
  }

  // Checkpoint cycle.
  else if( row == CheckpointCycleRow )
  {
    output->setCheckpointCycle( value );
  }
}


//---------------------------------------------------------------------
// ClampTimeToOutputCallback
//     Custom UI callback for a check box
//---------------------------------------------------------------------
void visit_ClampTimeToOutputCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  appInterface->setSimTimeClampToOutput( val );
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

  int newValue = atoi(val);

  if( newValue <= 0 )
  {
    std::stringstream msg;
    msg << "Visit libsim - the value (" << newValue << ") for "
        << "the image height must be greater than zero. Resetting value.";
    VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
    VisItUI_setValueI("ImageHeight",   sim->imageHeight, 1);
  }
  else
    sim->imageHeight = newValue;
}

//---------------------------------------------------------------------
// ImageWidthCallback
//     Custom UI callback for a line edit
//---------------------------------------------------------------------
void visit_ImageWidthCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  int newValue = atoi(val);

  if( newValue <= 0 )
  {
    std::stringstream msg;
    msg << "Visit libsim - the value (" << newValue << ") for "
        << "the image width must be greater than zero. Resetting value.";
    VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
    VisItUI_setValueI("ImageWidth",    sim->imageWidth,  1);
  }
  else
    sim->imageWidth = newValue;
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

  int newValue = atoi(val);
  
  if( newValue <= sim->cycle )
  {
    std::stringstream msg;
    msg << "Visit libsim - the value (" << newValue << ") for "
        << "stopping the simulation is before the current time step. "
        << "Setting the value to the next time step.";
    VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
    VisItUI_setValueI("StopAtTimeStep", sim->cycle+1, 1);
    sim->stopAtTimeStep = sim->cycle+1;
  }
  else
    sim->stopAtTimeStep = newValue;
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

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  unsigned int row, column;
  std::string text;

  parseCompositeCMD( val, row, column, text);

  std::vector< ApplicationInterface::interactiveVar > &vars =
    appInterface->getStateVars();
      
  ApplicationInterface::interactiveVar &var = vars[row];

  switch( var.type )
  {       
    case Uintah::TypeDescription::bool_type:
    {
      bool *val = (bool*) var.value;
      bool oldValue = *val;
      int  newValue = atoi( text.c_str() );

      if( newValue != (bool) var.range[0] && newValue != (bool) var.range[1] )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetStateVars( sim );
        return;
      }
      
      *val = (bool) newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, (bool) newValue );
      break;
    }
    
    case Uintah::TypeDescription::int_type:
    {
      int *val = (int*) var.value;
      int oldValue = *val;
      int newValue = atoi( text.c_str() );

      if( newValue < (int) var.range[0] || (int) var.range[1] < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetStateVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
      break;
    }
    
    case Uintah::TypeDescription::double_type:
    {
      double *val = (double*) var.value;
      double oldValue = *val;
      double newValue = atof( text.c_str() );

      if( newValue < (double) var.range[0] || (double) var.range[1] < newValue )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetStateVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
      break;
    }
    
    case Uintah::TypeDescription::Point:
    {
      double x, y, z;
      sscanf(text.c_str(), "%lf,%lf,%lf", &x, &y, &z);

      Point *val = (Point*) var.value;
      Point oldValue = *val;
      Point newValue = Point(x, y, z);

      if( newValue.x() < (double) var.range[0] ||
          (double) var.range[1] < newValue.x() ||
          newValue.y() < (double) var.range[0] ||
          (double) var.range[1] < newValue.y() ||
          newValue.z() < (double) var.range[0] ||
          (double) var.range[1] < newValue.z() )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetStateVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
      break;
    }
    case Uintah::TypeDescription::Vector:
    {
      double x, y, z;
      sscanf(text.c_str(), "%lf,%lf,%lf", &x, &y, &z);

      Vector *val = (Vector*) var.value;
      Vector oldValue = *val;
      Vector newValue = Vector(x, y, z);

      if( newValue.length() < (double) var.range[0] ||
          (double) var.range[1] < newValue.length() )
      {
        std::stringstream msg;
        msg << "Visit libsim - the value (" << newValue << ") for "
            << var.name << " is outside the range [" << var.range[0] << ", "
            << var.range[1] << "]. Resetting value.";
        VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
        visit_SetStateVars( sim );
        return;
      }

      *val = newValue;

      visit_VarModifiedMessage( sim, var.name, oldValue, newValue );
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
    sim->simController->setRecompileTaskGraph( true );
}

//---------------------------------------------------------------------
// DebugStreamCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_DebugStreamCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  unsigned int row, column;
  std::string text;

  parseCompositeCMD( val, row, column, text);

  // Find the debugStream
  DebugStream *debugStream = nullptr;
    
  unsigned int i = 0;
  for (auto iter = DebugStream::m_all_debug_streams.begin();
       iter != DebugStream::m_all_debug_streams.end();
       ++iter, ++i) {
    if( i == row ) {
      debugStream = (*iter).second;
      break;
    }
  }

  if( debugStream && column == 0 )
  {
    if( text != "FALSE" && text != "False" && text != "false" &&
        text != "TRUE"  && text != "True"  && text != "true" && 
        text != "0" && text != "1" && text != "-" && text != "+" )
    {
      std::stringstream msg;
      msg << "Visit libsim - the value (" << text << ") for "
          << debugStream->getName()
          << " is not 'true' or 'false'. Resetting value.";
      VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
      visit_SetDebugStreams( sim );
      return;
    }
    
    debugStream->setActive( text == "TRUE" || text == "True" ||
                            text == "true" || text == "1" || text == "+" );
    
    if( debugStream->m_outstream == nullptr )
    {
      debugStream->setFilename( "cout" );
      debugStream->m_outstream = &std::cout;
    }
  }
  else if( debugStream && column == 3 )
  {
    if( text.find("cerr") != std::string::npos )
    {
      debugStream->setFilename( "cerr" );
      debugStream->m_outstream = &std::cerr;
    }
    else if( text.find("cout") != std::string::npos )
    {
      debugStream->setFilename( "cout" );
      debugStream->m_outstream = &std::cout;
    }
    else
    {
      debugStream->setFilename( text );

      if( debugStream->m_outstream &&
          debugStream->m_outstream != &std::cerr &&
          debugStream->m_outstream != &std::cout )
        delete debugStream->m_outstream;

      debugStream->m_outstream = new std::ofstream(text);
    }
  }
}


//---------------------------------------------------------------------
// DoutCallback
//     Custom UI callback for a table
//---------------------------------------------------------------------
void visit_DoutCallback(char *val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  unsigned int row, column;
  std::string text;

  parseCompositeCMD( val, row, column, text);

  // Find the dout
  Dout *dout = nullptr;
    
  unsigned int i = 0;
  for (auto iter = Dout::m_all_douts.begin();
       iter != Dout::m_all_douts.end();
       ++iter, ++i) {
    if( i == row ) {
      dout = (*iter).second;
      break;
    }
  }
    
  if( dout && column == 0 )
  {
    if( text != "FALSE" && text != "False" && text != "false" &&
        text != "TRUE"  && text != "True"  && text != "true" && 
        text != "0" && text != "1" && text != "-" && text != "+" )

    {
      std::stringstream msg;
      msg << "Visit libsim - the value (" << text << ") for "
          << dout->name()
          << " is not 'true' or 'false'. Resetting value.";
      VisItUI_setValueS("SIMULATION_MESSAGE_BOX", msg.str().c_str(), 1);
      visit_SetDebugStreams( sim );
      return;
    }

    dout->setActive( text == "TRUE" || text == "True" ||
                     text == "true" || text == "1" || text == "+" );

    // if( dout->m_outstream == nullptr )
    // {
    //   dout->setFilename( "cout" );
    //   dout->m_outstream = &std::cout;
    // }
  }
  // else if( dout && column == 3 )
  // {
  //   std::string filename( value );

  //   if( filename.find("cerr") != std::string::npos )
  //   {
  //     dout->setFilename( "cerr" );
  //     dout->m_outstream = &std::cerr;
  //   }
  //   else if( filename.find("cout") != std::string::npos )
  //   {
  //     dout->setFilename( "cout" );
  //     dout->m_outstream = &std::cout;
  //   }
  //   else
  //   {
  //     dout->setFilename( filename );

  //     if( dout->m_outstream &&
  //      dout->m_outstream != &std::cerr &&
  //      dout->m_outstream != &std::cout )
  //    delete dout->m_outstream;

  //     dout->m_outstream = new std::ofstream(filename);
  //   }
  // }
}

//---------------------------------------------------------------------
// LoadExtraElementsCallback
//     Custom UI callback for a check box
//---------------------------------------------------------------------
void visit_LoadExtraElementsCallback(int val, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  sim->loadExtraElements = val;
}


} // End namespace Uintah
