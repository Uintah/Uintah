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
#include "VisItDataInterface_V2.h"

#include "visit_libsim.h"
#include "visit_libsim_callbacks.h"
#include "visit_libsim_customUI.h"
#include "visit_libsim_database.h"

#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Components/OnTheFlyAnalysis/MinMax.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/Material.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/UintahMPI.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/visit_defs.h>

#include <StandAlone/tools/uda2vis/uda2vis.h>

#include <dlfcn.h>

#define ALL_LEVELS 99

static Uintah::DebugStream visitdbg( "VisItLibSim", true );

static std::string simFileName( "Uintah" );
static std::string simExecName;
static std::string simArgs;
static std::string simComment("Uintah Simulation");
static std::string simUI("uintah.ui");

namespace Uintah {

//---------------------------------------------------------------------
// ProcessLibSimArguments
//     This routine handles command line arguments
//     -dir <VisIt directory> 
//     -options <VisIt Options> 
//     -trace <VisIt trace file>
//---------------------------------------------------------------------
void visit_LibSimArguments(int argc, char **argv)
{
  bool setVisItDir = false;

  simExecName = std::string( argv[0] );

  for (int i=1; i<argc; ++i)
  {
    if( strcmp( argv[i], "-visit" ) == 0 )
    {
      simFileName += std::string( "." ) + std::string( argv[++i] );
    }
    else if( strcmp( argv[i], "-visit_comment" ) == 0 )
    {
      simComment = std::string( argv[++i] );
    }
    else if( strcmp( argv[i], "-visit_dir" ) == 0 )
    {
      VisItSetDirectory(argv[++i]);

      setVisItDir = true;
    }
    else if( strcmp( argv[i], "-visit_options" ) == 0 )
    {
      VisItSetOptions(argv[++i]);
    }
    else if( strcmp( argv[i], "-visit_trace" ) == 0 )
    {
      VisItOpenTraceFile(argv[++i]);
    }
    else if( strcmp( argv[i], "-visit_ui" ) == 0 )
    {
      simUI = std::string( argv[++i] );
    }
    // Save off the Uintah args.
    else
      simArgs += std::string( argv[i] ) + std::string( " " );
  }

  // Set the VisIt path as defined in sci_defs/visit_defs.h. This path
  // is slurped up from the --with-visit configure argument.
  if( !setVisItDir )
  {
      VisItSetDirectory( VISIT_PATH );
  }
}


//---------------------------------------------------------------------
// InitLibSim
//     Initialize the VisIt Lib Sim 
//---------------------------------------------------------------------
void visit_InitLibSim( visit_simulation_data *sim )
{
  // The simulation will wait for VisIt to connect after first step.
  // sim->runMode = VISIT_SIMMODE_STOPPED;

  // Default is to run the simulation and VisIt can connect any time.
  // sim->runMode = VISIT_SIMMODE_RUNNING;

  // Assume the simulation will be running (or about to run) when
  // initializing.
  sim->simMode = VISIT_SIMMODE_RUNNING;

  sim->useExtraCells = true;
  sim->forceMeshReload = true;
  sim->mesh_for_patch_data = "";

  sim->timeRange = 0;
  sim->timeStart = 0;
  sim->timeStep  = 1;
  sim->timeStop  = 0;

  sim->imageGenerate = 0;
  sim->imageFilename = simFileName;
  sim->imageHeight   = 480;
  sim->imageWidth    = 640;
  sim->imageFormat   = 2;

  sim->stopAtTimeStep = 0;
  sim->stopAtLastTimeStep = 0;

  sim->stepInfo = nullptr;

  for( int i=0; i<5; ++i )
    for( int j=0; j<5; ++j )
      sim->stripChartNames[i][j] = std::string("");
  
  if( Parallel::usingMPI() )
  {
    int par_rank, par_size;

    // Initialize MPI
    Uintah::MPI::Comm_rank( MPI_COMM_WORLD, &par_rank );
    Uintah::MPI::Comm_size( MPI_COMM_WORLD, &par_size );
    
    // Tell libsim if the simulation is running in parallel.
    VisItSetParallel( par_size > 1 );
    VisItSetParallelRank( par_rank );

    // Install callback functions for global communication.
    VisItSetBroadcastIntFunction( visit_BroadcastIntCallback );
    VisItSetBroadcastStringFunction( visit_BroadcastStringCallback );

    sim->rank = par_rank;
    sim->isProc0 = isProc0_macro;
  }
  else
  {
    sim->rank = 0;
    sim->isProc0 = true;
  }

  // TODO: look for the VisItSetupEnvironment2 function.
  // Has better scaling, but has not been release for fortran.

  // NOTE: This call must be AFTER the parallel related calls.
  VisItSetupEnvironment();

  // Have the rank 0 process create the sim file.
  if(sim->isProc0)
  {
    std::string exeCommand;

    if( simExecName.find( "/" ) != 0 )
    {
      char *path = nullptr;
      
      Dl_info info;
      if (dladdr(__builtin_return_address(0), &info))
      {
        // The last slash removes the library
        const char *lastslash = strrchr(info.dli_fname,'/');
        
        if( lastslash )
        {
          // Remove the library and library directory.
          int pathLen = strlen(info.dli_fname) - strlen(lastslash) - 3;
          
          if( pathLen > 0 )
          {
            path = (char *) malloc( pathLen + 2 );
            
            strncpy( path, info.dli_fname, pathLen );
            path[pathLen] = '\0';

            exeCommand = std::string( path ) + simExecName;

            free( path );
          }
        }
      }
    }
    else
      exeCommand = simExecName;

    exeCommand += std::string( " " ) + simArgs;

    VisItInitializeSocketAndDumpSimFile(simFileName.c_str(),
                                        simComment.c_str(),
                                        exeCommand.c_str(),
                                        nullptr, simUI.c_str(), nullptr);
  }
}


//---------------------------------------------------------------------
// EndLibSim
//     End the VisIt Lib Sim - but let the user disconnet
//---------------------------------------------------------------------
void visit_EndLibSim( visit_simulation_data *sim )
{
  // Only go into finish mode if connected and the user has not force
  // the simulation to terminate early.
  if( VisItIsConnected() && sim->simMode != VISIT_SIMMODE_TERMINATED )
  {
    // The simulation is finished but the user may want to stay
    // conntected to analyze the last time step. So stop the run mode
    // but do not let the simulation complete until the user says they
    // are finished.
    sim->runMode = VISIT_SIMMODE_STOPPED;
    sim->simMode = VISIT_SIMMODE_FINISHED;

    if(sim->isProc0)
    {
      VisItUI_setValueS("SIMULATION_MODE", "Stopped", 1);

      std::stringstream msg;
      msg << "Visit libsim - "
          << "The simulation has finished, stopping at the last time step.";
      
      visitdbg << msg.str().c_str() << std::endl;
      visitdbg.flush();
      
      VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
    }

    // Now check for the user to have finished or issue a terminate.
    do
    {
      visit_CheckState(sim);
    }
    while( sim->runMode != VISIT_SIMMODE_FINISHED &&
           sim->simMode != VISIT_SIMMODE_TERMINATED );

    VisItUI_setValueS("SIMULATION_MODE", "Unknown", 1);
  }
}

//---------------------------------------------------------------------
// CheckState
//     Check the state from the viewer on all processors.
//---------------------------------------------------------------------
bool visit_CheckState( visit_simulation_data *sim )
{
  int err = 0;

  // If the simulation is running update the time step and plots.
  if( sim->simMode == VISIT_SIMMODE_RUNNING )
  {
    // Check if we are connected to VisIt
    if( VisItIsConnected() )
    {
      if(sim->isProc0)
      {
        // VisItUI_setValueS("SIMULATION_MESSAGE", sim->message.c_str(), 1);

        std::stringstream msg;
        msg << "Visit libsim - Completed simulation "
            << "timestep " << sim->cycle << ",  "
            << "Time = "   << sim->time;
        
//      visitdbg << msg.str().c_str() << std::endl;
//      visitdbg.flush();
        VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
      }

      // Tell VisIt that the timestep changed
      VisItTimeStepChanged();

      if( ( sim->timeRange == 0 ) ||
          ( sim->timeRange == 1 &&
            sim->timeStart <= sim->cycle && sim->cycle <= sim->timeStop &&
            (sim->cycle-sim->timeStart) % sim->timeStep == 0 ) )
      {
        // Tell VisIt to update its plots
        VisItUpdatePlots();
      
        // Tell VisIt to save the window.
        if( sim->imageGenerate )
        {
          std::stringstream fname;
          fname << sim->imageFilename << "_" << sim->cycle;
          VisItSaveWindow( fname.str().c_str(),
                           sim->imageWidth, sim->imageHeight,
                           sim->imageFormat );
        }
      }

      // Check to see if the user wants to stop.
      if( sim->cycle == sim->stopAtTimeStep )
      {
        sim->runMode = VISIT_SIMMODE_STOPPED;
      }
    }
  }

  do
  {
    /* If running do not block */
    int blocking = (sim->runMode == VISIT_SIMMODE_RUNNING) ? 0 : 1;

    // State change so update
    if( sim->blocking != blocking )
    {
      sim->blocking = blocking;

      if( blocking )
      {
	// If blocking the run mode is not running so the simulation
	// will not be running so change the state to allow
	// asyncronious commands like saving a timestep or a
	// checkpoint to happen.
	if( sim->simMode != VISIT_SIMMODE_FINISHED )
        {
	  sim->simMode = VISIT_SIMMODE_STOPPED;
	  
	  if(sim->isProc0)
          {
	    VisItUI_setValueS("SIMULATION_MODE", "Stopped", 1);
              
	    std::stringstream msg;
	    msg << "Visit libsim - Stopped the simulation at "
		<< "timestep " << sim->cycle << ",  "
		<< "Time = " << sim->time;
            
	    visitdbg << msg.str().c_str() << std::endl;
	    visitdbg.flush();
	    VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
          }
        }
      }
    }

    /* Get input from VisIt or timeout so the simulation can run. */
    int visitstate;

    if(sim->isProc0)
      visitstate = VisItDetectInput(blocking, -1);

    if( Parallel::usingMPI() )
      Uintah::MPI::Bcast(&visitstate, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Do different things depending on the output from VisItDetectInput. */
    if(visitstate <= -1 || 5 <= visitstate)
    {
      std::stringstream msg;      
      msg << "Visit libsim - CheckState cannot recover from error ("
          << visitstate << ") !!";
          
      visitdbg << msg.str().c_str() << std::endl;
      visitdbg.flush();
      VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);

      err = 1;
    }
    else if(visitstate == 0)
    {
      if( sim->simMode != VISIT_SIMMODE_FINISHED &&
          sim->simMode != VISIT_SIMMODE_TERMINATED )
      {
        VisItUI_setValueS("SIMULATION_MODE", "Running", 1);
        sim->simMode = VISIT_SIMMODE_RUNNING;
        
        if(sim->isProc0)
        {
          std::stringstream msg;          
          msg << "Visit libsim - No input, continuing the simulation.";

          // visitdbg << msg.str().c_str() << std::endl;
          // visitdbg.flush();
          VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
        }
      }

      /* There was no input from VisIt, return control to sim. */
      break;
    }
    else if(visitstate == 1)
    {
      /* VisIt is trying to connect to sim. */
      if(VisItAttemptToCompleteConnection())
      {
        visit_Initialize( sim );
      }
      else
      {
        std::stringstream msg;
        msg << "Visit libsim - Can not connect.";

        // visitdbg << msg.str().c_str() << std::endl;
        // visitdbg.flush();
        VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);
      }
    }
    else if(visitstate == 2)
    {
      if( !visit_ProcessVisItCommand(sim) )
      {
	if(sim->isProc0)
	{
	  VisItUI_setValueS("SIMULATION_MESSAGE_CLEAR", "NoOp", 1);
	  VisItUI_setValueS("STRIP_CHART_CLEAR_ALL",    "NoOp", 1);

          VisItUI_setValueS("SIMULATION_MODE", "Unknown", 1);
	}

        /* Start running again if VisIt closes. */
        sim->runMode = VISIT_SIMMODE_RUNNING;

        /* Disconnect on an error or closed connection. */
        VisItDisconnect();
      }

      /* If in step mode return control back to the simulation. */
      if( sim->runMode == VISIT_SIMMODE_STEP )
      {
        sim->simMode = VISIT_SIMMODE_RUNNING;

        if(sim->isProc0)
        {
          VisItUI_setValueS("SIMULATION_MODE", "Running", 1);

          std::stringstream msg;          
          msg << "Visit libsim - Continuing the simulation for one time step";
	  
          // visitdbg << msg.str().c_str() << std::endl;
          // visitdbg.flush();
          VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
        }

        // Swap over to stop mode because if an error occurs VisIt
        // will otherwise advance to the next time step.
        sim->runMode = VISIT_SIMMODE_STOPPED;

        sim->blocking = 0;
        break;
      }

      /* If finished return control back to the simulation. */
      else if( sim->runMode == VISIT_SIMMODE_FINISHED )
      {
        if(sim->isProc0)
        {
          VisItUI_setValueS("SIMULATION_MODE", "Unknown", 1);

          std::stringstream msg;          
          msg << "Visit libsim - Finished the simulation ";

          // visitdbg << msg.str().c_str() << std::endl;
          // visitdbg.flush();
          VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
          VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
        }

        sim->blocking = 0;
        break;
      }
    }
  } while(err == 0);

  return (sim->simMode == VISIT_SIMMODE_TERMINATED);  
}


//---------------------------------------------------------------------
// UpdateSimData
//     Update the simulation data on all processors
//---------------------------------------------------------------------
void visit_UpdateSimData( visit_simulation_data *sim, 
                          GridP currentGrid,
                          double time, double delt, double delt_next,
			  bool first, bool last )
{
  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  // Update all of the simulation grid and time dependent
  // variables.
  sim->gridP         = currentGrid;
  sim->time          = time;
  sim->delt          = delt;
  sim->delt_next     = delt_next;
  
  sim->first         = first;

  sim->cycle         = simStateP->getCurrentTopLevelTimeStep();

  // Check to see if at the last iteration. If so stop so the
  // user can have once last chance see the data.
  if( sim->stopAtLastTimeStep && last )
  {
    sim->runMode = VISIT_SIMMODE_STOPPED;

    if(sim->isProc0)
    {
      std::stringstream msg;
      msg << "Visit libsim - "
          << "The simulation has finished, stopping at the last time step.";
      
      visitdbg << msg.str().c_str() << std::endl;
      visitdbg.flush();
      
      VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
    }
  }
}

//---------------------------------------------------------------------
// Initialize()
//     Initialize everything
//---------------------------------------------------------------------
void visit_Initialize( visit_simulation_data *sim )
{    
  std::stringstream msg;
      
  if( Parallel::usingMPI() )
  {
    msg << "Visit libsim - Processor " << sim->rank << " connected";
  }
  else
  {
    msg << "Visit libsim - Connected";
  }

  if(sim->isProc0)
  {
    VisItUI_setValueS("SIMULATION_MESSAGE_CLEAR", "NoOp", 1);
    VisItUI_setValueS("STRIP_CHART_CLEAR_ALL",    "NoOp", 1);
    
    // visitdbg << msg.str().c_str() << std::endl;
    // visitdbg.flush();
    VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);    
    VisItUI_setValueS("SIMULATION_MODE", "Connected", 1);
  }
  

  if( Parallel::usingMPI() )
    VisItSetSlaveProcessCallback(visit_SlaveProcessCallback);

  /* Register command callback */
  VisItSetCommandCallback(visit_ControlCommandCallback, (void*) sim);

  // These are one time initializations.
  VisItUI_setValueI("SIMULATION_TIME_LIMITS_ENABLED", sim->timeRange, 1);
  VisItUI_setValueI("SIMULATION_TIME_START_CYCLE",    sim->timeStart, 1);
  VisItUI_setValueI("SIMULATION_TIME_STEP_CYCLE",     sim->timeStep,  1);
  VisItUI_setValueI("SIMULATION_TIME_STOP_CYCLE",     sim->timeStop,  1);

  /* Register data access callbacks */
  VisItSetGetMetaData(visit_SimGetMetaData, (void*) sim);

  VisItSetGetMesh(visit_SimGetMesh, (void*) sim);

  VisItSetGetVariable(visit_SimGetVariable, (void*) sim);

  if( Parallel::usingMPI() )
    VisItSetGetDomainList(visit_SimGetDomainList, (void*) sim);

  VisItUI_textChanged("MaxTimeStep", visit_MaxTimeStepCallback, (void*) sim);
  VisItUI_textChanged("MaxTime",     visit_MaxTimeCallback,     (void*) sim);
  VisItUI_valueChanged("EndOnMaxTime",
                       visit_EndOnMaxTimeCallback, (void*) sim);

  VisItUI_cellChanged("DeltaTVariableTable",
                      visit_DeltaTVariableCallback,          (void*) sim);
  VisItUI_cellChanged("WallTimesVariableTable",
                      visit_WallTimesVariableCallback,       (void*) sim);
  VisItUI_cellChanged("UPSVariableTable",
                      visit_UPSVariableCallback,             (void*) sim);
  VisItUI_cellChanged("OutputIntervalVariableTable",
                      visit_OutputIntervalVariableCallback,  (void*) sim);
  VisItUI_valueChanged("ClampTimestepsToOutput",
                       visit_ClampTimestepsToOutputCallback, (void*) sim);
        
  VisItUI_valueChanged("ImageGroupBox",
                       visit_ImageGenerateCallback, (void*) sim);
  VisItUI_textChanged ("ImageFilename",
                       visit_ImageFilenameCallback, (void*) sim);
  VisItUI_textChanged("ImageHeight",    visit_ImageHeightCallback, (void*) sim);
  VisItUI_textChanged("ImageWidth",     visit_ImageWidthCallback,  (void*) sim);
  VisItUI_valueChanged("ImageFormat",   visit_ImageFormatCallback, (void*) sim);

  VisItUI_textChanged("StopAtTimeStep",
                      visit_StopAtTimeStepCallback,      (void*) sim);
  VisItUI_valueChanged("StopAtLastTimeStep",
                       visit_StopAtLastTimeStepCallback, (void*) sim);

  VisItUI_cellChanged("StateVariableTable",
                      visit_StateVariableCallback, (void*) sim);

  VisItUI_cellChanged("DebugStreamTable",
                      visit_DebugStreamCallback,  (void*) sim);
  VisItUI_cellChanged("DoutTable",
                      visit_DoutCallback,  (void*) sim);

  VisItUI_valueChanged("LoadExtraCells",
                       visit_LoadExtraCellsCallback, (void*) sim);        
}
  
} // End namespace Uintah
