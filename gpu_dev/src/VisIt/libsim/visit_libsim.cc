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
#include "visit_libsim_database.h"

#include <sci_defs/mpi_defs.h>
#include <sci_defs/visit_defs.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Util/DebugStream.h>

#include "StandAlone/tools/uda2vis/uda2vis.h"

#include <dlfcn.h>

static SCIRun::DebugStream visitdbg( "VisItLibSim", true );

static std::string simFileName( "Uintah" );
static std::string simExecName;
static std::string simArgs;
static std::string simComment("Uintah Simulation");

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
#ifdef VISIT_STOP
  // The simulation will wait for VisIt to connect after first step.
  sim->runMode = VISIT_SIMMODE_STOPPED;
#else
    // Default is to run the simulation and VisIt can connect any time.
  sim->runMode = VISIT_SIMMODE_RUNNING;
#endif

  // Assume the simulation will be running (or about to run) when
  // initializing.
  sim->simMode = VISIT_SIMMODE_RUNNING;
    
  // TODO: look for the VisItSetupEnvironment2 function.
  // Has better scaling, but has not been release for fortran.
  VisItSetupEnvironment();

  if( Parallel::usingMPI() )
  {
    sim->isProc0 = isProc0_macro;

    int par_rank, par_size;

    // Initialize MPI
    MPI_Comm_rank (MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &par_size);
    
    // Tell libsim if the simulation is running in parallel.
    VisItSetParallel( par_size > 1 );
    VisItSetParallelRank( par_rank );

    // Install callback functions for global communication.
    VisItSetBroadcastIntFunction( visit_BroadcastIntCallback );
    VisItSetBroadcastStringFunction( visit_BroadcastStringCallback );
  }
  else
  {
    sim->isProc0 = 1;
  }

  // Have the rank 0 process create the sim file.
  if(sim->isProc0)
  {
    std::string exeCommand;

    if( simExecName.find( "/" ) != 0 )
    {
      char *path = NULL;
      
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
                                        NULL, "uintah.ui", NULL);
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
void visit_CheckState( visit_simulation_data *sim )
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
      // Tell VisIt to update its plots
      VisItUpdatePlots();
    }
  }

  do
  {
    /* If running do not block */
    int blocking = (sim->runMode == VISIT_SIMMODE_RUNNING) ? 0 : 1;

    if( sim->blocking != blocking )
    {
      sim->blocking = blocking;
        
      if( VisItIsConnected() )
      {
	if( blocking )
	{
	  // If blocking the run mode is not running so the
	  // simulation will not be running so change the state to
	  // allow asyncronious commands like saving a timestep or a
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
    }

    /* Get input from VisIt or timeout so the simulation can run. */
    int visitstate;

    if(sim->isProc0)
      visitstate = VisItDetectInput(blocking, -1);

    if( Parallel::usingMPI() )
      MPI_Bcast(&visitstate, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
	  msg << "Visit libsim - Continuing the simulation - no input";
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
        if( Parallel::usingMPI() )
          VisItSetSlaveProcessCallback(visit_SlaveProcessCallback);

        /* Register command callback */
        VisItSetCommandCallback(visit_ControlCommandCallback, (void*) sim);

	std::stringstream msg;
      
        if( Parallel::usingMPI() )
        {
          int par_rank;
          MPI_Comm_rank (MPI_COMM_WORLD, &par_rank);

          msg << "Visit libsim - Processor " << par_rank << " connected";
        }
        else
        {
          msg << "Visit libsim - Connected";
        }

	if(sim->isProc0)
	{
	  VisItUI_setValueS("SIMULATION_MESSAGE_CLEAR", "NoOp", 1);

	  // visitdbg << msg.str().c_str() << std::endl;
	  // visitdbg.flush();
	  VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);

	  VisItUI_setValueS("SIMULATION_MODE", "Connected", 1);
	}

        /* Register data access callbacks */
        VisItSetGetMetaData(visit_SimGetMetaData, (void*) sim);

        VisItSetGetMesh(visit_SimGetMesh, (void*) sim);

        VisItSetGetVariable(visit_SimGetVariable, (void*) sim);

        if( Parallel::usingMPI() )
          VisItSetGetDomainList(visit_SimGetDomainList, (void*) sim);
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
        /* Disconnect on an error or closed connection. */
        VisItDisconnect();

        /* Start running again if VisIt closes. */
        sim->runMode = VISIT_SIMMODE_RUNNING;

	if(sim->isProc0)
	{
	  VisItUI_setValueS("SIMULATION_MODE", "Unknown", 1);
	}
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
}

} // End namespace Uintah
