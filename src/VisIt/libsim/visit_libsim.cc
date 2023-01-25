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
#include <CCA/Ports/ApplicationInterface.h>

#include <Core/Grid/Material.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/UintahMPI.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Environment.h>
#include <Core/Util/SysUtils.h>

#include <sci_defs/visit_defs.h>

#include <fstream>
#include <dlfcn.h>

#define ALL_LEVELS 99

namespace Uintah {

static std::string simFileName( "Uintah" );
static std::string simExecName;
static std::string simArgs;
static std::string simComment("Uintah Simulation");
static std::string simUI("uintah.ui");
static bool        simConsole{false};

Dout visitdbg("VisItLibSim", "VisIt", "Interface to VisIt's libsim", true);

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
      // VisItGetVersion();

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
    else if( strcmp( argv[i], "-visit_console" ) == 0 )
    {
      simConsole = true;
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
  // sim->runMode is set in AMRSimulationController.cc

  // The simulation will wait for VisIt to connect after first step.
  // sim->runMode = VISIT_SIMMODE_STOPPED;

  // Default is to run the simulation and VisIt can connect any time.
  // sim->runMode = VISIT_SIMMODE_RUNNING;

  // Assume the simulation will be running (or about to run) when
  // initializing.
  sim->simMode = VISIT_SIMMODE_RUNNING;

  sim->loadExtraGeometry = NO_EXTRA_GEOMETRY;
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
    // Tell libsim if the simulation is running in parallel.
    VisItSetParallel( sim->myworld->nRanks() > 1 );
    VisItSetParallelRank( sim->myworld->myRank() );

    // Install callback functions for global communication.
    VisItSetBroadcastIntFunction( visit_BroadcastIntCallback );
    VisItSetBroadcastStringFunction( visit_BroadcastStringCallback );

    sim->isProc0 = isProc0_macro;
  }
  else
  {
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

  // Add in the machine layout details if present for this machine.
  sim->hostName.clear();
  sim->hostNode.clear();
  sim->switchNodeList.clear();

  sim->nodeStart.clear();
  sim->nodeStop.clear();
  sim->nodeCores.clear();

  sim->maxCores = 0;
  sim->maxNodes = 0;
  sim->xNode = 0;
  sim->yNode = 0;

  sim->switchIndex = -1;
  sim->nodeIndex = -1;

  // Get the host name without any digits and the host node without
  // any characters. This assumes the machine name contains no numbers
  // and node numbers contain no letters.
  std::string hostName = sim->myworld->myNodeName();
  std::string hostNode = sim->myworld->myNodeName();

  hostName.erase(std::remove_if(hostName.begin(), hostName.end(), (int(*)(int))std::isdigit), hostName.end());
  hostNode.erase(std::remove_if(hostNode.begin(), hostNode.end(), (int(*)(int))std::isalpha), hostNode.end());

  sim->hostName = hostName;
  sim->hostNode = hostNode;

  // A machine layout file could exist for this machine.
  if( sim->hostName.size() && sim->hostNode.size() )
  {
    // int nRanks = sim->myworld->nRanks();

    // // Get the processor's number of physical cores
    // int my_node_phyical_cores = sysGetNumSockets() * sysGetNumCoresPerSockets();

    // // Gather all processor's number of physical core sizes
    // std::vector< unsigned int > all_node_phyical_cores;
    // all_node_phyical_cores.resize(nRanks);

    // MPI::Allgather( &my_node_phyical_cores,        1,      MPI_INT,
    //              all_node_phyical_cores.data(), nRanks, MPI_INT,
    //              sim->myworld->getComm());

    // // Get the maximum number of physical core size across all
    // // processors.
    // int max_node_phyical_cores;
    // MPI::Allreduce( &my_node_phyical_cores, &max_node_phyical_cores,
    //              1, MPI_INT, MPI_MAX, sim->myworld->getComm());

    // The machine layout files are in the in situ source dir
    std::string path = std::string( sci_getenv("SCIRUN_OBJDIR") ) +
      std::string("/../src/VisIt/libsim/");

    std::string filename = path + sim->hostName + "_layout.txt";

    std::ifstream infile(filename);

    if( infile.is_open() )
    {
      // Read the text file line by line.
      std::string line;
      while( std::getline(infile, line) )
      {
        // Skip empty lines
        if( line.empty() )
        {
        }
        // This is part of the node table.
        else if( line.find("Nodes") == 0 )
        {
          // Get the node details (number of cores). The
          // number of digits is the number of digits that would be
          // present for a node number (machine001). Versus a head
          // node(machine1).
          std::string tmpNode, tmpDigits, tmpTo, tmpCores;
          unsigned int nDigits, start, stop, nCores;

          std::istringstream iss(line);

          if (!(iss >> tmpNode >> start >> tmpTo >> stop >> tmpDigits >> nDigits >> tmpCores >> nCores))
            break; // error

          // If on a head node skip the rest and bail out. Assume that
          // a head node index would never have more than one digit?
          if( sim->hostNode.size() != nDigits )
          {
            infile.close();

            sim->hostName = sim->myworld->myNodeName();
            sim->hostNode = "";
            break;
          }

          sim->nodeStart.push_back( start );
          sim->nodeStop.push_back( stop );
          sim->nodeCores.push_back( nCores );

          // Get the maximum number of cores.
          if( sim->maxCores < nCores )
            sim->maxCores = nCores;
        }
        // Skip these lines that are part of the call to ibnetdiscover
        else if( line.find("--") == 0 ||
                 line.find("devid") == 0 ||
                 line.find("sysimgguid") == 0 ||
                 line.find("switchguid") == 0 ||
                 line.find("switchguid") == 0 )
        {
        }
        else if(line.find("Switch") == 0 )
        {
          // Found a new switch so start a new node group.
          std::vector< unsigned int > nodes;

          sim->switchNodeList.push_back( nodes );
        }
        // A switch connection
        else if( line.find("[") == 0 )
        {
          // Find the hostname which will be quoted.
          size_t found = line.find( "\"" + sim->hostName );

          if( found != std::string::npos )
          {
            // Remove the hostname leaving only the node number.
            std::string nodeStr = line.substr(found + sim->hostName.size()+1);
            found = nodeStr.find(" ");
            nodeStr = nodeStr.substr(0, found);

            // Nodes with n digits are compute nodes.
            // Compute node node001 vs head node node1
            if( nodeStr.size() == sim->hostNode.size() )
            {
              std::istringstream iss(nodeStr);
              unsigned int node;

              iss >> node;

              // Add this node to the list.
              sim->switchNodeList.back().push_back( node );

              // Get the maximum number of nodes on a switch.
              if( sim->maxNodes < sim->switchNodeList.back().size() )
                sim->maxNodes = sim->switchNodeList.back().size();

              // Get the switch and node index for this processor.
              if( nodeStr == sim->hostNode )
              {
                sim->switchIndex = sim->switchNodeList.size()-1;
                sim->nodeIndex = sim->switchNodeList.back().size()-1;
              }
            }
          }
        }
        else
        {
          std::stringstream msg;
          msg << "Visit libsim - "
              << "Parse error \"" << line << "\" "
              << "in the current network file: " << filename;

          VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
        }
      }

      if( sim->switchNodeList.size() &&
          ((int) sim->switchIndex == -1 && (int) sim->nodeIndex == -1) )
      {
        sim->switchNodeList.clear();
        sim->nodeStart.clear();
        sim->nodeStop.clear();
        sim->nodeCores.clear();

        std::stringstream msg;
        msg << "Visit libsim - "
            << "Can not find node " << sim->myworld->myNodeName() << " "
            << "in the current network file: " << filename;

        VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);

        DOUT( true, msg.str() );
      }

      infile.close();
    }
  }

  // If no information is availble for this machine create a generic
  // view assuming running on a single node or the number of MPI ranks.
  if( sim->switchNodeList.size() == 0 )
  {
    sim->switchIndex = 0;
    sim->nodeIndex = 0;

    sim->maxNodes = 1;
    sim->maxCores = std::max( (unsigned int) sim->myworld->nRanks(),
                              sysGetNumSockets() * sysGetNumCoresPerSockets() );

    sim->nodeStart.push_back( 0 );
    sim->nodeStop.push_back( 1 );
    sim->nodeCores.push_back( sim->maxCores );

    // Found a new switch so start a new node group.
    std::vector< unsigned int > nodes;
    sim->switchNodeList.push_back( nodes );
    sim->switchNodeList.back().push_back( sim->nodeIndex );
  }

  // Get the greatest common demoninator so to have multiple
  // columns for the cores.
  unsigned int gcd = 2;

  for( unsigned int i=2; i<sqrt(sim->maxCores); ++i )
  {
    unsigned int cc = 0;

    for( unsigned int j=0; j<sim->nodeCores.size(); ++j )
    {
      if( sim->nodeCores[j] % i == 0 )
        ++cc;
    }

    if( cc == sim->nodeCores.size() )
      gcd = i;
  }

  // Size of a node based on the number of cores and GCD.
  sim->xNode = gcd;
  sim->yNode = sim->maxCores / gcd;
}


//---------------------------------------------------------------------
// EndLibSim
//     End the VisIt Lib Sim - but let the user disconnet
//---------------------------------------------------------------------
void visit_EndLibSim( visit_simulation_data *sim )
{
  // Only go into finish mode if connected and the user has not force
  // the simulation to terminate early.
  if( VisItIsConnected() )
  {
    if( sim->simMode == VISIT_SIMMODE_TERMINATED )
      VisItDisconnect();
    else // if( sim->simMode != VISIT_SIMMODE_TERMINATED )
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

        DOUT( visitdbg, msg.str().c_str() );

        VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
      }

      // Now check for the user to have finished or issue a terminate.
      do
      {
        visit_CheckState(sim);
      }
      while( sim->runMode != VISIT_SIMMODE_FINISHED &&
             sim->simMode != VISIT_SIMMODE_TERMINATED );

      VisItUI_setValueS("SIMULATION_MODE", "Not connected", 1);
    }
  }
}

//---------------------------------------------------------------------
// CheckState
//     Check the state from the viewer on all processors.
//---------------------------------------------------------------------
bool visit_CheckState( visit_simulation_data *sim )
{
  bool err = false;

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
        msg << "Visit libsim - Completed the simulation at "
            << "time step " << sim->cycle << ",  "
            << "time = "   << sim->time;

        // DOUT( visitdbg, msg.str().c_str() );

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
    // If running do not block
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
        if( sim->simMode != VISIT_SIMMODE_FINISHED &&
            sim->simMode != VISIT_SIMMODE_TERMINATED )
        {
          sim->simMode = VISIT_SIMMODE_STOPPED;

          if(sim->isProc0)
          {
            VisItUI_setValueS("SIMULATION_MODE", "Stopped", 1);

            std::stringstream msg;
            msg << "Visit libsim - Stopped the simulation at "
                << "time step " << sim->cycle << ",  "
                << "time = " << sim->time;

            DOUT( visitdbg, msg.str().c_str() );

            VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
          }
        }
      }
    }

    // Get input from VisIt or timeout so the simulation can run.
    int visitstate;

    if(sim->isProc0) // Pass -1 for no console input.
    {
      visitstate = VisItDetectInput(blocking,
                                    simConsole ? fileno(stdin) : -1);
    }

    if( Parallel::usingMPI() )
      Uintah::MPI::Bcast(&visitstate, 1, MPI_INT, 0, sim->myworld->getComm());

    // Do different things depending on the output from VisItDetectInput.

    // No input - timed out.
    if(visitstate == 0)
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

          // DOUT( visitdbg, msg.str().c_str() );

          VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
        }
      }

      // There was no input from VisIt, return control to sim.
      break;
    }
    // Connecting to the simulation, listen socket input.
    else if(visitstate == 1)
    {
      // VisIt is trying to connect to sim.
      if(VisItAttemptToCompleteConnection())
      {
        visit_Initialize( sim );
      }
      else
      {
        // Get the error message
        char *error = VisItGetLastError();

        std::stringstream msg;
        msg << "Visit libsim - Can not connect. " << error;

        DOUT( visitdbg, msg.str().c_str() );

        VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);

        free(error);
      }
    }
    // Process a command from the engine socket input.
    else if(visitstate == 2)
    {
      if( !visit_ProcessVisItCommand(sim) )
      {
        if(sim->isProc0)
        {
          VisItUI_setValueS("SIMULATION_MESSAGE_CLEAR", "NoOp", 1);
          VisItUI_setValueS("STRIP_CHART_CLEAR_ALL",    "NoOp", 1);

          VisItUI_setValueS("SIMULATION_MODE", "Not connected", 1);
        }

        // Start running again if VisIt closes.
        sim->runMode = VISIT_SIMMODE_RUNNING;

        // Disconnect on an error or closed connection.
        VisItDisconnect();
      }
    }
    // Process a command from the console socket
    else if(visitstate == 3)
    {
      // Read the command from the condole
      char cconsoleInput[1000];

      if(sim->isProc0)
      {
        if(VisItReadConsole(1000, cconsoleInput) == VISIT_ERROR)
        {
          std::stringstream msg;
          msg << "Visit libsim - Cannot read console input.";

          DOUT( visitdbg, msg.str().c_str() );

          VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
          VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
        }
      }

      if( Parallel::usingMPI() )
      {
        // Broadcast the console input to all processors.
        Uintah::MPI::Bcast(cconsoleInput, 1000, MPI_CHAR, 0, sim->myworld->getComm());
      }

      // Split the console input into the command and optional args.
      std::string cmd = std::string(cconsoleInput);
      std::string args;

      size_t found = cmd.find_first_of(" ");

      // Check to see if the command has optional arguments.
      if( found != std::string::npos )
      {
        args = cmd.substr(found + 1);
        cmd = cmd.substr(0, found);
      }

      // Process the command.
      visit_ControlCommandCallback(cmd.c_str(), args.c_str(), sim, true);
    }
    // Error
    else
    {
      const char *errorMsg[] =
        {"",
         "-1 Interrupted by EINTR in select."
         "-2 Unknown error in select.",
         "-3 Logic error (a socket was selected but not one we set).",
         "-4 Logic error (no descriptors but blocking).",
         "-5 Logic error (fell through all cases)." };

      std::stringstream msg;
      msg << "Visit libsim - CheckState cannot recover from error : ";

      if( -5 <= visitstate && visitstate <= -1 )
        msg << std::endl << errorMsg[abs(visitstate)];
      else
        msg << visitstate << " Unknown error.";

      DOUT( visitdbg, msg.str().c_str() );

      VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);

      err = true;
    }

    if(visitstate == 2 || visitstate == 3)
    {
      // If in step mode return control back to the simulation.
      if( sim->runMode == VISIT_SIMMODE_STEP )
      {
        sim->simMode = VISIT_SIMMODE_RUNNING;

        if(sim->isProc0)
        {
          VisItUI_setValueS("SIMULATION_MODE", "Running", 1);

          std::stringstream msg;
          msg << "Visit libsim - Continuing the simulation for one time step";

          // DOUT( visitdbg, msg.str().c_str() );

          VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
        }

        // Swap over to stop mode because if an error occurs VisIt
        // will otherwise advance to the next time step.
        sim->runMode = VISIT_SIMMODE_STOPPED;

        sim->blocking = 0;
        break;
      }

      // If finished return control back to the simulation.
      else if( sim->runMode == VISIT_SIMMODE_FINISHED )
      {
        if(sim->isProc0)
        {
          VisItUI_setValueS("SIMULATION_MODE", "Not connected", 1);

          std::stringstream msg;
          msg << "Visit libsim - Finished the simulation ";

          // DOUT( visitdbg, msg.str().c_str() );

          VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
          VisItUI_setValueS("SIMULATION_MESSAGE", " ", 1);
        }

        sim->blocking = 0;
        break;
      }
    }
  } while(err == false);

  return (sim->simMode == VISIT_SIMMODE_TERMINATED);
}


//---------------------------------------------------------------------
// UpdateSimData
//     Update the simulation data on all processors
//---------------------------------------------------------------------
void visit_UpdateSimData( visit_simulation_data *sim,
                          GridP currentGrid,
                          bool first, bool last )
{
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  // Update all of the simulation grid and time dependent variables.
  sim->gridP     = currentGrid;

  sim->time      = appInterface->getSimTime();
  sim->cycle     = appInterface->getTimeStep();

  sim->first     = first;

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

      DOUT( visitdbg, msg.str().c_str() );

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
    msg << "Visit libsim - Processor " << sim->myworld->myRank() << " connected";
  }
  else
  {
    msg << "Visit libsim - Connected";
  }

  if(sim->isProc0)
  {
    VisItUI_setValueS("SIMULATION_MESSAGE_CLEAR", "NoOp", 1);
    VisItUI_setValueS("STRIP_CHART_CLEAR_ALL",    "NoOp", 1);

    // DOUT( visitdbg, msg.str().c_str() );

    VisItUI_setValueS("SIMULATION_MESSAGE", msg.str().c_str(), 1);
    VisItUI_setValueS("SIMULATION_MODE", "Connected", 1);
  }

  if( Parallel::usingMPI() )
    VisItSetSlaveProcessCallback(visit_SlaveProcessCallback);

  // Register command callback
  VisItSetCommandCallback(visit_ControlCommandCallback, (void*) sim);

  // These are one time initializations.
  VisItUI_setValueI("SIMULATION_TIME_LIMITS_ENABLED", sim->timeRange, 1);
  VisItUI_setValueI("SIMULATION_TIME_START_CYCLE",    sim->timeStart, 1);
  VisItUI_setValueI("SIMULATION_TIME_STEP_CYCLE",     sim->timeStep,  1);
  VisItUI_setValueI("SIMULATION_TIME_STOP_CYCLE",     sim->timeStop,  1);

  // Register data access callbacks
  VisItSetGetMetaData(visit_SimGetMetaData, (void*) sim);
  VisItSetGetMesh(    visit_SimGetMesh,     (void*) sim);
  VisItSetGetVariable(visit_SimGetVariable, (void*) sim);

  // Register AMR data access callbacks
  VisItSetGetDomainBoundaries(visit_SimGetDomainBoundaries, (void*) sim);
  VisItSetGetDomainNesting   (visit_SimGetDomainNesting,    (void*) sim);

  if( Parallel::usingMPI() )
    VisItSetGetDomainList(visit_SimGetDomainList, (void*) sim);


  VisItUI_textChanged("TimeStepsMax", visit_TimeStepsMaxCallback, (void*) sim);
  VisItUI_textChanged("SimTime",      visit_SimTimeMaxCallback,   (void*) sim);
  VisItUI_valueChanged("SimTimeEndAtMax",
                       visit_SimTimeEndAtMaxCallback, (void*) sim);

  VisItUI_cellChanged("DeltaTVariableTable",
                      visit_DeltaTVariableCallback,          (void*) sim);
  VisItUI_cellChanged("WallTimesVariableTable",
                      visit_WallTimesVariableCallback,       (void*) sim);
  VisItUI_cellChanged("UPSVariableTable",
                      visit_UPSVariableCallback,             (void*) sim);
  VisItUI_cellChanged("OutputIntervalVariableTable",
                      visit_OutputIntervalVariableCallback,  (void*) sim);
  VisItUI_valueChanged("ClampTimeToOutput",
                       visit_ClampTimeToOutputCallback,      (void*) sim);

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

  VisItUI_valueChanged("LoadExtraGeometry",
                       visit_LoadExtraGeometryCallback, (void*) sim);
  VisItUI_valueChanged("LoadVariables",
                       visit_LoadVariablesCallback, (void*) sim);
}

} // End namespace Uintah
