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

#include "visit_libsim.h"

#include <Core/Parallel/Parallel.h>
#include <Core/Grid/Variables/VarLabel.h>

#include "StandAlone/tools/uda2vis/uda2vis.h"

#include <iostream.h>
#include <stdio.h>
#include <string.h>

namespace Uintah {

#define VISIT_COMMAND_PROCESS 0
#define VISIT_COMMAND_SUCCESS 1
#define VISIT_COMMAND_FAILURE 2

//---------------------------------------------------------------------
// ProcessLibSimArguments
//     This routine handles command line arguments
//     -dir <VisIt directory> 
//     -options <VisIt Options> 
//     -trace <VisIt trace file>
//---------------------------------------------------------------------
void visit_LibSimArguments(int argc, char **argv)
{
  for( int i=1; i<argc; ++i )
  {
    if( strcmp( argv[i], "-dir" ) == 0 )
    {
      VisItSetDirectory(argv[++i]);
    }
    else if( strcmp( argv[i], "-options" ) == 0 )
    {
      VisItSetOptions(argv[++i]);
    }
    else if( strcmp( argv[i], "-trace" ) == 0 )
    {
      VisItOpenTraceFile(argv[++i]);
    }
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

#ifdef HAVE_MPICH
  std::cerr << "****************************  "
	    << "usingMPI " << (Parallel::usingMPI() ? "Yes" : "No")
	    << std::endl;
      
  if( Parallel::usingMPI() )
  {
    sim->isProc0 = isProc0_macro;

    int par_rank, par_size;

    // Initialize MPI
    MPI_Comm_rank (MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &par_size);
    
    // Tell libsim whether the simulation is parallel.
    VisItSetParallel( par_size > 1 );
    VisItSetParallelRank( par_rank );

    std::cerr << "****************************  "
	      << " isProc0 " << isProc0_macro << "  "
	      << "par_size " << par_size << "  " 
	      << "par_rank " << par_rank << "  " 
	      << std::cerr;

    // Install callback functions for global communication.
    VisItSetBroadcastIntFunction( visit_BroadcastIntCallback );
    VisItSetBroadcastStringFunction( visit_BroadcastStringCallback );
  }
  else
  {
    sim->isProc0 = 1;
  }
#endif

  // Have the master process create the sim file.
#ifdef HAVE_MPICH
  if(sim->isProc0)
#endif
    VisItInitializeSocketAndDumpSimFile("Uintah",
					"Uintah Simulation",
					"/no/useful/path",
					NULL, NULL, NULL);
}


//---------------------------------------------------------------------
// EndLibSim
//     End the VisIt Lib Sim - but let the user disconnet
//---------------------------------------------------------------------
void visit_EndLibSim( visit_simulation_data *sim )
{
  if( VisItIsConnected() )
  {
    // The simulation is finished but the user may want to stay
    // conntected to analyze the last time step. So stop the run mode
    // but do not let the simulation exit until the user says they are
    // finished.
    sim->runMode = VISIT_SIMMODE_STOPPED;
    sim->simMode = VISIT_SIMMODE_FINISHED;

    // Now check for the user to have finished.
    do
    {
      visit_CheckState(sim);
    }
    while( sim->runMode != VISIT_SIMMODE_FINISHED );
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

    /* Get input from VisIt or timeout so the simulation can run. */
    int visitstate;

#ifdef HAVE_MPICH
    if(sim->isProc0)
#endif
      visitstate = VisItDetectInput(blocking, -1);

#ifdef HAVE_MPICH
    if( Parallel::usingMPI() )
      MPI_Bcast(&visitstate, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    /* Do different things depending on the output from VisItDetectInput. */
    if(visitstate <= -1 || 5 <= visitstate)
    {
      fprintf(stderr, "Can’t recover from error!\n");
      err = 1;
    }
    else if(visitstate == 0)
    {
      /* There was no input from VisIt, return control to sim. */
      return;
    }
    else if(visitstate == 1)
    {
      /* VisIt is trying to connect to sim. */
      if(VisItAttemptToCompleteConnection())
      {
#ifdef HAVE_MPICH
	VisItSetSlaveProcessCallback(visit_SlaveProcessCallback);
#endif
	/* Register command callback */
	VisItSetCommandCallback(visit_ControlCommandCallback, (void*) sim);

	fprintf(stderr, "VisIt connected\n");

	/* Register data access callbacks */
	VisItSetGetMetaData(visit_SimGetMetaData, (void*)sim);

 	VisItSetGetMesh(visit_SimGetMesh, (void*)sim);

	VisItSetGetVariable(visit_SimGetVariable, (void*)sim);
      }
      else
      {
	fprintf(stderr, "VisIt did not connect\n");
      }
    }
    else if(visitstate == 2)
    {
#ifdef HAVE_MPICH
      if(!visit_ProcessVisItCommand(sim))
#else
      if(!VisItProcessEngineCommand())
#endif
      {
	/* Disconnect on an error or closed connection. */
	VisItDisconnect();

	/* Start running again if VisIt closes. */
	sim->runMode = VISIT_SIMMODE_RUNNING;
      }

      /* If in step mode or have finished return control back to the
	 simulation. */
      if( sim->runMode == VISIT_SIMMODE_STEP ||
	  sim->runMode == VISIT_SIMMODE_FINISHED )
	return;
    }
  } while(err == 0);
}


//---------------------------------------------------------------------
// visit_BroadcastIntCallback
//     Callback for processing integers
//---------------------------------------------------------------------
#ifdef HAVE_MPICH
static int visit_BroadcastIntCallback(int *value, int sender)
{
  if( Parallel::usingMPI() )
    return MPI_Bcast(value, 1, MPI_INT, sender, MPI_COMM_WORLD);
  else
    return 0;
}
#endif


//---------------------------------------------------------------------
// visit_BroadcastStringCallback
//     Callback for processing strings
//---------------------------------------------------------------------
#ifdef HAVE_MPICH
static int visit_BroadcastStringCallback(char *str, int len, int sender)
{
  if( Parallel::usingMPI() )
    return MPI_Bcast(str, len, MPI_CHAR, sender, MPI_COMM_WORLD);
  else
    return 0;
}
#endif


//---------------------------------------------------------------------
// visit_BroadcastSlaveCommand
//     Helper function for ProcessVisItCommand
//---------------------------------------------------------------------
#ifdef HAVE_MPICH
static void visit_BroadcastSlaveCommand(int *command)
{
  if( Parallel::usingMPI() )
    MPI_Bcast(command, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
#endif


//---------------------------------------------------------------------
// visit_SlaveProcessCallback
//     Callback involved in command communication.
//---------------------------------------------------------------------
#ifdef HAVE_MPICH
void visit_SlaveProcessCallback()
{
  int command = VISIT_COMMAND_PROCESS;
  visit_BroadcastSlaveCommand(&command);
}
#endif


//---------------------------------------------------------------------
// visit_ControlCommandCallback
//     Process user commands from the viewer on all processors.
//---------------------------------------------------------------------
void
visit_ControlCommandCallback(const char *cmd, const char *args, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  if(strcmp(cmd, "Stop") == 0)
    sim->runMode = VISIT_SIMMODE_STOPPED;
  else if(strcmp(cmd, "Step") == 0)
    sim->runMode = VISIT_SIMMODE_STEP;
  else if(strcmp(cmd, "Run") == 0)
    sim->runMode = VISIT_SIMMODE_RUNNING;
  else if(strcmp(cmd, "Exit") == 0)
    exit( 0 );
  // Only allow the runMode to finish if the simulation is finished.
  else if(strcmp(cmd, "Finish") == 0 && 
	  sim->simMode == VISIT_SIMMODE_FINISHED)
  {
    sim->runMode = VISIT_SIMMODE_FINISHED;
  }
}


//---------------------------------------------------------------------
// ProcessVisItCommand
//     Process commands from the viewer on all processors.
//---------------------------------------------------------------------
#ifdef HAVE_MPICH
int visit_ProcessVisItCommand( visit_simulation_data *sim )
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

  return 1;
}
#endif


//---------------------------------------------------------------------
// visit_SimGetMetaData
//     Callback for processing meta data
//---------------------------------------------------------------------
visit_handle visit_SimGetMetaData(void *cbdata)
{
#ifdef SERIALIZED_READS

  int numProcs, rank;
  int msg = 128, tag = 256;
  MPI_Status status;

  MPI_Comm_size(VISIT_MPI_COMM, &numProcs);
  MPI_Comm_rank(VISIT_MPI_COMM, &rank);
  //debug5 << "Proc: " << rank << " sent to mdserver" << std::endl;  

  if (rank == 0) {
    ReadMetaData(md, timeState);
    MPI_Send(&msg, 1, MPI_INT, 1, tag, VISIT_MPI_COMM);
  }
  else {
    MPI_Recv(&msg, 1, MPI_INT, rank - 1, tag, VISIT_MPI_COMM, &status);
    if (msg == 128 && tag == 256) {
      return visit_ReadMetaData(cbdata);
      if (rank < (numProcs - 1))
        MPI_Send(&msg, 1, MPI_INT, rank + 1, tag, VISIT_MPI_COMM);
    }
  }
#else      
  return visit_ReadMetaData(cbdata);
#endif
}


// ****************************************************************************
//  Method: ReadMetaData
//
//  Purpose:
//      Does the actual work for visit_SimGetMetaData()
//
// ****************************************************************************
visit_handle visit_ReadMetaData(void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

#ifdef HAVE_MPICH
  if(!sim->isProc0)
    return VISIT_INVALID_HANDLE;
#endif

  SchedulerP schedulerP = sim->schedulerP;
  GridP      gridP      = sim->gridP;

  int timestep = sim->cycle;

  sim->useExtraCells = true;
  bool &useExtraCells = sim->useExtraCells;

  sim->forceMeshReload = true;
  bool &forceMeshReload = sim->forceMeshReload;

  sim->nodeCentered = false;
  bool &nodeCentered = sim->nodeCentered;


  if( !schedulerP.get_rep() || !gridP.get_rep() )
  {
    return VISIT_INVALID_HANDLE;
  }

  sim->stepInfo = getTimeStepInfo2(schedulerP,
				   gridP,
				   timestep,
				   useExtraCells);
  TimeStepInfo* &stepInfo = sim->stepInfo;

  visit_handle md = VISIT_INVALID_HANDLE;

  /* Create metadata with no variables. */
  if(VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY)
  {
    /* Set the simulation state. */
    if(sim->runMode == VISIT_SIMMODE_STOPPED ||
       sim->runMode == VISIT_SIMMODE_STEP)
      VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_STOPPED);
    else if(sim->runMode == VISIT_SIMMODE_RUNNING)
      VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_RUNNING);

    VisIt_SimulationMetaData_setCycleTime(md, sim->cycle, sim->time);

    bool forceMeshReload = true;

    int numLevels = stepInfo->levelInfo.size();
    
    int totalPatches = 0;
    for (int i = 0; i < numLevels; i++)
      totalPatches +=  stepInfo->levelInfo[i].patchInfo.size();
  //debug5 << "avtUintahFileFormat::ReadMetaData: Levels: " << numLevels << " Patches: " << totalPatches << std::endl;

    std::vector<int> groupIds(totalPatches);
    std::vector<std::string> pieceNames(totalPatches);

    for (int i = 0; i < totalPatches; i++) {
      char tmpName[64];
      int level, local_patch;
      
      GetLevelAndLocalPatchNumber(stepInfo, i, level, local_patch);
      sprintf(tmpName,"level%d, patch%d", level, local_patch);
      
      groupIds[i] = level;
      pieceNames[i] = tmpName;
    }
    
    // compute the bounding box of the mesh from the grid indices of
    // level 0
    LevelInfo &levelInfo = stepInfo->levelInfo[0];

    // don't add proc id unless CC_Mesh or NC_Mesh exists (some only
    // have SFCk_MESH)
    bool addProcId = false;
    std::string mesh_for_procid("NC_Mesh");

    // grid meshes are shared between materials, and particle meshes are
    // shared between variables - keep track of what has been added so
    // they're only added once
    std::set<std::string> meshes_added;

    // If a variable exists in multiple materials, we don't want to add
    // it more than once to the meta data - it can mess up visit's
    // expressions variable lists.
    std::set<std::string> mesh_vars_added;

    // get CC bounds
    int low[3],high[3];
    getBounds(low,high,"CC_Mesh",levelInfo);

    // this can be done once for everything because the spatial range is
    // the same for all meshes
    double box_min[3] = { levelInfo.anchor[0] + low[0] * levelInfo.spacing[0],
			  levelInfo.anchor[1] + low[1] * levelInfo.spacing[1],
			  levelInfo.anchor[2] + low[2] * levelInfo.spacing[2] };

    double box_max[3] = { levelInfo.anchor[0] + high[0] * levelInfo.spacing[0],
			  levelInfo.anchor[1] + high[1] * levelInfo.spacing[1],
			  levelInfo.anchor[2] + high[2] * levelInfo.spacing[2] };
    // debug5 << "box_min/max=["
    // 	   << box_min[0] << "," << box_min[1] << ","
    // 	   << box_min[2] << "] to ["
    // 	   << box_max[0] << "," << box_max[1] << ","
    // 	   << box_max[2] << "]" << std::endl;

    int logical[3];
    for (int i=0; i<3; i++)
      logical[i] = high[i] - low[i];

    // debug5 << "logical: " << logical[0] << ", " << logical[1] << ", "
    // 	   << logical[2] << std::endl;

    for (int i=0; i<(int)stepInfo->varInfo.size(); i++)
    {
      if (stepInfo->varInfo[i].type.find("ParticleVariable") ==
	  std::string::npos)
      {
	std::string varname = stepInfo->varInfo[i].name;
	std::string vartype = stepInfo->varInfo[i].type;
	
	std::string mesh_for_this_var;
	VisIt_VarCentering cent = VISIT_VARCENTERING_ZONE;
	//    avtCentering cent = AVT_ZONECENT;

	if (vartype.find("NC") != std::string::npos)
	{
	  cent = VISIT_VARCENTERING_NODE;
//        cent = AVT_NODECENT;
	  mesh_for_this_var.assign("NC_Mesh"); 
	  addProcId = true;
	}
	else if (vartype.find("CC") != std::string::npos)
	{
	  cent = VISIT_VARCENTERING_ZONE;
//        cent = AVT_ZONECENT;
	  mesh_for_this_var.assign("CC_Mesh");
	  addProcId = true;
	  mesh_for_procid=mesh_for_this_var;
	}
	else if (vartype.find("SFC") != std::string::npos)
        { 
	  cent = VISIT_VARCENTERING_ZONE;
//        cent = AVT_ZONECENT;

	  if (vartype.find("SFCX") != std::string::npos)               
	    mesh_for_this_var.assign("SFCX_Mesh");
	  else if (vartype.find("SFCY") != std::string::npos)          
	    mesh_for_this_var.assign("SFCY_Mesh");
	  else if (vartype.find("SFCZ") != std::string::npos)          
	    mesh_for_this_var.assign("SFCZ_Mesh");
	}
	else
	  std::cerr << "Uintah/VisIt Libsim Error: unknown vartype: "
		    << vartype << std::endl;

	if (meshes_added.find(mesh_for_this_var) == meshes_added.end())
	{
	  // Mesh meta data
	  visit_handle mmd = VISIT_INVALID_HANDLE;
	  
	  /* Set the first mesh’s properties.*/
	  if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
	  {
	    /* Set the mesh’s properties.*/
	    VisIt_MeshMetaData_setName(mmd, mesh_for_this_var.c_str());
	    VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_AMR);
	    VisIt_MeshMetaData_setTopologicalDimension(mmd, 3);
	    VisIt_MeshMetaData_setSpatialDimension(mmd, 3);
	    
	    VisIt_MeshMetaData_setNumDomains(mmd, totalPatches);
	    VisIt_MeshMetaData_setDomainTitle(mmd, "patches");
	    VisIt_MeshMetaData_setDomainPieceName(mmd, "patch");
	    VisIt_MeshMetaData_setNumGroups(mmd, numLevels);
	    VisIt_MeshMetaData_setGroupTitle(mmd, "levels");
	    VisIt_MeshMetaData_setGroupPieceName(mmd, "level");

	    for(unsigned int k = 0; k < totalPatches; ++k)
	      VisIt_MeshMetaData_addGroupId(mmd, groupIds[k]);

	  // ARS - FIXME
//	  VisIt_MeshMetaData_setBlockNames(mmd, pieceNames);
//	  VisIt_MeshMetaData_setContainsExteriorBoundaryGhosts(mmd, false);

	    VisIt_MeshMetaData_setHasSpatialExtents(mmd, 1);

	    double extents[6] = { box_min[0], box_max[0],
				  box_min[1], box_max[1],
				  box_min[2], box_max[2] };

	    VisIt_MeshMetaData_setSpatialExtents(mmd, extents);

	    // ARS - FIXME
	    // VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
	    // VisIt_MeshMetaData_logicalBounds(mmd, logical[0]);

	    VisIt_SimulationMetaData_addMesh(md, mmd);
	  }

	  // avtMeshMetaData *mesh = new avtMeshMetaData;

	  // mesh->name = mesh_for_this_var;
	  // mesh->meshType = AVT_AMR_MESH;
	  // mesh->topologicalDimension = 3;
	  // mesh->spatialDimension = 3;

	  // mesh->numBlocks = totalPatches;
	  // mesh->blockTitle = "patches";
	  // mesh->blockPieceName = "patch";
	  // mesh->numGroups = numLevels;

	  // mesh->groupPieceName = "level";
	  // mesh->blockNames = pieceNames;
	  // mesh->containsExteriorBoundaryGhosts = false;

	  // mesh->hasSpatialExtents = true; 
	  // mesh->minSpatialExtents[0] = box_min[0];
	  // mesh->maxSpatialExtents[0] = box_max[0];
	  // mesh->minSpatialExtents[1] = box_min[1];
	  // mesh->maxSpatialExtents[1] = box_max[1];
	  // mesh->minSpatialExtents[2] = box_min[2];
	  // mesh->maxSpatialExtents[2] = box_max[2];

	  // mesh->hasLogicalBounds = true;
	  // mesh->logicalBounds[0] = logical[0];
	  // mesh->logicalBounds[1] = logical[1];
	  // mesh->logicalBounds[2] = logical[2];

	  // md->Add(mesh);
	  meshes_added.insert(mesh_for_this_var);
	}

	// Add mesh vars
	for (int j=0; j<(int)stepInfo->varInfo[i].materials.size(); j++)
	{
	  char buffer[128];
	  std::string newVarname = varname;
	  sprintf(buffer, "%d", stepInfo->varInfo[i].materials[j]);
	  newVarname.append("/");
	  newVarname.append(buffer);

	  if (mesh_vars_added.find(mesh_for_this_var+newVarname) ==
	      mesh_vars_added.end())
	  {
	    mesh_vars_added.insert(mesh_for_this_var+newVarname);
	    
	    visit_handle vmd = VISIT_INVALID_HANDLE;
	  
	    if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
	    {
	      VisIt_VariableMetaData_setName(vmd, newVarname.c_str());
	      VisIt_VariableMetaData_setMeshName(vmd, mesh_for_this_var.c_str());
	      VisIt_VariableMetaData_setCentering(vmd, cent);

	      // 3 -> vector dimension
	      if (vartype.find("Vector") != std::string::npos)
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 3);
//          AddVectorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 3);
	      }
	      // 9 -> tensor 
	      else if (vartype.find("Matrix3") != std::string::npos)
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_TENSOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 9);
//          AddTensorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 9);
	      }
	      // 7 -> vector
	      else if (vartype.find("Stencil7") != std::string::npos)
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 7);
//          AddVectorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 7);
	      }
	      // 4 -> vector
	      else if (vartype.find("Stencil4") != std::string::npos)
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 4);
//          AddVectorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 4);
	      }
	      // scalar
	      else 
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
		VisIt_VariableMetaData_setNumComponents(vmd, 1);
//          AddScalarVarToMetaData(md, newVarname, mesh_for_this_var, cent);
	      }

	      VisIt_SimulationMetaData_addVariable(md, vmd);
	    }
	  }
	}
      }   
    }

    // add a proc id enum variable
    if (addProcId)
    {
      visit_handle vmd = VISIT_INVALID_HANDLE;
      
      if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
      {
	VisIt_VariableMetaData_setName(vmd, "proc_id");
	VisIt_VariableMetaData_setMeshName(vmd, mesh_for_procid.c_str());
	VisIt_VariableMetaData_setCentering(vmd,  VISIT_VARCENTERING_ZONE);
	VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
	VisIt_VariableMetaData_setNumComponents(vmd, 1);
	// ARS - FIXME
//      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
	VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
	VisIt_SimulationMetaData_addVariable(md, vmd);
      }

      // avtScalarMetaData *scalar = new avtScalarMetaData();
      
      // scalar->name = "proc_id";
      // scalar->meshName = mesh_for_procid;
      // scalar->centering = AVT_ZONECENT;
      // scalar->hasDataExtents = false;
      // scalar->treatAsASCII = false;
      // md->Add(scalar);
    }
  

    // Nothing needs to be modifed for particle data, as they exist only
    // on a single level
    for (int i=0; i<(int)stepInfo->varInfo.size(); i++)
    {
      if (stepInfo->varInfo[i].type.find("ParticleVariable") != std::string::npos)
      {
	std::string varname = stepInfo->varInfo[i].name;
	std::string vartype = stepInfo->varInfo[i].type;
	
	// j=-1 -> all materials (*)
	for (int j=-1; j<(int)stepInfo->varInfo[i].materials.size(); j++)
	{
	  std::string mesh_for_this_var = std::string("Particle_Mesh/");
	  std::string newVarname = varname+"/";
	  
	  if (j >= 0)
	  {
	    char buffer[128];
	    sprintf(buffer, "%d", stepInfo->varInfo[i].materials[j]);
	    mesh_for_this_var.append(buffer);
	    newVarname.append(buffer);
	  }
	  else
	  {
	    mesh_for_this_var.append("*");
	    newVarname.append("*");
	  }

	  if (meshes_added.find(mesh_for_this_var)==meshes_added.end())
	  {
	    // Mesh meta data
	    visit_handle mmd = VISIT_INVALID_HANDLE;
	    
	    /* Set the first mesh’s properties.*/
	    if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
	    {
	      /* Set the mesh’s properties.*/
	      VisIt_MeshMetaData_setName(mmd, mesh_for_this_var.c_str());
	      VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_POINT);
	      VisIt_MeshMetaData_setTopologicalDimension(mmd, 0);
	      VisIt_MeshMetaData_setSpatialDimension(mmd, 3);
	      
	      VisIt_MeshMetaData_setNumDomains(mmd, totalPatches);
	      VisIt_MeshMetaData_setDomainTitle(mmd, "patches");
	      VisIt_MeshMetaData_setDomainPieceName(mmd, "patch");
	      VisIt_MeshMetaData_setNumGroups(mmd, numLevels);
	      VisIt_MeshMetaData_setGroupTitle(mmd, "levels");
	      VisIt_MeshMetaData_setGroupPieceName(mmd, "level");
	      for(unsigned int k = 0; k < totalPatches; ++k)
		VisIt_MeshMetaData_addGroupId(mmd, groupIds[k]);

	      // ARS - FIXME
	      // VisIt_MeshMetaData_setBlockNames(mmd, pieceNames );

	      VisIt_MeshMetaData_setHasSpatialExtents(mmd, 1);

	      double extents[6] = { box_min[0], box_max[0],
				    box_min[1], box_max[1],
				    box_min[2], box_max[2] };

	      VisIt_MeshMetaData_setSpatialExtents(mmd, extents);

	      // ARS - FIXME
	      // VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
	      // VisIt_MeshMetaData_seteLogicalBounds(mmd, logical[0]);

	      VisIt_SimulationMetaData_addMesh(md, mmd);
	    }

	    // avtMeshMetaData *mesh = new avtMeshMetaData;

	    // mesh->name = mesh_for_this_var;
	    // mesh->meshType = AVT_POINT_MESH;
	    // mesh->topologicalDimension = 0;
	    // mesh->spatialDimension = 3;

	    // mesh->numBlocks = totalPatches;
	    // mesh->blockTitle = "patches";
	    // mesh->blockPieceName = "patch";
	    // mesh->numGroups = numLevels;
	    // mesh->groupTitle = "levels";
	    // mesh->groupPieceName = "level";
	    // mesh->blockNames = pieceNames;

	    // mesh->hasSpatialExtents = true; 
	    // mesh->minSpatialExtents[0] = box_min[0];
	    // mesh->maxSpatialExtents[0] = box_max[0];
	    // mesh->minSpatialExtents[1] = box_min[1];
	    // mesh->maxSpatialExtents[1] = box_max[1];
	    // mesh->minSpatialExtents[2] = box_min[2];
	    // mesh->maxSpatialExtents[2] = box_max[2];

	    // mesh->hasLogicalBounds = true;
	    // mesh->logicalBounds[0] = logical[0];
	    // mesh->logicalBounds[1] = logical[1];
	    // mesh->logicalBounds[2] = logical[2];

	    // md->Add(mesh);
	    meshes_added.insert(mesh_for_this_var);
	  }

	  if (mesh_vars_added.find(mesh_for_this_var+newVarname) ==
	      mesh_vars_added.end())
	  {
	    mesh_vars_added.insert(mesh_for_this_var+newVarname);
	    
	    VisIt_VarCentering cent = VISIT_VARCENTERING_NODE;
//                avtCentering cent = AVT_NODECENT;

	    visit_handle vmd = VISIT_INVALID_HANDLE;
	    
	    if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
	    {
	      VisIt_VariableMetaData_setName(vmd, newVarname.c_str());
	      VisIt_VariableMetaData_setMeshName(vmd, mesh_for_this_var.c_str());
	      VisIt_VariableMetaData_setCentering(vmd, cent);

	      // 3 -> vector dimension
	      if ((vartype.find("Vector") != std::string::npos) ||
		  (vartype.find("Point") != std::string::npos))
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 3);
//          AddVectorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 3);
	      }
	      // 9 -> tensor 
	      else if (vartype.find("Matrix3") != std::string::npos)
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_TENSOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 9);
//          AddTensorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 9);
	      }
	      // 7 -> vector
	      else if (vartype.find("Stencil7") != std::string::npos)
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 7);
//          AddVectorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 7);
	      }
	    // 4 -> vector
	      else if (vartype.find("Stencil4") != std::string::npos)
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
		VisIt_VariableMetaData_setNumComponents(vmd, 4);
//          AddVectorVarToMetaData(md, newVarname, mesh_for_this_var, cent, 4);
	      }
	    // scalar
	      else
	      {
		VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
		VisIt_VariableMetaData_setNumComponents(vmd, 1);
 //         AddScalarVarToMetaData(md, newVarname, mesh_for_this_var, cent);
	      }
	      
	      VisIt_SimulationMetaData_addVariable(md, vmd);
	    }
	  }
	}
      }   
    }

    // ARS - FIXME
    // md->AddGroupInformation(numLevels, totalPatches, groupIds);
    // md->AddDefaultSILRestrictionDescription(std::string("!TurnOnAll"));


    // ARS - NOT NEEDED FOR LIBSIM
    // Set the cycles and times.
    // md->SetCyclesAreAccurate(true);

    // std::vector<int> cycles;

    // cycles.resize( cycleTimes.size() );

    // for(int i=0; i<(int)cycleTimes.size(); ++i )
    //   cycles[i] = i;

    // md->SetCycles( cycles );

    // md->SetTimesAreAccurate(true);
    // md->SetTimes( cycleTimes );
  
    // AddExpressionsToMetadata(md);



#ifdef COMMENT_OUT_THIS_EXAMPLE_CODE
    // Mesh meta data
    visit_handle mmd = VISIT_INVALID_HANDLE;
    visit_handle m2 = VISIT_INVALID_HANDLE;
    
    /* Set the first mesh’s properties.*/
    if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
    {
      /* Set the mesh’s properties.*/
      VisIt_MeshMetaData_setName(mmd, "mesh2d");
      VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_RECTILINEAR);
      VisIt_MeshMetaData_setTopologicalDimension(mmd, 2);
      VisIt_MeshMetaData_setSpatialDimension(mmd, 2);
      VisIt_MeshMetaData_setXUnits(mmd, "cm");
      VisIt_MeshMetaData_setYUnits(mmd, "cm");
      VisIt_MeshMetaData_setXLabel(mmd, "Width");
      VisIt_MeshMetaData_setYLabel(mmd, "Height");
      VisIt_SimulationMetaData_addMesh(md, mmd);
    }
    /* Set the second mesh’s properties.*/
    if(VisIt_MeshMetaData_alloc(&m2) == VISIT_OKAY)
    {
      /* Set the mesh’s properties.*/
      VisIt_MeshMetaData_setName(m2, "mesh3d");
      VisIt_MeshMetaData_setMeshType(m2, VISIT_MESHTYPE_CURVILINEAR);
      VisIt_MeshMetaData_setTopologicalDimension(m2, 3);
      VisIt_MeshMetaData_setSpatialDimension(m2, 3);
      VisIt_MeshMetaData_setXUnits(m2, "cm");
      VisIt_MeshMetaData_setYUnits(m2, "cm");
      VisIt_MeshMetaData_setZUnits(m2, "cm");
      VisIt_MeshMetaData_setXLabel(m2, "Width");
      VisIt_MeshMetaData_setYLabel(m2, "Height");
      VisIt_MeshMetaData_setZLabel(m2, "Depth");
      VisIt_SimulationMetaData_addMesh(md, m2);
    }


    // Scalar variables
    visit_handle vmd1 = VISIT_INVALID_HANDLE;
    visit_handle vmd2 = VISIT_INVALID_HANDLE;

    /* Add a zonal scalar variable on mesh2d. */
    if(VisIt_VariableMetaData_alloc(&vmd1) == VISIT_OKAY)
    {
      VisIt_VariableMetaData_setName(vmd1, "zonal");
      VisIt_VariableMetaData_setMeshName(vmd1, "mesh2d");
      VisIt_VariableMetaData_setType(vmd1, VISIT_VARTYPE_VECTOR);
      VisIt_VariableMetaData_setNumComponents(vmd1, 3);
      VisIt_VariableMetaData_setCentering(vmd1, VISIT_VARCENTERING_ZONE);
      VisIt_SimulationMetaData_addVariable(md, vmd1);
    }
    
    /* Add a nodal scalar variable on mesh3d. */
    if(VisIt_VariableMetaData_alloc(&vmd2) == VISIT_OKAY)
    {
      VisIt_VariableMetaData_setName(vmd2, "nodal");
      VisIt_VariableMetaData_setMeshName(vmd2, "mesh3d");
      VisIt_VariableMetaData_setType(vmd2, VISIT_VARTYPE_SCALAR);
      VisIt_VariableMetaData_setCentering(vmd2, VISIT_VARCENTERING_NODE);
      VisIt_SimulationMetaData_addVariable(md, vmd2);
    }

    /* Add some expressions. */
    visit_handle emd1 = VISIT_INVALID_HANDLE;
    visit_handle emd2 = VISIT_INVALID_HANDLE;

    if(VisIt_ExpressionMetaData_alloc(&emd1) == VISIT_OKAY)
    {
      VisIt_ExpressionMetaData_setName(emd1, "zvec");
      VisIt_ExpressionMetaData_setDefinition(emd1, "{zonal, zonal}");
      VisIt_ExpressionMetaData_setType(emd1, VISIT_VARTYPE_VECTOR);
      VisIt_SimulationMetaData_addExpression(md, emd1);
    }

    if(VisIt_ExpressionMetaData_alloc(&emd2) == VISIT_OKAY)
    {
      VisIt_ExpressionMetaData_setName(emd2, "nid");
      VisIt_ExpressionMetaData_setDefinition(emd2, "nodeid(mesh3d)");
      VisIt_ExpressionMetaData_setType(emd2, VISIT_VARTYPE_SCALAR);
    }
#endif

    /* Add some commands. */
    const char *cmd_names[] = {"Stop", "Step", "Run", "Exit", "Finish"};

    for(int i = 0; i < sizeof(cmd_names)/sizeof(const char *); ++i)
    {
      visit_handle cmd = VISIT_INVALID_HANDLE;

      if(VisIt_CommandMetaData_alloc(&cmd) == VISIT_OKAY)
      {
	VisIt_CommandMetaData_setName(cmd, cmd_names[i]);
	VisIt_SimulationMetaData_addGenericCommand(md, cmd);
      }
    }
  }

  return md;
}


// ****************************************************************************
//  Method: visit_CalculateDomainNesting
//
//  Purpose:
//      Calculates two important data structures.  One is the structure domain
//      nesting, which tells VisIt how the AMR patches are nested, which allows
//      VisIt to ghost out coarse zones that are refined by smaller zones.
//      The other structure is the rectilinear domain boundaries, which tells
//      VisIt which patches are next to each other, allowing VisIt to create
//      a layer of ghost zones around each patch.  Note that this only works
//      within a refinement level, not across refinement levels.
//  
//
// NOTE: The cache variable for the mesh MUST be called "any_mesh",
// which is a problem when there are multiple meshes or one of them is
// actually named "any_mesh" (see
// https://visitbugs.ornl.gov/issues/52). Thus, for each mesh we keep
// around our own cache variable and if this function finds it then it
// just uses it again instead of recomputing it.
//
// ****************************************************************************
void visit_CalculateDomainNesting(TimeStepInfo* stepInfo,
				  bool &forceMeshReload,
				  int timestate, const std::string &meshname)
{
  // ARS - FIX ME - NOT NEEDED
  //lookup mesh in our cache and if it's not there, compute it
  // if (mesh_domains[meshname] == NULL || forceMeshReload == true)
  {
    //
    // Calculate some info we will need in the rest of the routine.
    //
    int num_levels = stepInfo->levelInfo.size();
    int totalPatches = 0;
    for (int level = 0 ; level < num_levels ; level++)
      totalPatches += stepInfo->levelInfo[level].patchInfo.size();

    //
    // Now set up the data structure for patch boundaries.  The data 
    // does all the work ... it just needs to know the extents of each patch.
    //
    visit_handle rdb;
    
    if(VisIt_DomainBoundaries_alloc(&rdb) == VISIT_OKAY)
    {
      VisIt_DomainBoundaries_set_type(rdb, 0); // 0 = Rectilinear
      VisIt_DomainBoundaries_set_numDomains(rdb, totalPatches );

      // debug5 << "Calculating avtRectilinearDomainBoundaries for "
      // 	   << meshname << " mesh (" << rdb << ")." << std::endl;

      // avtRectilinearDomainBoundaries *rdb =
      // 	new avtRectilinearDomainBoundaries(true);
      // rdb->SetNumDomains(totalPatches);

      for (int patch = 0 ; patch < totalPatches ; patch++)
      {
	int my_level, local_patch;
	GetLevelAndLocalPatchNumber(stepInfo, patch, my_level, local_patch);
	
	PatchInfo &patchInfo =
	  stepInfo->levelInfo[my_level].patchInfo[local_patch];

	int low[3],high[3];
	patchInfo.getBounds(low,high,meshname);
	
	int e[6] = { low[0], high[0],
		     low[1], high[1],
		     low[2], high[2] };
	// debug5 << "\trdb->SetIndicesForAMRPatch(" << patch << ","
	// 	     << my_level << ", <" << e[0] << "," << e[2] << "," << e[4]
	// 	     << "> to <" << e[1] << "," << e[3] << "," << e[5] << ">)"
	//             << std::endl;

	VisIt_DomainBoundaries_set_amrIndices(rdb, patch, my_level, e);
//	VisIt_DomainBoundaries_finish(rdb, patch);
	// rdb->SetIndicesForAMRPatch(patch, my_level, e);
      }

      // rdb->CalculateBoundaries();
      
      // ARS - FIX ME - NOT NEEDED 
      // mesh_boundaries[meshname] =
      //    void_ref_ptr(rdb, avtStructuredDomainBoundaries::Destruct);
    }

    //
    // Domain Nesting
    //
    visit_handle dn;
    
    if(VisIt_DomainNesting_alloc(&dn) == VISIT_OKAY)
    {
      VisIt_DomainNesting_set_dimensions(dn, totalPatches, num_levels, 3);

      // avtStructuredDomainNesting *dn =
      // 	new avtStructuredDomainNesting(totalPatches, num_levels);
      // dn->SetNumDimensions(3);

      //debug5 << "Calculating avtStructuredDomainNesting for "
      //       << meshname << " mesh (" << dn << ")." << std::endl;
      
      //
      // Calculate what the refinement ratio is from one level to the next.
      //
      for (int level = 0 ; level < num_levels ; level++) {
	// SetLevelRefinementRatios requires data as a vector<int>
	int rr[3];
	//vector<int> rr(3);
	for (int i=0; i<3; i++)
	  rr[i] = stepInfo->levelInfo[level].refinementRatio[i];
	
	// debug5 << "\tdn->SetLevelRefinementRatios(" << level << ", <"
	//        << rr[0] << "," << rr[1] << "," << rr[2] << ">)\n";

	VisIt_DomainNesting_set_levelRefinement(dn, level, rr);

	// dn->SetLevelRefinementRatios(level, rr);
      }      

      //
      // Calculating the child patches really needs some better sorting than
      // what I am doing here.  This is likely to become a bottleneck in extreme
      // cases.  Although this routine has performed well for a previous 55K
      // patch run.
      //
      std::vector< std::vector<int> > childPatches(totalPatches);
      
      for (int level = num_levels-1 ; level > 0 ; level--)
      {
	int prev_level = level-1;
	LevelInfo &levelInfoParent = stepInfo->levelInfo[prev_level];
	LevelInfo &levelInfoChild = stepInfo->levelInfo[level];
	
	for (int child=0; child<(int)levelInfoChild.patchInfo.size(); child++)
	{
	  PatchInfo &childPatchInfo = levelInfoChild.patchInfo[child];
	  int child_low[3],child_high[3];
	  childPatchInfo.getBounds(child_low,child_high,meshname);
	  
	  for (int parent = 0;
	       parent<(int)levelInfoParent.patchInfo.size(); parent++)
	  {
	    PatchInfo &parentPatchInfo = levelInfoParent.patchInfo[parent];
	    int parent_low[3],parent_high[3];
	    parentPatchInfo.getBounds(parent_low,parent_high,meshname);
	    
	    int mins[3], maxs[3];
	    for (int i=0; i<3; i++)
	    {
	      mins[i] = std::max(child_low[i],
				 parent_low[i]*levelInfoChild.refinementRatio[i]);
	      maxs[i] = std::min(child_high[i],
				 parent_high[i]*levelInfoChild.refinementRatio[i]);
	    }
	    
	    bool overlap = (mins[0] < maxs[0] &&
			    mins[1] < maxs[1] &&
			    mins[2] < maxs[2]);
	    
	    if (overlap)
	    {
	      int child_gpatch = GetGlobalDomainNumber(stepInfo, level, child);
	      int parent_gpatch = GetGlobalDomainNumber(stepInfo, prev_level, parent);
	      childPatches[parent_gpatch].push_back(child_gpatch);
	    }
	  }
	}
      }

      //
      // Now that we know the extents for each patch and what its children are,
      // tell the structured domain boundary that information.
      //
      for (int p=0; p<totalPatches ; p++)
      {
	int my_level, local_patch;
	GetLevelAndLocalPatchNumber(stepInfo, p, my_level, local_patch);
	
	PatchInfo &patchInfo =
	  stepInfo->levelInfo[my_level].patchInfo[local_patch];
	int low[3],high[3];
	patchInfo.getBounds(low,high,meshname);
	
	int e[6];
	for (int i=0; i<3; i++) {
	  e[i+0] = low[i];
	  e[i+3] = high[i]-1;
	}

	// debug5 << "\tdn->SetNestingForDomain("
	//        << p << "," << my_level << ", <>, <"
	//        << e[0] << "," << e[1] << "," << e[2] << "> to <"
	//        << e[3] << "," << e[4] << "," << e[5] << ">)\n";

	if( childPatches[p].size() )
	{
	  int *cp = new int[childPatches[p].size()];
	  
	  for (int i=0; i<3; i++) {
	    cp[i] = childPatches[p][i];
	    
	    VisIt_DomainNesting_set_nestingForPatch(dn, p, my_level,
						    cp, childPatches[p].size(),
						    e);
//	    delete cp;
	    
	    // dn->SetNestingForDomain(p, my_level, childPatches[p], e);
	  }
	}
      }
    }
    
    // ARS - FIX ME - NOT NEEDED
    // mesh_domains[meshname] =
    //    void_ref_ptr(dn, avtStructuredDomainNesting::Destruct);

    forceMeshReload = false;
  }

  // ARS - FIX ME - NOT NEEDED
  //
  // Register these structures with the generic database so that it knows
  // to ghost out the right cells.
  //
  // cache->CacheVoidRef("any_mesh", // key MUST be called any_mesh
  //                     AUXILIARY_DATA_DOMAIN_BOUNDARY_INFORMATION,
  //                     timestate, -1, mesh_boundaries[meshname]);
  // cache->CacheVoidRef("any_mesh", // key MUST be called any_mesh
  //                     AUXILIARY_DATA_DOMAIN_NESTING_INFORMATION,
  //                     timestate, -1, mesh_domains[meshname]);

  //VERIFY we got the mesh boundary and domain in there
  // void_ref_ptr vrTmp =
  //   cache->GetVoidRef("any_mesh", // MUST be called any_mesh
  // 		      AUXILIARY_DATA_DOMAIN_BOUNDARY_INFORMATION,
  // 		      timestate, -1);
  // if (*vrTmp == NULL || *vrTmp != mesh_boundaries[meshname])
  //   throw InvalidFilesException("uda boundary mesh not registered");

  // vrTmp = cache->GetVoidRef("any_mesh", // MUST be called any_mesh
  //                           AUXILIARY_DATA_DOMAIN_NESTING_INFORMATION,
  //                           timestate, -1);
  // if (*vrTmp == NULL || *vrTmp != mesh_domains[meshname])
  //   throw InvalidFilesException("uda domain mesh not registered");
}


//---------------------------------------------------------------------
// visit_SimGetMesh
//     Callback for processing a mesh
//---------------------------------------------------------------------
visit_handle visit_SimGetMesh(int domain, const char *meshname, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SchedulerP schedulerP = sim->schedulerP;
  GridP      gridP      = sim->gridP;

  bool &useExtraCells   = sim->useExtraCells;
  bool &forceMeshReload = sim->forceMeshReload;
  bool &nodeCentered    = sim->nodeCentered;

  TimeStepInfo* &stepInfo = sim->stepInfo;

  int timestate = sim->cycle;

  visit_handle meshH = VISIT_INVALID_HANDLE;

  std::string meshName(meshname);

  int level, local_patch;
  GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

  // particle data
  if (meshName.find("Particle_Mesh") != std::string::npos)
  {
    size_t found = meshName.find("/");
    std::string matl = meshName.substr(found + 1);

    int matlNo = -1;
    if (matl.compare("*") != 0)
      matlNo = atoi(matl.c_str());

    // we always want p.x when setting up the mesh
//    string vars = "p.x";

    const std::string &vars = Uintah::VarLabel::getParticlePositionName();
//  string vars = getParticlePositionName(schedulerP);

    ParticleDataRaw *pd =
      getParticleData2(schedulerP, gridP, level, local_patch, vars,
		       matlNo, timestate);

    visit_handle cordsH = VISIT_INVALID_HANDLE;

    if(VisIt_VariableData_alloc(&cordsH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataD(cordsH, VISIT_OWNER_SIM,
				  3, pd->num*pd->components, pd->data);
    }

    // Create the vtkPoints object and copy points into it.
    // vtkDoubleArray *doubleArray = vtkDoubleArray::New();
    // doubleArray->SetNumberOfComponents(3);
    // doubleArray->SetArray(pd->data, pd->num*pd->components, 0);

    // vtkPoints *points = vtkPoints::New();
    // points->SetData(doubleArray);
    // doubleArray->Delete();

    // 
    // Create a vtkUnstructuredGrid to contain the point cells. 
    // 
    // vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::New(); 
    // ugrid->SetPoints(points); 
    // points->Delete(); 
    // ugrid->Allocate(pd->num); 
    // vtkIdType onevertex; 

    // for(int i = 0; i < pd->num; ++i) {
    //   onevertex = i; 
    //   ugrid->InsertNextCell(VTK_VERTEX, 1, &onevertex); 
    // } 

    // ARS - FIX ME - CHECK FOR LEAKS
    // don't delete pd->data - vtk owns it now!
    // delete pd;

#ifdef COMMENTOUT_FOR_NOW
    //try to retrieve existing cache ref
    void_ref_ptr vrTmp =
      cache->GetVoidRef(meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
			timestate, domain);

    vtkDataArray *pID = NULL;

    if (*vrTmp == NULL)
    {
      //
      // add globel node ids to facilitate point cloud usage
      //
      //basically same as GetVar(timestate, domain, "particleID");
      int level, local_patch;
      //debug5<<"\tGetLevelAndLocalPatchNumber...\n";
      GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

      int matlNo = -1;
      if (matl.compare("*") != 0)
        matlNo = atoi(matl.c_str());

      ParticleDataRaw *pd = NULL;

      //debug5<<"\t(*getParticleData)...\n";
      //todo: this returns an array of doubles. Need to return
      //expected datatype to avoid unnecessary conversion.
      pd = getParticleData2(schedulerP, gridP, level, local_patch,
			    "p.particleID", matlNo, timestate);

      //debug5 << "got particle data: "<<pd<<"\n";
      if (pd)
      {
        vtkDoubleArray *rv = vtkDoubleArray::New();
        //vtkLongArray *rv = vtkLongArray::New();
        //debug5<<"\tSetNumberOfComponents("<<pd->components<<")...\n";
        rv->SetNumberOfComponents(pd->components);

        //debug5<<"\tSetArray...\n";
        rv->SetArray(pd->data, pd->num*pd->components, 0);

        // don't delete pd->data - vtk owns it now!
        delete pd;
        
        //todo: this is the unnecesary conversion, from long
        //int->double->int, to say nothing of the implicit curtailing
        //that might occur (note also: this is a VisIt bug that uses
        //ints to store particle ids rather than long ints)
        vtkIntArray *iv=ConvertToInt(rv);
        //vtkLongArray *iv=ConvertToLong(rv);
        rv->Delete(); // this should now delete pd->data

        pID=iv;
      }

      //debug5<<"read particleID ("<<pID<<")\n";
      if(pID != NULL)
      {
        //debug5<<"adding global node ids from particleID\n";
        pID->SetName("avtGlobalNodeId");
        void_ref_ptr vr =
	  void_ref_ptr( pID , avtVariableCache::DestructVTKObject );

        cache->CacheVoidRef( meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
			     timestate, domain, vr );

        //make sure it worked
        void_ref_ptr vrTmp =
	  cache->GetVoidRef(meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
			    timestate, domain);

        if (*vrTmp == NULL || *vrTmp != *vr)
          throw InvalidFilesException("failed to register uda particle global node");
      }
    }

    return ugrid;
#endif
  }

  // volume data
  else //if (meshName.find("Particle_Mesh") == std::string::npos)
  {
    // make sure we have ghosting info for this mesh
    visit_CalculateDomainNesting( stepInfo,
				  forceMeshReload, timestate, meshname );

    LevelInfo &levelInfo = stepInfo->levelInfo[level];

    //get global bounds
    int glow[3], ghigh[3];
    getBounds(glow, ghigh, meshName, levelInfo);

    //get patch bounds
    int low[3], high[3];
    getBounds(low, high, meshName, levelInfo, local_patch);

    if (meshName.find("NC_") != std::string::npos)
      nodeCentered = true;

    int dims[3], base[3] = {0,0,0};
    for (int i=0; i<3; i++) 
    {
      int offset = 1; // always one for non-node-centered

      if (nodeCentered && high[i] == ghigh[i]) 
        offset = 0; // nodeCentered and patch end is on high boundary

      dims[i] = high[i]-low[i]+offset;
    }

    // debug5 << "Calculating vtkRectilinearGrid mesh for "
    // 	   << meshName << " mesh (" << rgrid << ").\n";

    // vtkRectilinearGrid *rgrid = vtkRectilinearGrid::New();
    // rgrid->SetDimensions(dims);

    // need these to offset grid points in order to preserve face 
    // centered locations on node-centered domain.
    bool sfck[3] = { meshName.find("SFCX") != std::string::npos,
                     meshName.find("SFCY") != std::string::npos,
                     meshName.find("SFCZ") != std::string::npos };

    visit_handle cordH[3] = { VISIT_INVALID_HANDLE,
			      VISIT_INVALID_HANDLE,
			      VISIT_INVALID_HANDLE };

    // Set the coordinates of the grid points in each direction.
    for (int c=0; c<3; c++)
    {
      // vtkFloatArray *coords = vtkFloatArray::New(); 
      // coords->SetNumberOfTuples(dims[c]); 
      // float *array = (float *) coords->GetVoidPointer(0);

      float *array = new float[ dims[c] ];

      if(VisIt_VariableData_alloc(&cordH[c]) == VISIT_OKAY)
      {
	for (int i=0; i<dims[c]; i++)
	{
	  // Face centered data gets shifted towards -inf by half a cell.
	  // Boundary patches are special shifted to preserve global domain.
	  // Internal patches are always just shifted.
	  float face_offset=0;
	  if (sfck[c]) 
	  {
	    if (i==0)
	    {
	      // patch is on low boundary
	      if (low[c]==glow[c])
		face_offset = 0.0;
	      // patch boundary is internal to the domain
	      else
		face_offset = -0.5;
	    }
	    else if (i==dims[c]-1)
	    {
	      // patch is on high boundary
	      if (high[c]==ghigh[c])
	      {
		// periodic means one less value in the face-centered direction
		if (levelInfo.periodic[c])
		  face_offset = 0.0;
		else
		  face_offset = -1;
	      }
	      // patch boundary is internal to the domain
	      else
	      {
		face_offset = -0.5;
	      }
	    }
	    else
	    {
	      face_offset = -0.5;
	    }
	  }

	  array[i] = levelInfo.anchor[c] +
	    (i + low[c] + face_offset) * levelInfo.spacing[c];
	}

	VisIt_VariableData_setDataF(cordH[c], VISIT_OWNER_SIM,
				    1, dims[c], array);


	// switch(c) {
	// case 0:
	//   rgrid->SetXCoordinates(coords); break;
	// case 1:
	//   rgrid->SetYCoordinates(coords); break;
	// case 2:
	//   rgrid->SetZCoordinates(coords); break;
	// }

	// coords->Delete();
      }
    }

    if(VisIt_RectilinearMesh_alloc(&meshH) == VISIT_OKAY)
    {
      /* Fill in the attributes of the RectilinearMesh. */
      VisIt_RectilinearMesh_setCoordsXYZ(meshH, cordH[0], cordH[1], cordH[2]);
      VisIt_RectilinearMesh_setRealIndices(meshH, base, dims);
      VisIt_RectilinearMesh_setBaseIndex(meshH, base);

      // VisIt_RectilinearMesh_setGhostCells(meshH, visit_handle gz);
      // VisIt_RectilinearMesh_setGhostNodes(meshH, visit_handle gn);
    }
  }

  return meshH;
}


//---------------------------------------------------------------------
// visit_SimGetVariable
//     Callback for processing a variable
//---------------------------------------------------------------------
visit_handle visit_SimGetVariable(int domain, const char *varname, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SchedulerP schedulerP = sim->schedulerP;
  GridP      gridP      = sim->gridP;

  bool &useExtraCells   = sim->useExtraCells;
  bool &forceMeshReload = sim->forceMeshReload;
  bool &nodeCentered    = sim->nodeCentered;

  TimeStepInfo* &stepInfo = sim->stepInfo;

  int timestate = sim->cycle;

  visit_handle varH = VISIT_INVALID_HANDLE;

  if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
  {
    std::string varName(varname);
    bool isParticleVar = false;
    
    size_t found = varName.find("/");
    std::string tmpVarName = varName;
    
    std::string matl = varName.substr(found + 1);
    varName = varName.substr(0, found);
    
    std::string varType="CC_Mesh";
    if (strcmp(varname, "proc_id")!=0) {
      for (int k=0; k<(int)stepInfo->varInfo.size(); k++) {
	if (stepInfo->varInfo[k].name == varName) {
	  varType = stepInfo->varInfo[k].type;
	  if (stepInfo->varInfo[k].type.find("ParticleVariable") != std::string::npos) {
	    isParticleVar = true;
	    break;
	  }
	}
      }
    }

    int level, local_patch;
    GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

    // particle data
    if (isParticleVar)
    {
      int matlNo = -1;
      if (matl.compare("*") != 0)
	matlNo = atoi(matl.c_str());
      
      ParticleDataRaw *pd = NULL;
      
#ifdef SERIALIZED_READS
      int numProcs, rank;
      int msg = 128, tag = 256;
      MPI_Status status;
      
      MPI_Comm_size(VISIT_MPI_COMM, &numProcs);
      MPI_Comm_rank(VISIT_MPI_COMM, &rank);
      
      int totalPatches = 0;
      for (int i = 0; i < stepInfo->levelInfo.size(); i++)
	totalPatches += stepInfo->levelInfo[i].patchInfo.size();

      // calculate which process we should wait for a message from
      // if we're processing doiman 0 don't wait for anyone else
      int prev = (rank+numProcs-1)%numProcs;
      int next = (rank+1)%numProcs;
      
      // domain 0 always reads right away
      if (domain==0)
	prev = -1;
      //debug5 << "Proc: " << rank << " sent to GetVar" << std::endl;
      
      // wait for previous read to finish
      if (prev>=0)
	MPI_Recv(&msg, 1, MPI_INT, prev, tag, VISIT_MPI_COMM, &status);
      
      pd = getParticleData2(schedulerP, gridP, level, local_patch, varName,
			    matlNo, timestate);

      // let the next read go
      if (next>=0)
	MPI_Send(&msg, 1, MPI_INT, next, tag, VISIT_MPI_COMM); 
#else
      pd = getParticleData2(schedulerP, gridP, level, local_patch, varName,
			    matlNo, timestate);
#endif
      CheckNaNs(pd->num*pd->components,pd->data,level,local_patch);

      VisIt_VariableData_setDataD(varH, VISIT_OWNER_SIM, pd->components,
       				  pd->num * pd->components, pd->data);

      // vtkDoubleArray *rv = vtkDoubleArray::New();
      // rv->SetNumberOfComponents(pd->components);
      // rv->SetArray(pd->data, pd->num*pd->components, 0);
      
      // ARS - FIX ME - CHECK FOR LEAKS
      // don't delete pd->data - vtk owns it now!
      // delete pd;
    }

    // volume data
    else //if (!isParticleVar)
    {
      LevelInfo &levelInfo = stepInfo->levelInfo[level];
      PatchInfo &patchInfo = levelInfo.patchInfo[local_patch];

      // The region we're going to ask uintah for (from qlow to qhigh-1)      
      int qlow[3], qhigh[3];
      patchInfo.getBounds(qlow,qhigh,varType);
      
      GridDataRaw *gd=NULL;
      
      if (strcmp(varname, "proc_id")==0)
      {
	gd = new GridDataRaw;
	for (int i=0; i<3; i++)
	{
	  gd->low[i ] = qlow[i];
	  gd->high[i] = qhigh[i];
	}
	gd->components = 1;

	int ncells = (qhigh[0]-qlow[0])*(qhigh[1]-qlow[1])*(qhigh[2]-qlow[2]);
	gd->data = new double[ncells];

	for (int i=0; i<ncells; i++) 
	  gd->data[i] = patchInfo.getProcId();
      }
      else
      {
	if (nodeCentered == true)
	{
	  int glow[3], ghigh[3];
	  getBounds(glow,ghigh,varType, levelInfo);
	  patchInfo.getBounds(qlow,qhigh,varType);
	  
	  for (int j=0; j<3; j++)
	  {
	    if (qhigh[j] != ghigh[j]) // patch is on low boundary
	      qhigh[j] = qhigh[j]+1;
	    else
	      qhigh[j] = qhigh[j];
	  }
	}

#ifdef SERIALIZED_READS
	int numProcs, rank;
	int msg = 128, tag = 256;
	MPI_Status status;
	
	MPI_Comm_size(VISIT_MPI_COMM, &numProcs);
	MPI_Comm_rank(VISIT_MPI_COMM, &rank);
	
	int totalPatches = 0;
	for (int i = 0; i < stepInfo->levelInfo.size(); i++)
	  totalPatches += stepInfo->levelInfo[i].patchInfo.size();
	
	// calculate which process we should wait for a message from
	// if we're processing doiman 0 don't wait for anyone else
	int prev = (rank+numProcs-1)%numProcs;
	int next = (rank+1)%numProcs;
	
	// domain 0 always reads right away
	if (domain==0)
	  prev = -1;
	//debug5 << "Proc: " << rank << " sent to GetVar" << std::endl;
	
	// wait for previous read to finish
	if (prev>=0)
	  MPI_Recv(&msg, 1, MPI_INT, prev, tag, VISIT_MPI_COMM, &status);

	gd = getGridData2(schedulerP, gridP, level, local_patch, varName,
			  atoi(matl.c_str()), timestate, qlow, qhigh);

	// let the next read go
	if (next>=0)
	  MPI_Send(&msg, 1, MPI_INT, next, tag, VISIT_MPI_COMM);
#else
	gd = getGridData2(schedulerP, gridP, level, local_patch, varName,
			  atoi(matl.c_str()), timestate, qlow, qhigh);
#endif
      }

      if( gd )
      {
	int n = (qhigh[0]-qlow[0])*(qhigh[1]-qlow[1])*(qhigh[2]-qlow[2]);
      
	CheckNaNs(n*gd->components,gd->data,level,local_patch);
	
	VisIt_VariableData_setDataD(varH, VISIT_OWNER_SIM, gd->components,
				    n * gd->components, gd->data);      

	// vtkDoubleArray *rv = vtkDoubleArray::New();
	// rv->SetNumberOfComponents(gd->components);
	// rv->SetArray(gd->data, n*gd->components, 0);
	
	// ARS - FIX ME - CHECK FOR LEAKS
	// don't delete gd->data - vtk owns it now!
	// delete gd;
      }
    }
  }

  return varH;
}

} // End namespace Uintah

