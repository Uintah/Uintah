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

#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/Schedulers/UnifiedScheduler.h>
#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/VarLabel.h>


#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/SysUtils.h>

#include <VisIt/interfaces/warehouseInterface.h>

#include <stdio.h>

namespace Uintah {

void nameCleanup( std::string &str )
{
  size_t found;
  while( (found = str.find_first_of(":")) != std::string::npos)
  {
    str[found] = '_';
  }  
  // while( (found = str.find_first_of(" ")) != std::string::npos)
  // {
  //   str[found] = '_';
  // }  
  // while( (found = str.find_first_of("(")) != std::string::npos)
  // {
  //   str[found] = '_';
  // }  
  // while( (found = str.find_first_of(")")) != std::string::npos)
  // {
  //   str[found] = '_';
  // }  
}

// ****************************************************************************
//  Method: visit_SimGetCustomUIData
//
//  Purpose:
//      Callback for processing meta data
//
// ****************************************************************************
void visit_SimGetCustomUIData(void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  // Clear the strip chart
  VisItUI_setValueS("STRIP_CHART_CLEAR_MENU", "NoOp", 0);

  // Set the custom UI time values.
  visit_SetTimeValues( sim );
    
  // Set the custom UI delta t values.
  visit_SetDeltaTValues( sim );
    
  // Set the custom UI wall time values.
  visit_SetWallTimes( sim );
    
  // Set the custom UI optional UPS variable table
  visit_SetUPSVars( sim );

  // Set the custom UI reduction variable table
  visit_SetReductionVariables( sim );

  // Set the custom UI output variable table
  visit_SetOutputIntervals( sim );

  // Set the custom UI optional min/max variable table
  visit_SetAnalysisVars( sim );

  // Set the custom UI Grid Info
  visit_SetGridInfo( sim );

  // Set the custom UI Runtime Stats
  visit_SetRuntimeStats( sim );

  // Set the custom UI MPI Stats
  visit_SetMPIStats( sim );

  // Set the custom UI Application Stats
  visit_SetApplicationStats( sim );

  // Setup the custom UI Image variables
  visit_SetImageVars( sim );

  // Set the custom UI optional state variable table
  visit_SetStateVars( sim );

  // Set the custom UI debug stream table
  visit_SetDebugStreams( sim );

  // Set the custom UI debug stream table
  visit_SetDouts( sim );

  // Set the custom UI database behavior
  visit_SetDatabase( sim );

  // Set the custom UI variable loading behavior
  visit_SetVariables( sim );
}


// ****************************************************************************
//  Method: visit_SimGetMetaData
//
//  Purpose:
//      Callback for processing meta data
//
// ****************************************************************************
visit_handle visit_SimGetMetaData(void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
    (sim->simController->getSchedulerP().get_rep());
      
  UnifiedScheduler *unifiedScheduler = dynamic_cast<UnifiedScheduler*>
    (sim->simController->getSchedulerP().get_rep());
      
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();
          
  SchedulerP     schedulerP = sim->simController->getSchedulerP();
  Output       * output     = sim->simController->getOutput();
  GridP          gridP      = sim->gridP;
      
  if( !schedulerP.get_rep() || !gridP.get_rep() )
  {
    return VISIT_INVALID_HANDLE;
  }

  if( sim->stepInfo )
    delete sim->stepInfo;
  
  sim->stepInfo = getTimeStepInfo(schedulerP, gridP,
                                  sim->loadExtraGeometry, sim->loadVariables);

  unsigned int addMachineData = (sim->switchNodeList.size() &&
                                 (int) sim->switchIndex != -1 &&
                                 (int) sim->nodeIndex   != -1 );

  TimeStepInfo* &stepInfo = sim->stepInfo;
  
  visit_handle md = VISIT_INVALID_HANDLE;

  // Create metadata with no variables.
  if(VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY)
  {
    // Set the simulation state.

    // NOTE visit_SimGetMetaData is called as a results of calling
    // visit_CheckState which calls VisItTimeStepChanged at this point
    // the sim->runMode will always be VISIT_SIMMODE_RUNNING.

    // To get the "Simulation status" in the Simulation window correct
    // one needs to call VisItUI_setValueS("SIMULATION_MODE", "Stopped", 1);
    if(sim->runMode == VISIT_SIMMODE_FINISHED ||
       sim->runMode == VISIT_SIMMODE_STOPPED ||
       sim->runMode == VISIT_SIMMODE_STEP)
      VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_STOPPED);
    else if(sim->runMode == VISIT_SIMMODE_RUNNING)
      VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_RUNNING);

    VisIt_SimulationMetaData_setCycleTime(md, sim->cycle, sim->time);

    // Don't add node data unless NC_Mesh exists (some only have
    // CC_MESH or SFCk_MESH)
    bool addNodeData = false;
  
    // Don't add patch data unless CC_Mesh or NC_Mesh exists (some only
    // have SFCk_MESH)
    bool addPatchData = false;

    // grid meshes are shared between materials, and particle meshes are
    // shared between variables - keep track of what has been added so
    // they're only added once
    std::set<std::string> meshes_added;

    // If a variable exists in multiple materials, we don't want to add
    // it more than once to the meta data - it can mess up visit's
    // expressions variable lists.
    std::set<std::string> mesh_vars_added;

    int numVars = stepInfo->varInfo.size();

    // Do a hasty search for a node or cell mesh.
    for (int i=0; i<numVars; ++i)
    {
      if (stepInfo->varInfo[i].type.find("NC") != std::string::npos)
      {
        addNodeData = true;
        addPatchData = true;
      }
      else if (stepInfo->varInfo[i].type.find("CC") != std::string::npos)
      {
          addPatchData = true;
      }
    }

    // Loop through all vars and add them to the meta data.
    for (int i=0; i<numVars; ++i)
    {
      // Particle variable - nothing needs to be modifed for particle
      // data, as they exist only on a single level
      if (stepInfo->varInfo[i].type.find("ParticleVariable") != std::string::npos)
      {
        std::string varname = stepInfo->varInfo[i].name;
        std::string vartype = stepInfo->varInfo[i].type;
        
        if (vartype.find("filePointer") != std::string::npos)
        {
          continue;
        }
        
        VisIt_VarCentering cent = VISIT_VARCENTERING_NODE;

        // j = -1 -> all materials (*)
        int numMaterials = stepInfo->varInfo[i].materials.size();

        for (int j=-1; j<numMaterials; ++j)
        {
          std::string mesh_for_this_var = std::string("Particle_Mesh/");
          std::string newVarname = varname + "/";
          
          if (j == -1)
          {
            mesh_for_this_var.append("*");
            newVarname.append("*");
          }
          else
          {
            char buffer[128];
            sprintf(buffer, "%d", stepInfo->varInfo[i].materials[j]);
            mesh_for_this_var.append(buffer);
            newVarname.append(buffer);
          }

          addParticleMesh( md, meshes_added, mesh_for_this_var, sim);

          addMeshVariable( md, mesh_vars_added,
                           newVarname, vartype, mesh_for_this_var, cent );
        }
      }   

      // Grid variables
      else
      {
        std::string varname = stepInfo->varInfo[i].name;
        std::string vartype = stepInfo->varInfo[i].type;

        // from the variable type get the mesh needed
        std::string mesh_for_this_var;
        VisIt_VarCentering cent;
        
        bool isPerPatchVar = false;

        if (vartype.find("NC") != std::string::npos)
        {
          mesh_for_this_var.assign("NC_Mesh");
          cent = VISIT_VARCENTERING_NODE;
        }
        else if (vartype.find("CC") != std::string::npos)
        {
          mesh_for_this_var.assign("CC_Mesh");
          cent = VISIT_VARCENTERING_ZONE;
        }
        else if (vartype.find("SFC") != std::string::npos)
        { 
          if (vartype.find("SFCX") != std::string::npos)               
            mesh_for_this_var.assign("SFCX_Mesh");
          else if (vartype.find("SFCY") != std::string::npos)          
            mesh_for_this_var.assign("SFCY_Mesh");
          else if (vartype.find("SFCZ") != std::string::npos)          
            mesh_for_this_var.assign("SFCZ_Mesh");

          cent = VISIT_VARCENTERING_ZONE;
        }
        else if (vartype.find("PerPatch") != std::string::npos)
        {
          if (varname.find("FileInfo") == 0 ||
              varname.find("CellInformation") == 0 ||
              varname.find("CutCellInfo") == 0)
            continue;
          
          mesh_for_this_var.assign("Patch_Mesh");
          cent = VISIT_VARCENTERING_ZONE;

          isPerPatchVar = true;
        }
        else if (vartype.find("ReductionVariable") != std::string::npos ||
                 vartype.find("SoleVariable")      != std::string::npos)
        {
          continue;
        }
        else
        {
          if(sim->isProc0)
          {
            std::stringstream msg;
            msg << "Visit libsim - "
                << "Uintah variable \"" << varname << "\"  "
                << "has an unknown grid variable type \""
                << vartype << "\"";
            
            VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
          }

          continue;
        }

        // Add the mesh for this variable.
        addRectilinearMesh( md, meshes_added, mesh_for_this_var, sim);
        
        // If the variable is a PerPatch var and there is a machine
        // layout then the it can be placed on both the simulation
        // patch mesh and the machine patch mesh.
        std::string mesh_name[2] = {mesh_for_this_var,
                                    ("Machine_" + sim->hostName + "/Patch") };

        std::string mesh_layout[2] = {"/Sim", "/"+sim->hostName};

        for( int k=0; k<1+int(isPerPatchVar && addMachineData); ++k )
        {
          // Add mesh vars foreach material. If no materials loop just once.
          int numMaterials = stepInfo->varInfo[i].materials.size();
          
          for (int j=0; j<std::max(1,numMaterials); ++j)
          {
            std::string newVarname = varname;
            
            // PerPatch vars do not have a material index.
            if( isPerPatchVar )
            {
              newVarname = "Patch/" + newVarname;

              // If the is machine data add a intermediate menu level.
              if( addMachineData )
                newVarname += mesh_layout[k];
            }
            // Add the material index.
            else if( numMaterials )
            {
              char buffer[128];
              sprintf(buffer, "%d", stepInfo->varInfo[i].materials[j]);
              newVarname.append("/");
              newVarname.append(buffer);
            }
            // For variables with no materials add a material index of 0.
            else
              newVarname.append("/0");
            
            addMeshVariable( md, mesh_vars_added,
                             newVarname, vartype, mesh_name[k], cent );
          }
        }
      }   
    }

    // Ancillary grid data.
    
    // Add the node data (e.g. node id's)
    if (addNodeData)
    {
      visit_handle vmd = VISIT_INVALID_HANDLE;

      if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
      {
        std::string varname = "Patch/Nodes";

        VisIt_VariableMetaData_setName(vmd, varname.c_str() );
        VisIt_VariableMetaData_setMeshName(vmd, "NC_Mesh");
        VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_NODE);
        VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
        VisIt_VariableMetaData_setNumComponents(vmd, 3);
        VisIt_VariableMetaData_setUnits(vmd, "");
            
        // ARS - FIXME
        //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
        VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
        VisIt_SimulationMetaData_addVariable(md, vmd);
      }
    }

    // Add the patch data
    if (addPatchData)
    {
      std::string mesh_for_this_var = "Patch_Mesh";

      addRectilinearMesh( md, meshes_added, mesh_for_this_var, sim);
      
      // Per rank and per node ids on the simulation patch mesh and
      // possibly machine mesh.
      visit_handle vmd = VISIT_INVALID_HANDLE;

      int cent = VISIT_VARCENTERING_ZONE;

      // Bounds for node and cell based patch variables on just the
      // simulation mesh.
      for (std::set<std::string>::iterator it=meshes_added.begin();
           it!=meshes_added.end(); ++it)
      {
        if ( (*it).find("NC") != std::string::npos ||
             (*it).find("CC") != std::string::npos )
        {
          if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
          {
            std::string varname = "Patch/Bounds/Low/" + *it;

            VisIt_VariableMetaData_setName(vmd, varname.c_str() );
            VisIt_VariableMetaData_setMeshName(vmd, mesh_for_this_var.c_str());
            VisIt_VariableMetaData_setCentering(vmd, cent);
            VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
            VisIt_VariableMetaData_setNumComponents(vmd, 3);
            VisIt_VariableMetaData_setUnits(vmd, "");
            
            // ARS - FIXME
            //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
            VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
            VisIt_SimulationMetaData_addVariable(md, vmd);
          }
          
          if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
          {
            std::string varname = "Patch/Bounds/High/" + *it;
            
            VisIt_VariableMetaData_setName(vmd, varname.c_str() );
            VisIt_VariableMetaData_setMeshName(vmd, mesh_for_this_var.c_str());
            VisIt_VariableMetaData_setCentering(vmd, cent);
            VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
            VisIt_VariableMetaData_setNumComponents(vmd, 3);
            VisIt_VariableMetaData_setUnits(vmd, "");
            
            // ARS - FIXME
            //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
            VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
            VisIt_SimulationMetaData_addVariable(md, vmd);
          }
        }
      }

      // Patch id goes on both the simulation patch mesh and the
      // machine patch mesh. The patch rank and node goes on just the
      // simulation patch mesh
      std::string mesh_name[2] = {mesh_for_this_var,
                                  ("Machine_" + sim->hostName + "/Patch")};

      std::string mesh_layout[2] = {"/Sim", "/"+sim->hostName};

      const char *patch_names[3] = {"Patch/Id", "Patch/Rank", "Patch/Node" };

      for( unsigned k=0; k<1+addMachineData; ++k )
      {
        for( unsigned int i=0; i<3; ++i )
        {
          if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
          {
            std::string tmp_name = patch_names[i];

            // Only put the patch id goes to both the simulation and
            // machine patch mesh.
            if( i == 0 && addMachineData )
              tmp_name += mesh_layout[k];

            VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
            VisIt_VariableMetaData_setMeshName(vmd, mesh_name[k].c_str());
            VisIt_VariableMetaData_setCentering(vmd, cent);
            VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
            VisIt_VariableMetaData_setNumComponents(vmd, 1);
            VisIt_VariableMetaData_setUnits(vmd, "");
            
            // ARS - FIXME
            //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
            VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
            VisIt_SimulationMetaData_addVariable(md, vmd);
          }

          // Only put the patch id on the machine patch mesh.
          if( k == 1 )
            break;
        }
      }

      // Per rank and per node performance data on the simulation
      // patch mesh and possibly machine rank mesh.
      mesh_name[1] = "Machine_" + sim->hostName + "/Local";
      
      // Runtime data on both the sim and machine.
      for( unsigned k=0; k<1+addMachineData; ++k )
      {
        // Add in the infrastructure runtime stats.
        addReductionStats( md, sim->simController->getRuntimeStats(),
                           "Processor/Runtime/", mesh_name[k],
                           (addMachineData ? mesh_layout[k] : "") );
          
        // Add in the mpi runtime stats.
        if( mpiScheduler ) {
          addReductionStats( md, mpiScheduler->m_mpi_info,
                             "Processor/MPI/", mesh_name[k],
                             (addMachineData ? mesh_layout[k] : "") );
        }
        
        // Add in the application runtime stats.
        addReductionStats( md, appInterface->getApplicationStats(),
                           "Processor/Application/", mesh_name[k],
                           (addMachineData ? mesh_layout[k] : "") );
      }
    }

    // If there is a machine layout then the performance data can be
    // placed on the machine mesh.
    if( addMachineData )
    {
      // Number of additioanl threads.
      unsigned int addThreads = (Uintah::Parallel::getNumThreads() - 1 > 1);
      unsigned int addComms   = (sim->myworld->nRanks() > 1);

      // If there is a machine layout then there is a global, local,
      // and patch machine mesh. The global is all of the nodes and
      // cores. The local is the nodes and cores actually used. The
      // patch is patches on the each core.
      for( unsigned int i=0; i<6; ++i )
      {
        // Set the mesh’s properties.
        std::string meshName = "Machine_" + sim->hostName;

        unsigned int nLoops = 1;
        
        if( i == 0 && sim->switchNodeList.size() > 0 ) { // Global rank mesh
          meshName += "/Global";
        }
        else if( i == 1 ) { // Local rank mesh
          meshName += "/Local";
        }
        else if( i == 2 ) { // Local patch mesh
          meshName += "/Patch";
        }
        else if( i == 3 && addThreads ) { // Local thread mesh
          meshName += "/Thread";
        }
        else if( i == 4 && addComms ) { // Local communication mesh
            meshName += "/Communication";
            nLoops = mpiScheduler->getNumTaskGraphs();
        }
        else if( i == 5 && mpiScheduler ) { // Local thread mesh
          meshName += "/Tasks";
        }
        else
          continue;

        for( unsigned int j=0; j<nLoops; ++j )
        {
          visit_handle mmd = VISIT_INVALID_HANDLE;

          // Set the first mesh’s properties.
          if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
          {
            // There may be multiple communication matrices, the index
            // will be last.
            if( i == 4 && addComms && nLoops > 1 ) {
              VisIt_MeshMetaData_setName(mmd, (meshName + "/" + std::to_string(j)).c_str());
            }
            else {
              VisIt_MeshMetaData_setName(mmd, meshName.c_str());
            }
            VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_UNSTRUCTURED);
            VisIt_MeshMetaData_setTopologicalDimension(mmd, 2);
            VisIt_MeshMetaData_setSpatialDimension(mmd, 2);
            VisIt_MeshMetaData_setXLabel(mmd, "Switches");
            VisIt_MeshMetaData_setYLabel(mmd, "Nodes");

            // For the global view there is only one domain. For the
            // local, patch, and thread view there is one domain per rank.
            VisIt_MeshMetaData_setNumDomains(mmd,
                                             (i == 0 ? 1 : sim->myworld->nRanks()));

            if( i > 0 )
            {
              VisIt_MeshMetaData_setDomainTitle(mmd, "ranks");
              VisIt_MeshMetaData_setDomainPieceName(mmd, "rank");
              
              VisIt_MeshMetaData_setNumGroups(mmd, sim->myworld->nNodes());
              VisIt_MeshMetaData_setGroupTitle(mmd, "nodes");
              VisIt_MeshMetaData_setGroupPieceName(mmd, sim->hostName.c_str());
              
              for (int node=0; node<sim->myworld->nNodes(); ++node)
              {
                VisIt_MeshMetaData_addGroupName(mmd, sim->myworld->getNodeName( node ).c_str());
              }
              
              for (int rank=0; rank<sim->myworld->nRanks(); ++rank)
              {
                int node = sim->myworld->getNodeIndexFromRank( rank );
                char tmpName[MPI_MAX_PROCESSOR_NAME+24];
                sprintf(tmpName,"%s, rank%d", sim->myworld->getNodeName( node ).c_str(), rank);
                
                VisIt_MeshMetaData_addGroupId(mmd, node);
                VisIt_MeshMetaData_addDomainName(mmd, tmpName);
              }
            }
            
            // ARS - FIXME
            // VisIt_MeshMetaData_setContainsExteriorBoundaryGhosts(mmd, false);
            
            VisIt_MeshMetaData_setHasSpatialExtents(mmd, 1);
            
            double extents[6] =
              { 0, double(sim->switchNodeList.size() * (sim->xNode+1) - 1),
                0, double(sim->maxNodes              * (sim->yNode+1) - 1),
                0, 0 };
            
            VisIt_MeshMetaData_setSpatialExtents(mmd, extents);
            
            // ARS - FIXME - NOT SURE SHOULD BE HERE
            // VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
            // VisIt_MeshMetaData_setLogicalBounds(mmd, logical);
        
            VisIt_SimulationMetaData_addMesh(md, mmd);
          }
        }
      }


      if( addComms ) { // Communication  mesh

        // There may be multiple communication matrices, the index
        // will be last.
        unsigned int nTaskGraphs = mpiScheduler->getNumTaskGraphs();
        
        for( unsigned int k=0; k<nTaskGraphs; ++k )
        {
          // Set the mesh’s properties.
          std::string meshName = "Communication_" + sim->hostName;

          if( nTaskGraphs > 1 )
            meshName += "/" + std::to_string(k);

          visit_handle mmd = VISIT_INVALID_HANDLE;
        
          // Set the first mesh’s properties.
          if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
          {
            VisIt_MeshMetaData_setName(mmd, meshName.c_str());
            VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_POINT);
            VisIt_MeshMetaData_setTopologicalDimension(mmd, 0);
            VisIt_MeshMetaData_setSpatialDimension(mmd, 3);
            VisIt_MeshMetaData_setXLabel(mmd, "Rank");
            VisIt_MeshMetaData_setYLabel(mmd, "Rank");

            // For the communication view there is one domain per rank.
            VisIt_MeshMetaData_setNumDomains(mmd, sim->myworld->nRanks());

            VisIt_MeshMetaData_setDomainTitle(mmd, "ranks");
            VisIt_MeshMetaData_setDomainPieceName(mmd, "rank");
          
            VisIt_MeshMetaData_setNumGroups(mmd, sim->myworld->nNodes());
            VisIt_MeshMetaData_setGroupTitle(mmd, "nodes");
            VisIt_MeshMetaData_setGroupPieceName(mmd, sim->hostName.c_str());
          
            for (int node=0; node<sim->myworld->nNodes(); ++node)
            {
              VisIt_MeshMetaData_addGroupName(mmd, sim->myworld->getNodeName( node ).c_str());
            }

            for (int rank=0; rank<sim->myworld->nRanks(); ++rank)
            {
              int node = sim->myworld->getNodeIndexFromRank( rank );
              char tmpName[MPI_MAX_PROCESSOR_NAME+24];
              sprintf(tmpName,"%s, rank%d", sim->myworld->getNodeName( node ).c_str(), rank);
            
              VisIt_MeshMetaData_addGroupId(mmd, node);
              VisIt_MeshMetaData_addDomainName(mmd, tmpName);
            }

            // ARS - FIXME
            // VisIt_MeshMetaData_setContainsExteriorBoundaryGhosts(mmd, false);

            VisIt_MeshMetaData_setHasSpatialExtents(mmd, 1);
        
            double extents[6] = { 0, double(sim->myworld->nRanks()-1),
                                  0, double(sim->myworld->nRanks()-1),
                                  0, 0 };

            VisIt_MeshMetaData_setSpatialExtents(mmd, extents);

            int logical[3] = { sim->myworld->nRanks(), 0, 0 };

            VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
            VisIt_MeshMetaData_setLogicalBounds(mmd, logical);
        
            VisIt_SimulationMetaData_addMesh(md, mmd);
          }
        }
      }

      // Tasks
      if( mpiScheduler )
      {
        // Add in the tasks runtime stats which go on to the
        // machine local and patch meshes.
        std::string varName[2] = { "Processor/Machine/Tasks/",
                                   "Patch/Tasks/" };
        
        std::string meshName[2] = { "Machine_" + sim->hostName + "/Local",
                                    "Patch_Mesh" };

        for( unsigned int j=0; j<2; ++j )
        {
          addMapIndividualStats( md, mpiScheduler->m_task_info,
                                 varName[j], "",
                                 meshName[j] );
        }

        varName[0] = "Processor/Machine/Tasks/All/";
        meshName[0] = "Machine_" + sim->hostName + "/Tasks";
        
        addMapAllStats( md, mpiScheduler->m_task_info,
                        varName[0], "",
                        meshName[0] );
      }
      
      const int nVars = 4;
      std::string vars[nVars] = {"NodeID",
                                 "MPI/Node",
                                 "MPI/Rank",
                                 "MPI/Comm/Rank"};

      std::string meshName = "Machine_" + sim->hostName + "/Local";
      
      for( unsigned int i=0; i<nVars; ++i )
      {
        visit_handle vmd = VISIT_INVALID_HANDLE;

        if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
        {
          std::string var = std::string("Processor/Machine/") + vars[i];

          VisIt_VariableMetaData_setName(vmd, var.c_str());
          VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
          VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
          VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
          VisIt_VariableMetaData_setNumComponents(vmd, 1);

          if( vars[i] ==  "NodeID" )
            VisIt_VariableMetaData_setUnits(vmd, sim->hostName.c_str());
          else
            VisIt_VariableMetaData_setUnits(vmd, "");

          // ARS - FIXME
          // VisIt_VariableMetaData_setHasDataExtents(vmd, false);
          VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
          VisIt_SimulationMetaData_addVariable(md, vmd);
        }
      }

      // Additional threads.
      if( unifiedScheduler && addThreads )
      {
        // Add in the unified thread reduction runtime stats which go
        // on to the local machine mesh (i.e. the rank).
        addVectorReductionStats( md, unifiedScheduler->m_thread_info,
                                 "Processor/Machine/Thread/",
                                 "Machine_" + sim->hostName + "/Local" );

        // Add in the unified thread runtime stats which go on the
        // thread mesh.
        std::string meshName = "Machine_" + sim->hostName + "/Thread";
        
        addVectorStats( md, unifiedScheduler->m_thread_info,
                        "Processor/Machine/Thread/", meshName );
      }

      // Additional comms.
      if( mpiScheduler && addComms )
      {
        // There may be multiple communication matrices, the index
        // will be last.
        unsigned int nTaskGraphs = mpiScheduler->getNumTaskGraphs();

        for( unsigned int k=0; k<nTaskGraphs; ++k )
        {
          std::string statIndex = (nTaskGraphs == 1 ? "" :
                                   "/" + std::to_string(k));

          // Add in the communication runtime stats which go on to the
          // machine communication and communication meshes.
          std::string varName[2] = { "Processor/Machine/Communication/",
                                     "Processor/Communication/" };
          std::string meshName[2] = { "Machine_" + sim->hostName + "/Communication",
                                      "Communication_" + sim->hostName };

          // Add in the communication across each task pairs.
          for (auto& info: mpiScheduler->getTaskGraph(k)->getDetailedTasks()->getCommInfo()) {
            std::pair< std::string, std::string > taskPair = info.first;
    
            std::string taskPairName = taskPair.first + "|" + taskPair.second;

            // Comment out this code so to get all tasks.
            if( taskPairName != "All|Tasks" )
              continue;

            // Add in the communication reduction runtime stats which go
            // on to the application mesh.
            addMapReductionStats( md, info.second,
                                  "Patch/Communication/" + taskPairName + "/", statIndex,
                                  "Patch_Mesh");

            // Add in the communication reduction runtime stats which go
            // on to the local machine mesh (i.e. the rank).
            addMapReductionStats( md, info.second,
                                  "Processor/Machine/Communication/" + taskPairName + "/", statIndex,
                                  "Machine_" + sim->hostName + "/Local" );

            for( unsigned int j=0; j<2; ++j )
            {
              addMapAllStats( md, info.second,
                              varName[j] + taskPairName + "/", statIndex,
                              meshName[j] + statIndex );
            }
          }

          // Add in the communication across all tasks.
          
          // Add in the communication reduction runtime stats which go
          // on to the application mesh.
          // addMapReductionStats( md, mpiScheduler->getTaskGraph(k)->getDetailedTasks()->getCommInfo(),
          //                       "Patch/Communication/AllTasks/", statIndex,
          //                       "Patch_Mesh");

          // Add in the communication reduction runtime stats which go
          // on to the local machine mesh (i.e. the rank).
          // addMapReductionStats( md, mpiScheduler->getTaskGraph(k)->getDetailedTasks()->getCommInfo(),
          //                       "Processor/Machine/Communication/AllTasks/", statIndex,
          //                       "Machine_" + sim->hostName + "/Local" );

          for( unsigned int j=0; j<2; ++j )
          {
            // addMapAllStats( md, mpiScheduler->getTaskGraph(k)->getDetailedTasks()->getCommInfo(),
            //              varName[j] + "AllTasks/", statIndex,
            //              meshName[j] + statIndex );

            const int nVars = 1;
            std::string vars[nVars] = { "CommID" };
            
            for( unsigned int i=0; i<nVars; ++i )
            {
              visit_handle vmd = VISIT_INVALID_HANDLE;
              
              if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
              {
                std::string var = varName[j] + vars[i] + statIndex;

                VisIt_VariableMetaData_setName(vmd, var.c_str());
                VisIt_VariableMetaData_setMeshName(vmd, (meshName[j] + statIndex).c_str());
                VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
                VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                VisIt_VariableMetaData_setNumComponents(vmd, 1);
                VisIt_VariableMetaData_setUnits(vmd, "");
                
                // ARS - FIXME
                // VisIt_VariableMetaData_setHasDataExtents(vmd, false);
                VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
                VisIt_SimulationMetaData_addVariable(md, vmd);
              }
            }
          }
        }
      }
    }
    
    // ARS - FIXME
    // int numLevels = stepInfo->levelInfo.size();

    // int totalPatches = 0;
    // for (int i=0; i<numLevels; ++i)
    //   totalPatches += stepInfo->levelInfo[i].patchInfo.size();
    
    // // Used the domain partitioning - the pieceNames are not needed.
    // std::vector<int> groupIds(totalPatches);

    // for (int i = 0; i < totalPatches; i++)
    // {
    //   int level, local_patch;
      
    //   GetLevelAndLocalPatchNumber(stepInfo, i, level, local_patch);
      
    //   groupIds[i] = level;
    // }
    
    // md->AddGroupInformation(numLevels, totalPatches, groupIds);
    // md->AddDefaultSILRestrictionDescription(std::string("!TurnOnAll"));

    
    // Add some commands.
    const char *cmd_names[] = {"Stop", "Step", "Run",
                               "Output", "Output Previous", "Terminate",
                               "Checkpoint", "Checkpoint Previous", "Abort"};

    unsigned int numNames = sizeof(cmd_names) / sizeof(const char *);

    for (unsigned int i=0; i<numNames; ++i)
    {
      bool enabled;

      // Do not output or checkpoint if has already happened.
      if(strcmp( "Output", cmd_names[i] ) == 0 )
        enabled = (!output->outputTimeStepExists( sim->cycle ));
      else if(strcmp( "Checkpoint", cmd_names[i] ) == 0 )
        enabled = (!output->checkpointTimeStepExists( sim->cycle ));
      // Do not allow the previous time step to outputed or
      // checkpointed if the current step was regridded as the grid
      // will have changed and the patches will align with the data.

      // Do not allow the previous time step to outputed or
      // checkpointed if the current time step was already outputed or
      // checkpointed already so to prevent out of order time steps.
      else if(strcmp( "Output Previous", cmd_names[i] ) == 0 )
        enabled = (appInterface->getLastRegridTimeStep() < sim->cycle &&
                   !output->outputTimeStepExists( sim->cycle ) &&
                   !output->outputTimeStepExists( sim->cycle-1));
      else if(strcmp( "Checkpoint Previous", cmd_names[i] ) == 0 )
        enabled = (appInterface->getLastRegridTimeStep() < sim->cycle &&
                   !output->checkpointTimeStepExists( sim->cycle ) &&
                   !output->checkpointTimeStepExists( sim->cycle-1));
      else
        enabled = true;

      visit_handle cmd = VISIT_INVALID_HANDLE;
      
      if(VisIt_CommandMetaData_alloc(&cmd) == VISIT_OKAY)
      {
          VisIt_CommandMetaData_setName(cmd, cmd_names[i]);
          VisIt_CommandMetaData_setEnabled(cmd, enabled);
          VisIt_SimulationMetaData_addGenericCommand(md, cmd);
      }
    }
    // if( sim->message.size() )
    // {
    //   visit_handle msg = VISIT_INVALID_HANDLE;
      
    //   if(VisIt_MessageMetaData_alloc(&msg) == VISIT_OKAY)
    //   {
    //  VisIt_MessageMetaData_setName(msg, sim->message.c_str());
    //  VisIt_SimulationMetaData_addMessage(md, msg);
    //   }
    // }

    visit_SimGetCustomUIData(cbdata);
  }

  return md;
}


// ****************************************************************************
//  Method: visit_GetDomainBoundaries
//
//  Purpose:
//      Calculates two important data structures.  One is the structure domain
//      nesting, which tells VisIt how the patches are nested, which allows
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
visit_handle visit_SimGetDomainBoundaries(const char *name, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  TimeStepInfo* &stepInfo = sim->stepInfo;
    
  const std::string meshname(name); 

  //
  // Calculate some info we will need in the rest of the routine.
  //
  int numLevels = stepInfo->levelInfo.size();
  int totalPatches = 0;

  for (int level=0; level<numLevels; ++level)
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
    
    for (int patch=0; patch<totalPatches; ++patch)
    {
      int my_level, local_patch;
      GetLevelAndLocalPatchNumber(stepInfo, patch, my_level, local_patch);
        
      PatchInfo &patchInfo =
        stepInfo->levelInfo[my_level].patchInfo[local_patch];

      int plow[3], phigh[3];
      patchInfo.getBounds(plow, phigh, meshname);

      // For node based meshes add one if there is a neighbor patch.
      if( meshname.find("NC_") == 0 )
      {
        int nlow[3], nhigh[3];
        patchInfo.getBounds(nlow, nhigh, "NEIGHBORS");
        
        for (int i=0; i<3; i++)
          phigh[i] += nhigh[i];
      }

      // These are indices, the high values are exclusive.
      int extents[6] = { plow[0], phigh[0],
                         plow[1], phigh[1],
                         plow[2], phigh[2] };

      VisIt_DomainBoundaries_set_amrIndices(rdb, patch, my_level, extents);
      // VisIt_DomainBoundaries_finish(rdb, patch);

      // std::cerr << "\trdb->SetIndicesForPatch(" << patch << ","
      //           << my_level << ", " << local_patch << ", <"
      //           << extents[0] << "," << extents[2] << "," << extents[4]
      //           << "> to <"
      //           << extents[1] << "," << extents[3] << "," << extents[5] << ">)"
      //           << std::endl;
    }

    return rdb;
  }
  else
    return VISIT_INVALID_HANDLE;
}

// ****************************************************************************
//  Method: visit_GetDomainNesting
//
//  Purpose:
//      Calculates two important data structures.  One is the structure domain
//      nesting, which tells VisIt how the  patches are nested, which allows
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
visit_handle visit_SimGetDomainNesting(const char *name, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  TimeStepInfo* &stepInfo = sim->stepInfo;
    
  const std::string meshname(name); 

  //
  // Calculate some info we will need in the rest of the routine.
  //
  int numLevels = stepInfo->levelInfo.size();
  int totalPatches = 0;

  for (int level=0; level<numLevels; ++level)
    totalPatches += stepInfo->levelInfo[level].patchInfo.size();

  //
  // Domain Nesting
  //
  visit_handle dn;
    
  if(VisIt_DomainNesting_alloc(&dn) == VISIT_OKAY)
  {
    VisIt_DomainNesting_set_dimensions(dn, totalPatches, numLevels, 3);

    //
    // Calculate what the refinement ratio is from one level to the next.
    //
    for (int level=0; level<numLevels; ++level)
    {
      // SetLevelRefinementRatios requires data as a vector<int>
      int rr[3];

      for (int i=0; i<3; ++i)
        rr[i] = stepInfo->levelInfo[level].refinementRatio[i];
        
      VisIt_DomainNesting_set_levelRefinement(dn, level, rr);

      // std::cerr << "\tdn->SetLevelRefinementRatios(" << level << ", <"
      //                << rr[0] << "," << rr[1] << "," << rr[2] << ">)\n";
    }      

    // Calculating the child patches really needs some better sorting
    // than what is crrently being done.  This is likely to become a
    // bottleneck in extreme cases.  Although this routine has
    // performed well for a previous 55K patch run.
    std::vector< std::vector<int> > childPatches(totalPatches);
      
    for (int level = numLevels-1; level>0; level--)
    {
      int prev_level = level-1;
      LevelInfo &levelInfoParent = stepInfo->levelInfo[prev_level];
      LevelInfo &levelInfoChild = stepInfo->levelInfo[level];
      
      for (int child=0; child<(int)levelInfoChild.patchInfo.size(); ++child)
      {
        PatchInfo &childPatchInfo = levelInfoChild.patchInfo[child];

        int child_low[3], child_high[3];
        childPatchInfo.getBounds(child_low, child_high, meshname);
          
        // For node based meshes add one if there is a neighbor patch.
        if( meshname.find("NC_") == 0 )
        {
          int nlow[3], nhigh[3];
          childPatchInfo.getBounds(nlow, nhigh, "NEIGHBORS");
          
          for (int i=0; i<3; i++)
            child_high[i] += nhigh[i];
        }
        
        for (int parent = 0;
             parent<(int)levelInfoParent.patchInfo.size(); ++parent)
        {
          PatchInfo &parentPatchInfo = levelInfoParent.patchInfo[parent];
          
          int parent_low[3], parent_high[3];
          parentPatchInfo.getBounds(parent_low, parent_high, meshname);
          
          // For node based meshes add one if there is a neighbor patch.
          if( meshname.find("NC_") == 0 )
          {
            int nlow[3], nhigh[3];
            parentPatchInfo.getBounds(nlow, nhigh, "NEIGHBORS");
            
            for (int i=0; i<3; i++)
              parent_high[i] += nhigh[i];
          }
          
          int mins[3], maxs[3];
          for (int i=0; i<3; ++i)
          {
            mins[i] = std::max( child_low[i],
                                parent_low[i] *levelInfoChild.refinementRatio[i]);
            maxs[i] = std::min( child_high[i],
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

    // Now that the extents for each patch is known and what its
    // children are, pass the structured domain boundary that
    // information.
    for (int p=0; p<totalPatches; ++p)
    {
      int my_level, local_patch;
      GetLevelAndLocalPatchNumber(stepInfo, p, my_level, local_patch);
        
      PatchInfo &patchInfo =
        stepInfo->levelInfo[my_level].patchInfo[local_patch];

      int plow[3], phigh[3];
      patchInfo.getBounds(plow, phigh, meshname);
        
      int extents[6];

      // For node based meshes add one if there is a neighbor patch.
      if( meshname.find("NC_") == 0 )
      {
        int nlow[3], nhigh[3];
        patchInfo.getBounds(nlow, nhigh, "NEIGHBORS");
        
        // For node meshes always subtract two because the domain
        // extents is inclusive.
        for (int i=0; i<3; i++)
        {
          phigh[i] += nhigh[i];

          extents[i+0] = plow[i];
          extents[i+3] = phigh[i] - 2;
        }
      }
      else
      {
        // For cell and face meshes always subtract one because the
        // domain extents is inclusive.
        for (int i=0; i<3; i++)
        {
          extents[i+0] = plow[i];
          extents[i+3] = phigh[i] - 1;
        }
      }

      // These extents are inclusive.
      VisIt_DomainNesting_set_nestingForPatch(dn, p, my_level,
                                              &(childPatches[p][0]),
                                              childPatches[p].size(),
                                              extents);

      // std::cerr << "\tdn->SetNestingForDomain("
      //                << p << "," << my_level << ") <"
      //                << extents[0] << "," << extents[1] << "," << extents[2] << "> to <"
      //                << extents[3] << "," << extents[4] << "," << extents[5] << ">";

      // std::cerr << "\t children patches <";
        
      // for (int i=0; i<childPatches[p].size(); ++i)
      //        std::cerr << childPatches[p][i] << ",  ";

      // std::cerr << ">" << std::endl;;
    }

    return dn;

    // forceMeshReload = false;

  }
  else
    return VISIT_INVALID_HANDLE;
}


// ****************************************************************************
//  Method: visit_SimGetMesh
//
//  Purpose:
//      Callback for processing a mesh
//
// ****************************************************************************
visit_handle visit_SimGetMesh(int domain, const char *meshname, void *cbdata)
{
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SchedulerP     schedulerP = sim->simController->getSchedulerP();
  LoadBalancer * lb         = sim->simController->getLoadBalancer();
  GridP          gridP      = sim->gridP;

  std::string meshName(meshname);
  
  // Communication mesh using an adjacency matrix via a point cloud.
  if( meshName.find("Communication_") == 0 )
  {   
    SchedulerCommon *scheduler = dynamic_cast<SchedulerCommon*>
      (sim->simController->getSchedulerP().get_rep());

    // There may be multiple communication matrices, the index will be last.
    unsigned int index = 0;    
    if( scheduler->getNumTaskGraphs() > 1 ) {
      size_t found = meshName.find_last_of("/");

      index = stoi( meshName.substr(found + 1) );
    }
    
    std::pair< std::string, std::string > allTasks("All", "Tasks");
    
    MapInfoMapper< unsigned int, CommunicationStatsEnum, unsigned int > &comm_info =
      scheduler->getTaskGraph(index)->getDetailedTasks()->getCommInfo()[allTasks];
    
    unsigned int nComms = comm_info.size();
    
    // Some nodes may not be communicating.
    if( nComms == 0 )
      return VISIT_INVALID_HANDLE;

    // Create the point locations. The abscissa is the rank while the
    // ordinate is the coresponding rank.
    double *values = new double[ 3*nComms ];
    
    for( unsigned int i=0; i<nComms; ++i) {
      values[i*3+0] = sim->myworld->myRank();
      values[i*3+1] = comm_info.getKey(i);
      values[i*3+2] = 0;
    }
 
    visit_handle meshH = VISIT_INVALID_HANDLE;

    if(VisIt_PointMesh_alloc(&meshH) == VISIT_OKAY)
    {
      visit_handle cordsH = VISIT_INVALID_HANDLE;

      if(VisIt_VariableData_alloc(&cordsH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(cordsH, VISIT_OWNER_VISIT,
                                    3, nComms, values);

        VisIt_PointMesh_setCoords(meshH, cordsH );
      }

      // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
      // owns the data (VISIT_OWNER_SIM - indicates the simulation owns
      // the data). However pd needs to be deleted.
    }

    return meshH;
  }
  else if( meshName.find("Machine_") == 0 )
  {
    // nConnections are for quads so the type plus four points.
    const unsigned int nQuadVals = 5;
    unsigned int nConnections = 0;
    int* connections = nullptr;

    unsigned int nPts = 0;
    float* xPts = nullptr;
    float* yPts = nullptr;
    // float* zPts = nullptr;

    // Create a quad for each rank on a node.
    if( meshName.find("/Global") != std::string::npos )
    {    
      // Only rank 0 return the global mesh.
      if( sim->myworld->myRank() != 0 )
        return VISIT_INVALID_HANDLE;

      // Total size of the layout - add one to have gap between nodes.
      const unsigned int xMax = sim->switchNodeList.size() * (sim->xNode+1);
      const unsigned int yMax = sim->maxNodes              * (sim->yNode+1);

      // Set all of the points as rectilinear grid.
      nPts = xMax * yMax;

      xPts = new float[nPts];
      yPts = new float[nPts];
      // zPts = new float[nPts];

      for( unsigned int j=0; j<yMax; ++j)
      {
        for( unsigned int i=0; i<xMax; ++i)
        {
          xPts[j*xMax+i] = i;
          yPts[j*xMax+i] = j;
          // zPts[j*xMax+i] = 0;
        }
      }

      const unsigned int totalCores =
        sim->switchNodeList.size()*sim->maxNodes*sim->maxCores;

      // Connections are for quads so the type plus four points.
      connections = new int[ totalCores * nQuadVals ];

      // Loop through each switch.
      for( unsigned int s=0; s<sim->switchNodeList.size(); ++s )
      {
        // Get the node x index based on the xNode size.
        unsigned int bx = s * (sim->xNode+1);
        
        // Loop through each node.
        for( unsigned int n=0; n<sim->switchNodeList[s].size(); ++n )
        {
          // Get the node y index based on the yNode size.
          unsigned int by = n * (sim->yNode+1);
          
          // Get the number of cores for this node.
          unsigned int nCores = 0;

          for( unsigned int i=0; i<sim->nodeCores.size(); ++i )
          {
            if( sim->nodeStart[i] <= sim->switchNodeList[s][n] &&
                sim->switchNodeList[s][n] <= sim->nodeStop[i] )
            {
              nCores = sim->nodeCores[i];
              break;
            }
          }
          
          // Loop through each core.
          for( unsigned int c=0; c<nCores; ++c )
          {
            // Get the core x y index based on the xNode size.
            unsigned int lx = bx + c % sim->xNode;
            unsigned int ly = by + c / sim->xNode;
            
            // All cells are quads
            connections[nConnections++] = VISIT_CELL_QUAD;
            
            // Set the index based on a rectilinear grid.
            connections[nConnections++] = (ly+0) * xMax + (lx+0);
            connections[nConnections++] = (ly+1) * xMax + (lx+0);
            connections[nConnections++] = (ly+1) * xMax + (lx+1);
            connections[nConnections++] = (ly+0) * xMax + (lx+1);
          }
        }
      }
    }

    // Create a quad for the rank (domain).
    else if( meshName.find("/Local" ) != std::string::npos )
    {
      // Total size of the layout one core per domain;
      const unsigned int xMax = 2;
      const unsigned int yMax = 2;

      // Set all of the points as rectilinear grid.
      nPts = xMax * yMax;

      // Indexes for the switch, node, and core.
      unsigned int s = sim->switchIndex;
      unsigned int n = sim->nodeIndex;
      unsigned int c = sim->myworld->myNode_myRank();

      // Get the node x y location based on the x/y node size.
      unsigned int bx = s * (sim->xNode+1);
      unsigned int by = n * (sim->yNode+1);
          
      // Get the core x y location based on the xNode size.
      unsigned int lx = bx + c % sim->xNode;
      unsigned int ly = by + c / sim->xNode;
            
      xPts = new float[nPts];
      yPts = new float[nPts];
      // zPts = new float[nPts];

      for( unsigned int j=0; j<yMax; ++j)
      {
        for( unsigned int i=0; i<xMax; ++i)
        {
          xPts[j*xMax+i] = lx + i;
          yPts[j*xMax+i] = ly + j;
          // zPts[j*xMax+i] = 0;
        }
      }

      // Connections are for quads so the type plus four points.
      connections = new int[ nQuadVals ];

      // All cells are quads
      connections[nConnections++] = VISIT_CELL_QUAD;
      
      // Set the index based on a rectilinear grid.
      connections[nConnections++] = (+0) * xMax + (+0);
      connections[nConnections++] = (+1) * xMax + (+0);
      connections[nConnections++] = (+1) * xMax + (+1);
      connections[nConnections++] = (+0) * xMax + (+1);
    }

    // Create a quad for each rank on a node.
    if( meshName.find("/Thread") != std::string::npos )
    {
      // Indexes for the switch, node, and core.
      unsigned int s = sim->switchIndex;
      unsigned int n = sim->nodeIndex;
      unsigned int c = sim->myworld->myNode_myRank();
      unsigned int nRanks = sim->myworld->myNode_nRanks();
      
      unsigned int nSockets = sysGetNumSockets();
      unsigned int nCoresPerSocket = sysGetNumCoresPerSockets();
      unsigned int nThreadsPerCore = sysGetNumThreadsPerCore();
      unsigned int nThreads = 0;

      // If the number of ranks is one then not taking advantage of
      // the sockets.
      if( nRanks == 1 )
      {
        nThreads = nSockets * nCoresPerSocket * nThreadsPerCore;
      }      
      // If the number of ranks greater than one then it must equal
      // the number of sockets.
      else if( nRanks == nSockets )
      {
        nThreads = nCoresPerSocket * nThreadsPerCore;
      }
      // Over subscribed - what to do ??
      else
      {
        return VISIT_INVALID_HANDLE;
      }

      // Over subscribed - what to do ??
      if( nThreads < (unsigned int) Uintah::Parallel::getNumThreads() )
      {
        return VISIT_INVALID_HANDLE;
      }

      // Connections are for quads so the type plus four points.
      connections = new int[ nThreads * nQuadVals ];

      // Total size of the layout for one thread;
      const unsigned int xMax = 2;
      const unsigned int yMax = 2;
        
      // Set all of the points as rectilinear grid.
      nPts = nThreads * xMax * yMax;
      xPts = new float[nPts];
      yPts = new float[nPts];
      // zPts = new float[nPts];

      // Get the node x y location based on the x/y node size.
      unsigned int bx = s * (sim->xNode+1);
      unsigned int by = n * (sim->yNode+1);

      float dx = 1.0 / (float) nThreadsPerCore;
      
      // Loop through each thread.
      for( unsigned int t=0; t<nThreads; ++t )
      {
        unsigned int threadID = c + t * nRanks;

        unsigned int core        = threadID % (nSockets*nCoresPerSocket);
        unsigned int hyperThread = threadID / (nSockets*nCoresPerSocket);
          
        // Get the core x y location based on the xNode size.
        float lx = bx + core % sim->xNode + hyperThread * dx;
        float ly = by + core / sim->xNode;
        
        for( unsigned int j=0; j<yMax; ++j)
        {
          for( unsigned int i=0; i<xMax; ++i)
          {
            xPts[t*xMax*yMax+j*xMax+i] = lx + i * dx;
            yPts[t*xMax*yMax+j*xMax+i] = ly + j;
            // zPts[t*xMax*yMax+j*xMax+i] = 0;
          }
        }

        // All cells are quads
        connections[nConnections++] = VISIT_CELL_QUAD;
        
        // Set the index based on a rectilinear grid.
        connections[nConnections++] = t*xMax*yMax + (+0) * xMax + (+0);
        connections[nConnections++] = t*xMax*yMax + (+1) * xMax + (+0);
        connections[nConnections++] = t*xMax*yMax + (+1) * xMax + (+1);
        connections[nConnections++] = t*xMax*yMax + (+0) * xMax + (+1);
      }
    }

    // For each rank create a quad for it's patches, point
    // communication rank, or tasks.
    else if( meshName.find("/Patch" ) != std::string::npos ||
             meshName.find("/Communication") != std::string::npos ||
             meshName.find("/Tasks") != std::string::npos )
    {
      unsigned int nValues = 0;

      if( meshName.find("/Patch") != std::string::npos ) {
        const PatchSubset* myPatches =
          lb->getPerProcessorPatchSet(gridP)->getSubset( domain );

        nValues = myPatches->size();
      }
      else if( meshName.find("/Communication") != std::string::npos ) {
      
        SchedulerCommon *scheduler = dynamic_cast<SchedulerCommon*>
          (sim->simController->getSchedulerP().get_rep());
      
        // There may be multiple communication matrices, the index
        // will be last.
        unsigned int index = 0;    
        if( scheduler->getNumTaskGraphs() > 1 ) {
          size_t found = meshName.find_last_of("/");
          
          index = stoi( meshName.substr(found + 1) );
        }

        std::pair< std::string, std::string > allTasks("All", "Tasks");
        
        MapInfoMapper< unsigned int, CommunicationStatsEnum, unsigned int > &comm_info =
          scheduler->getTaskGraph(index)->getDetailedTasks()->getCommInfo()[allTasks];
      
        nValues = comm_info.size();
      }
      else if( meshName.find("/Tasks") != std::string::npos ) {
        
        MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
          (sim->simController->getSchedulerP().get_rep());
        
        nValues = mpiScheduler->m_task_info.size();
      }
      
      // Some ranks may not have patches, threads or be communicating.
      if( nValues == 0 )
        return VISIT_INVALID_HANDLE;
      
      // Total size of the layout. Try to make rectangles.
      unsigned int xMax = sqrt(nValues);
      unsigned int yMax = xMax;

      // Make sure to cover all the thread - may be blank areas.
      while( xMax * yMax < nValues )
        ++yMax;

      // Add one to get the far boundary.
      xMax += 1;
      yMax += 1;

      // Set all of the points as rectilinear grid.
      nPts = xMax * yMax;

      // Indexes for the switch, node, and core
      unsigned int s = sim->switchIndex;
      unsigned int n = sim->nodeIndex;
      unsigned int c = sim->myworld->myNode_myRank();

      // Get the node x y location based on the x/y node size.
      unsigned int bx = s * (sim->xNode+1);
      unsigned int by = n * (sim->yNode+1);
          
      // Get the core x y location based on the xNode size.
      unsigned int lx = bx + c % sim->xNode;
      unsigned int ly = by + c / sim->xNode;
            
      xPts = new float[nPts];
      yPts = new float[nPts];
      // zPts = new float[nPts];

      float dx = 1.0 / (float) (xMax-1);
      float dy = 1.0 / (float) (yMax-1);

      for( unsigned int j=0; j<yMax; ++j)
      {
        for( unsigned int i=0; i<xMax; ++i)
        {
          xPts[j*xMax+i] = lx + i * dx;
          yPts[j*xMax+i] = ly + j * dy;
          // zPts[j*xMax+i] = 1;
        }
      }

      // Connections are for quads so the type plus four points.
      connections = new int[ nValues * nQuadVals ];

      for (unsigned int  p = 0; p < nValues; ++p) {

        // Get an x y index based on the xMax size.
        unsigned int px = p % (xMax-1);
        unsigned int py = p / (xMax-1);

        // All cells are quads
        connections[nConnections++] = VISIT_CELL_QUAD;
        
        // Set the index based on a rectilinear grid.
        connections[nConnections++] = (py+0) * xMax + (px+0);
        connections[nConnections++] = (py+1) * xMax + (px+0);
        connections[nConnections++] = (py+1) * xMax + (px+1);
        connections[nConnections++] = (py+0) * xMax + (px+1);
      }
    }

    // nConnections are for quads so the type plus four points.
    unsigned int nCells = nConnections / nQuadVals;

    visit_handle meshH = VISIT_INVALID_HANDLE;

    // Do not pass the Z coordinate as there is a bug for quads.
    if(VisIt_UnstructuredMesh_alloc(&meshH) == VISIT_OKAY)
    {
      visit_handle xH ,yH; //, zH;
      VisIt_VariableData_alloc( &xH );
      VisIt_VariableData_alloc( &yH );
      // VisIt_VariableData_alloc( &zH );
      
      VisIt_VariableData_setDataF( xH, VISIT_OWNER_VISIT, 1, nPts, xPts );
      VisIt_VariableData_setDataF( yH, VISIT_OWNER_VISIT, 1, nPts, yPts );
      // VisIt_VariableData_setDataF( zH, VISIT_OWNER_VISIT, 1, nPts, zPts );
    
      visit_handle connH;
      VisIt_VariableData_alloc( &connH );
      VisIt_VariableData_setDataI( connH, VISIT_OWNER_VISIT, 1,
                                   nConnections, connections );

      VisIt_UnstructuredMesh_setCoordsXY( meshH, xH, yH );
      // VisIt_UnstructuredMesh_setCoordsXYZ( meshH, xH, yH, zH );
      VisIt_UnstructuredMesh_setConnectivity( meshH, nCells, connH );

      // No need to delete the points or the connections as the flag
      // is VISIT_OWNER_VISIT so VisIt owns the data (VISIT_OWNER_SIM
      // - indicates the simulation owns the data).
    }

    return meshH;
  }  

  TimeStepInfo* &stepInfo = sim->stepInfo;

  visit_handle meshH = VISIT_INVALID_HANDLE;

  int level, local_patch;
  GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

  // Particle mesh
  if (meshName.find("Particle_Mesh") != std::string::npos)
  {
    size_t found = meshName.find("/");
    std::string matl = meshName.substr(found + 1);

    int matlNo = -1;
    if (matl.compare("*") != 0)
      matlNo = atoi(matl.c_str());

    const std::string &varName = Uintah::VarLabel::getParticlePositionName();
    
    ParticleDataRaw *pd =
      getParticleData(schedulerP, gridP, level, local_patch, varName, matlNo);
    
    if(pd && VisIt_PointMesh_alloc(&meshH) == VISIT_OKAY)
    {
      visit_handle cordsH = VISIT_INVALID_HANDLE;

      if(VisIt_VariableData_alloc(&cordsH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(cordsH, VISIT_OWNER_VISIT,
                                    pd->components, pd->num, pd->data);

        VisIt_PointMesh_setCoords(meshH, cordsH );
      }
      
      // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
      // owns the data (VISIT_OWNER_SIM - indicates the simulation owns
      // the data). However pd needs to be deleted.
      
      // delete pd->data
      delete pd;
    }

#ifdef COMMENTOUT_FOR_NOW
    //try to retrieve existing cache ref
    void_ref_ptr vrTmp =
      cache->GetVoidRef(meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
                        timeStep, domain);

    vtkDataArray *pID = nullptr;

    if (*vrTmp == nullptr)
    {
      //
      // add globel node ids to facilitate point cloud usage
      //
      //basically same as GetVar(timeStep, domain, "particleID");
      int level, local_patch;
      //debug5<<"\tGetLevelAndLocalPatchNumber...\n";
      GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

      int matlNo = -1;
      if (matl.compare("*") != 0)
        matlNo = atoi(matl.c_str());

      ParticleDataRaw *pd = nullptr;

      //debug5<<"\t(*getParticleData)...\n";
      //todo: this returns an array of doubles. Need to return
      //expected datatype to avoid unnecessary conversion.
      pd = getParticleData(schedulerP, gridP, level, local_patch,
                           "p.particleID", matlNo, timeStep);

      //debug5 << "got particle data: "<<pd<<"\n";
      if (pd)
      {
        vtkDoubleArray *rv = vtkDoubleArray::New();
        //vtkLongArray *rv = vtkLongArray::New();
        //debug5<<"\tSetNumberOfComponents("<<pd->components<<")...\n";
        rv->SetNumberOfComponents(pd->components);

        //debug5<<"\tSetArray...\n";
        rv->SetArray(pd->data, pd->num*pd->components, 0);

        // Don't delete pd->data - vtk owns it now!
        delete pd;
        
        // todo: this is the unnecesary conversion, from long
        // int->double->int, to say nothing of the implicit curtailing
        // that might occur (note also: this is a VisIt bug that uses
        // ints to store particle ids rather than long ints)
        vtkIntArray *iv = ConvertToInt(rv);
        //vtkLongArray *iv=ConvertToLong(rv);
        rv->Delete(); // this should now delete pd->data

        pID = iv;
      }

      //debug5<<"read particleID ("<<pID<<")\n";
      if(pID != nullptr)
      {
        //debug5<<"adding global node ids from particleID\n";
        pID->SetName("avtGlobalNodeId");
        void_ref_ptr vr =
          void_ref_ptr( pID , avtVariableCache::DestructVTKObject );

        cache->CacheVoidRef( meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
                             timeStep, domain, vr );

        //make sure it worked
        void_ref_ptr vrTmp =
          cache->GetVoidRef(meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
                            timeStep, domain);

        if (*vrTmp == nullptr || *vrTmp != *vr)
          throw InvalidFilesException("failed to register uda particle global node");
      }
    }

    return ugrid;
#endif
  }

  // Patch mesh
  else if (meshName.find("Patch_Mesh") != std::string::npos)
  {
    LevelInfo &levelInfo = stepInfo->levelInfo[level];
    PatchInfo &patchInfo = levelInfo.patchInfo[local_patch];

    int dims[3] = {2, 2, 2}, base[3] = {0, 0, 0};

    // Get the patch bounds
    int plow[3], phigh[3];
    patchInfo.getBounds(plow, phigh, "CC_MESH");

    // debug5 << "Calculating vtkRectilinearGrid mesh for "
    //     << meshName << " mesh (" << rgrid << ").\n";

    visit_handle cordH[3] = { VISIT_INVALID_HANDLE,
                              VISIT_INVALID_HANDLE,
                              VISIT_INVALID_HANDLE };

    // Set the coordinates of the grid points in each direction.
    for (int c=0; c<3; ++c)
    {
      if(VisIt_VariableData_alloc(&cordH[c]) == VISIT_OKAY)
      {
        float *array = new float[ 2 ];

        array[0] = levelInfo.anchor[c] +  plow[c] * levelInfo.spacing[c];
        array[1] = levelInfo.anchor[c] + phigh[c] * levelInfo.spacing[c];

        VisIt_VariableData_setDataF(cordH[c], VISIT_OWNER_VISIT,
                                    1, dims[c], array);

        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data).
        
        // delete[] array;
      }
    }

    if(VisIt_RectilinearMesh_alloc(&meshH) == VISIT_OKAY)
    {
      // Fill in the attributes of the RectilinearMesh.
      VisIt_RectilinearMesh_setCoordsXYZ(meshH, cordH[0], cordH[1], cordH[2]);
      VisIt_RectilinearMesh_setRealIndices(meshH, base, dims);
      VisIt_RectilinearMesh_setBaseIndex(meshH, base);

      // VisIt_RectilinearMesh_setGhostCells(meshH, visit_handle gz);
      // VisIt_RectilinearMesh_setGhostNodes(meshH, visit_handle gn);
    }
  }
  // CC, NC, or SFC* meshes
  else
  {
    LevelInfo &levelInfo = stepInfo->levelInfo[level];
    PatchInfo &patchInfo = levelInfo.patchInfo[local_patch];

    int dims[3], base[3] = {0, 0, 0};

    // Get the patch bounds
    int plow[3], phigh[3];
    patchInfo.getBounds(plow, phigh, meshName);

    // For node based meshes add one if there is a neighbor patch.
    if( meshName.find("NC_") == 0 )
    {
      int nlow[3], nhigh[3];
      patchInfo.getBounds(nlow, nhigh, "NEIGHBORS");
      
      for (int i=0; i<3; i++)
      {
        phigh[i] += nhigh[i];
        dims[i] = phigh[i] - plow[i];
      }
    }
    else
    {      
      // For cell and face meshes always add one.
      for (int i=0; i<3; i++) 
      {
        dims[i] = phigh[i] - plow[i] + 1;
      }
    }

    // debug5 << "Calculating vtkRectilinearGrid mesh for "
    //     << meshName << " mesh (" << rgrid << ").\n";

    // These are needed to offset grid points in order to preserve
    // face centered locations on node-centered domain.
    bool sfck[3] = { meshName.find("SFCX") != std::string::npos,
                     meshName.find("SFCY") != std::string::npos,
                     meshName.find("SFCZ") != std::string::npos };

    int nlow[3], nhigh[3];

    if( sfck[0] || sfck[1] || sfck[2] )
      patchInfo.getBounds(nlow, nhigh, "NEIGHBORS");

    visit_handle cordH[3] = { VISIT_INVALID_HANDLE,
                              VISIT_INVALID_HANDLE,
                              VISIT_INVALID_HANDLE };

    // Set the coordinates of the grid points in each direction.
    for (int c=0; c<3; ++c)
    {
      if(VisIt_VariableData_alloc(&cordH[c]) == VISIT_OKAY)
      {
        float *array = new float[ dims[c] ];

        for (int i=0; i<dims[c]; ++i)
        {
          // Face centered data gets shifted towards -inf by half a cell.
          // Boundary patches are special shifted to preserve global domain.
          // Internal patches are always just shifted.
          float face_offset=0;
          if (sfck[c]) 
          {
            if (i==0)
            {
              // No neighbor, so the patch is on low boundary
              if (nlow[c] == 0)
                face_offset = 0.0;
              // patch boundary is internal to the domain
              else
                face_offset = -0.5;
            }
            else if (i==dims[c]-1)
            {
              // No neighbor, so the patch is on high boundary
              if (nhigh[c] == 0)
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
            (i + plow[c] + face_offset) * levelInfo.spacing[c];
        }

        VisIt_VariableData_setDataF(cordH[c], VISIT_OWNER_VISIT,
                                    1, dims[c], array);

        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data).

        // delete[] array;
      }
    }

    if(VisIt_RectilinearMesh_alloc(&meshH) == VISIT_OKAY)
    {
      // Fill in the attributes of the RectilinearMesh.
      VisIt_RectilinearMesh_setCoordsXYZ(meshH, cordH[0], cordH[1], cordH[2]);
      VisIt_RectilinearMesh_setRealIndices(meshH, base, dims);
      VisIt_RectilinearMesh_setBaseIndex(meshH, base);

      // ARS - FIXME
      // VisIt_RectilinearMesh_setGhostCells(meshH, visit_handle gz);
      // VisIt_RectilinearMesh_setGhostNodes(meshH, visit_handle gn);
    }
  }

  return meshH;
}

// ****************************************************************************
//  Method: visit_SimGetVariable
//
//  Purpose:
//      Callback for processing a variable
//
// ****************************************************************************
visit_handle visit_SimGetVariable(int domain, const char *varname, void *cbdata)
{
  visit_handle varH = VISIT_INVALID_HANDLE;
    
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  unsigned int addMachineData = (sim->switchNodeList.size() &&
                                 (int) sim->switchIndex != -1 &&
                                 (int) sim->nodeIndex   != -1 );

  std::string varName(varname);

  // Machine mesh only variables: NodeID, MPI/Node, MPI/Rank, and MPI/Comm/Rank
  if( varName.find("Processor/Communication/") == 0 )
  {
    {
      SchedulerCommon *scheduler = dynamic_cast<SchedulerCommon*>
        (sim->simController->getSchedulerP().get_rep());
      
      // There may be multiple communication matrices, the index will be last.
      unsigned int index = 0;
      if( scheduler->getNumTaskGraphs() > 1 ) {
        size_t found = varName.find_last_of("/");
        
        index = stoi( varName.substr(found + 1) );
        varName = varName.substr(0, found);
      }

      std::pair< std::string, std::string > allTasks("All", "Tasks");
      
      std::map< std::pair< std::string, std::string >,
                MapInfoMapper< unsigned int, CommunicationStatsEnum, unsigned int > > &comm_info =
        scheduler->getTaskGraph(index)->getDetailedTasks()->getCommInfo();
      
      unsigned int nValues = comm_info[allTasks].size();

      // Some nodes may not be communicating.
      if( nValues == 0 )
        return varH;
      
      double *values = new double[ nValues ];
      
      if( varName.find("Processor/Communication/CommID") == 0 ) {
        for( unsigned int i=0; i<nValues; ++i)
          values[i] = comm_info[allTasks].getKey(i);
      }
      else if( varName.find("Processor/Communication/") == 0 ) {

        // Lop off the prefix
        size_t found = std::string("Processor/Communication/").size();
        std::string statName = varName.substr(found);

        // Task pair name and the stat
        found = statName.find("/");
        std::string pairName = statName.substr(0, found);
        statName = statName.substr(found + 1);

        found = pairName.find("|");
        std::string toTaskName   = pairName.substr(0, found);
        std::string fromTaskName = pairName.substr(found + 1);

        std::pair< std::string, std::string > taskPair( toTaskName, fromTaskName );
        
        for( unsigned int i=0; i<nValues; ++i) {

          unsigned int key = comm_info[taskPair].getKey(i);
            
          if( comm_info[taskPair][key].exists(statName) )
            values[i] = comm_info[taskPair][key].getRankValue(statName);
          else {
            values[i] = 0;
            
            std::stringstream msg;
            msg << "Visit libsim - for domain " << domain << "  "
                << "Uintah Processor/Machine/Communication/" << pairName << "  " << key
                << " variable \"" << statName << "\"  " << "does not exist.";
            
            VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);
          }
        }
      }
      else
      {
        for( unsigned int i=0; i<nValues; ++i)
          values[i] = 0;

        std::stringstream msg;
        msg << "Visit libsim - " << domain << "  "
            << "Uintah variable \"" << varName << "\"  "
            << "does not exist.";
        
        VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);
      }

      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, 1, nValues, values);
        
        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data).
      }

      return varH;
    }
  }
  // Machine mesh only variables: NodeID, MPI/Node, MPI/Rank, and MPI/Comm/Rank
  else if( varName.find("Processor/Machine/") == 0 )
  {
    // At the present time all vars are local.
    // bool global = (varName.find("Global") != std::string::npos);
    // bool local  = (varName.find("Local" ) != std::string::npos);
    // bool thread = (varName.find("Thread") != std::string::npos);
    // bool task   = (varName.find("Task  ") != std::string::npos);

    bool global = false;

    bool local = (varName.find("Processor/Machine/NodeID") == 0 ||
                  varName.find("Processor/Machine/MPI/Node") == 0 ||
                  varName.find("Processor/Machine/MPI/Rank") == 0 ||
                  varName.find("Processor/Machine/MPI/Comm/Rank") ||
                  varName.find("Processor/Machine/Tasks/") == 0);

    bool thread        = (varName.find("Processor/Machine/Thread") == 0 );
    bool communication = (varName.find("Processor/Machine/Communication") == 0 );
    bool task          = (varName.find("Processor/Machine/Tasks/All") == 0 );

    // Only rank 0 return the whole of the mesh.
    // if( global && sim->myworld->myRank() != 0 )
    //   return VISIT_INVALID_HANDLE;

    if( global )
    {
    //   unsigned int totalCores =
    //     sim->switchNodeList.size() * sim->maxNodes * sim->maxCores;

    //   nValues = 0;
    //   values = new int[ totalCores ];

    //   for( unsigned int i=0; i<totalCores; ++i)
    //     values[i] = 0;
    
    //   // Loop through each switch.
    //   for( unsigned int s=0; s<sim->switchNodeList.size(); ++s )
    //   {
    //     unsigned int nCores = 0;

    //     // Loop through each node.
    //     for( unsigned int n=0; n<sim->switchNodeList[s].size(); ++n )
    //     {
    //       for( unsigned int i=0; i<sim->nodeCores.size(); ++i )
    //       {
    //         if( sim->nodeStart[i] <= sim->switchNodeList[s][n] &&
    //             sim->switchNodeList[s][n] <= sim->nodeStop[i] )
    //         {
    //           nCores  = sim->nodeCores[i];
    //           break;
    //         }
    //       }

    //       // Loop through each core.
    //       for( unsigned int i=0; i<nCores; ++i )
    //       {
    //         if( varName.find("Processor/Machine/NodeID") == 0 )
    //           values[nValues++] = atoi(sim->hostNode.c_str());
    //         else if( varName.find("Processor/Machine/MPI/Node") == 0 )
    //           values[nValues++] = sim->myworld->myNode();
    //         else if( varName.find("Processor/Machine/MPI/Rank") == 0 )
    //           values[nValues++] = sim->myworld->myRank();
    //         else if( varName.find("Processor/Machine/MPI/Comm/Rank") == 0 )
    //           values[nValues++] = sim->myworld->myNode_myRank();
    //       }
    //     }
    //   }
    }

    else if( local && !task )
    {
      unsigned int nValues = 1;  // 1 Core
      double *values = new double[ nValues ];

      if( varName.find("Processor/Machine/NodeID") == 0 )
        values[0] = atoi(sim->hostNode.c_str());
      else if( varName.find("Processor/Machine/MPI/Node") == 0 )
        values[0] = sim->myworld->myNode();
      else if( varName.find("Processor/Machine/MPI/Rank") == 0 )
        values[0] = sim->myworld->myRank();
      else if( varName.find("Processor/Machine/MPI/Comm/Rank") == 0 )
        values[0] = sim->myworld->myNode_myRank();
      else if( varName.find("Processor/Machine/Tasks/") == 0 ) {

        MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
          (sim->simController->getSchedulerP().get_rep());
        
        size_t found = varName.find_last_of("/");
        std::string statName = varName.substr(found + 1);

        varName = varName.substr(0, found);
        found = std::string("Processor/Machine/Tasks/").size();
        std::string taskName = varName.substr(found);

        values[0] = mpiScheduler->m_task_info[taskName].getRankValue(statName);
      }
      
      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, 1, nValues, values);
        
        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data).
      }

      return varH;
    }

    else if( task )
    {
      MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
        (sim->simController->getSchedulerP().get_rep());

      MapInfoMapper< std::string, TaskStatsEnum, double >
        &task_info = mpiScheduler->m_task_info;
      
      unsigned int nValues = task_info.size();
      double *values = new double[ nValues ];

      size_t found = varName.find_last_of("/");
      std::string statName = varName.substr(found + 1);

      for( unsigned int i=0; i<nValues; ++i ) {

        std::string taskName = task_info.getKey(i);
        
        values[i] = task_info[taskName].getRankValue(statName);
      }

      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, 1, nValues, values);
        
        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data).
      }

      return varH;
    }
    
    else if( thread )
    {
      unsigned int nThreads = Uintah::Parallel::getNumThreads();

      // Some ranks may not have any threads.
      if( nThreads == 0 )
        return varH;
      
      UnifiedScheduler *unifiedScheduler = dynamic_cast<UnifiedScheduler*>
        (sim->simController->getSchedulerP().get_rep());

      unsigned int nRanks = sim->myworld->myNode_nRanks();
      
      unsigned int nSockets = sysGetNumSockets();
      unsigned int nCoresPerSocket = sysGetNumCoresPerSockets();
      unsigned int nThreadsPerCore = sysGetNumThreadsPerCore();

      unsigned int nValues = 0;
      double *values;

      // If the number of ranks is one then not taking advantage of
      // the sockets.
      if( nRanks == 1 )
      {
        nValues = nSockets * nCoresPerSocket * nThreadsPerCore;
      }      
      // If the number of ranks greater than one then it must equal
      // the number of sockets.
      else if( nRanks == nSockets )
      {
        nValues = nCoresPerSocket * nThreadsPerCore;
      }
      // Over subscribed - what to do ??
      else
      {
        return varH;
      }

      // Over subscribed - what to do ??
      if( nValues < nThreads )
      {
        return varH;
      }

      size_t found = std::string("Processor/Machine/Thread/").size();
      std::string statName = varName.substr(found);

      // MPI rank wise thread reduction variable
      if( statName.find_last_of("/") != std::string::npos )
      {
        nValues = 1;    
        values = new double[ nValues ];

        size_t found = statName.find_last_of("/");
        std::string reductionType = statName.substr(found + 1);
        statName = statName.substr(0, found);

        if( reductionType == "Size" )
          values[0] = unifiedScheduler->m_thread_info.size();
        else if( reductionType == "Sum" )
          values[0] = unifiedScheduler->m_thread_info.getSum( statName );
        else if( reductionType == "Average" )
          values[0] = unifiedScheduler->m_thread_info.getAverage( statName );
        else if( reductionType == "Minimum" )
          values[0] = unifiedScheduler->m_thread_info.getMinimum( statName );
        else if( reductionType == "Maximum" )
          values[0] = unifiedScheduler->m_thread_info.getMaximum( statName );
        else if( reductionType == "StdDev" )
          values[0] = unifiedScheduler->m_thread_info.getStdDev( statName );
      }
      // Thread wise variables
      else {

        // Set the initial values to -1 so they can be skipped when
        // visualized.
        values = new double[ nValues ];

        for( unsigned int i=0; i<nValues; ++i)
          values[i] = -1;

        // Get the thread values based on the affinity so they are
        // mapped to the correct core.
        for( unsigned int i=0; i<nThreads; ++i) {
          if( unifiedScheduler->m_thread_info[i].exists("Affinity") &&
              unifiedScheduler->m_thread_info[i].exists(statName) )
          {
            unsigned int core =
              unifiedScheduler->m_thread_info[i].getRankValue("Affinity");
            values[core] =
              unifiedScheduler->m_thread_info[i].getRankValue(statName);
          }
          else
          {
            std::stringstream msg;
            msg << "Visit libsim - for domain " << domain << "  "
                << "Uintah Processor/Machine/Thread " << i
                << " variable \"" << statName << "\"  " << "does not exist.";

            VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);
          }
        }
      }

      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, 1, nValues, values);
        
        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data).
      }

      return varH;
    }

    else if( communication )
    {
      SchedulerCommon *scheduler = dynamic_cast<SchedulerCommon*>
        (sim->simController->getSchedulerP().get_rep());

      // There may be multiple communication matrices, the index will be last.
      unsigned int index = 0;

      if( scheduler->getNumTaskGraphs() > 1 ) {
        size_t found = varName.find_last_of("/");
        
        index = stoi( varName.substr(found + 1) );
        varName = varName.substr(0, found);
      }

      std::pair< std::string, std::string > allTasks("All", "Tasks");
      
      std::map< std::pair< std::string, std::string >,
                MapInfoMapper< unsigned int, CommunicationStatsEnum, unsigned int > > &comm_info =
        scheduler->getTaskGraph(index)->getDetailedTasks()->getCommInfo();
      
      unsigned int nValues = comm_info[allTasks].size();
      double *values;
      
      // Some nodes may not be communicating.
      if( nValues == 0 )
        return varH;
      
      size_t found = std::string("Processor/Machine/Communication/").size();
      std::string statName = varName.substr(found);

      // The communication id is the other MPI rank.
      if( statName.find("CommID") == 0 ) {

        values = new double[ nValues ];

        for( unsigned int i=0; i<nValues; ++i)
          values[i] = comm_info[allTasks].getKey(i);
      }

      else {

        // Lop off the prefix
        found = std::string("Processor/Machine/Communication/").size();
        statName = varName.substr(found);

        // Task pair name and the stat
        found = statName.find("/");
        std::string pairName = statName.substr(0, found);
        statName = statName.substr(found + 1);

        found = pairName.find("|");
        std::string toTaskName   = pairName.substr(0, found);
        std::string fromTaskName = pairName.substr(found + 1);

        std::pair< std::string, std::string > taskPair( toTaskName, fromTaskName );

        // Reduction variable
        if( statName.find("/") != std::string::npos )
        {
          nValues = 1;    
          values = new double[ nValues ];
          
          size_t found = statName.find_last_of("/");
          std::string reductionType = statName.substr(found + 1);
          statName = statName.substr(0, found);
          
          if( reductionType == "Size" )
            values[0] = comm_info[taskPair].size();
          else if( reductionType == "Sum" )
            values[0] = comm_info[taskPair].getSum( statName );
          else if( reductionType == "Average" )
            values[0] = comm_info[taskPair].getAverage( statName );
          else if( reductionType == "Minimum" )
            values[0] = comm_info[taskPair].getMinimum( statName );
          else if( reductionType == "Maximum" )
            values[0] = comm_info[taskPair].getMaximum( statName );
          else if( reductionType == "StdDev" )
            values[0] = comm_info[taskPair].getStdDev( statName );
        }
        else {
          values = new double[ nValues ];
          
          // Access is via keys only so get the first key
          // unsigned int key = comm_info[taskPair].getKey(0);
          
          for( unsigned int i=0; i<nValues; ++i) {
            
            unsigned int key = comm_info[taskPair].getKey(i);
            
            if( comm_info[taskPair][key].exists(statName) )
              values[i] = comm_info[taskPair][key].getRankValue(statName);
            else {
              values[i] = 0;
              
              std::stringstream msg;
              msg << "Visit libsim - for domain " << domain << "  "
                  << "Uintah Processor/Machine/Communication/" << pairName << "  " << key
                  << " variable \"" << statName << "\"  " << "does not exist.";
              
              VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);
            }
          }
        }
      }

      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, 1, nValues, values);
        
        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data).
      }

      return varH;
    }
  }

  // Variables that can be on the simulation mesh and the machine mesh.
  SchedulerP schedulerP = sim->simController->getSchedulerP();
  LoadBalancer * lb     = sim->simController->getLoadBalancer();
  GridP gridP           = sim->gridP;

  TimeStepInfo* &stepInfo = sim->stepInfo;

  // int timeStep = sim->cycle;

  bool isParticleVar = false;
  bool isInternalVar = false;
  bool isMachineMeshVar = false;

  // Get the var name sans the material. If a patch or processor
  // variable then the var name will be either "Patch" or "Processor".
  size_t found = varName.find("/");
  std::string matl = varName.substr(found + 1);
  varName = varName.substr(0, found);

  // Get the varType except for processor and patch based data which
  // does not come from the data warehouse but instead from internal
  // structures.
  std::string varType("");

  if( strncmp(varname, "Patch/Id", 8) == 0 )
  {
    isInternalVar = true;
    varType = "Patch_Mesh";

    // If the machine profile is available get the sim or host name.
    if( addMachineData )
    {
      std::string hostName = std::string(varname);
      found = hostName.find_last_of("/");
      hostName = hostName.substr(found + 1);

      isMachineMeshVar = (hostName == sim->hostName);
    }
  }
  else if( strncmp(varname, "Patch/Communication", 19) == 0 ||
           strncmp(varname, "Patch/Rank",          10) == 0 ||
           strncmp(varname, "Patch/Node",          10) == 0 ||
           strncmp(varname, "Patch/Tasks",         11) == 0 ||
           strncmp(varname, "Patch/Bounds/Low",    16) == 0 ||
           strncmp(varname, "Patch/Bounds/High",   17) == 0 )
  {
    isInternalVar = true;
    varType = "Patch_Mesh";
  }
  else if( strncmp(varname, "Patch/Nodes", 11) == 0 )
  {
    isInternalVar = true;
    varType = "NC_Mesh";
  }
  else if( strncmp(varname, "Nodes/", 6) == 0 ||
           strncmp(varname, "Ranks/", 6) == 0 )
  {
    isInternalVar = true;

    if(strncmp(&(varname[6]), "Particle_Mesh", 13) == 0)
    {
      isParticleVar = true;

      varType = "Particle_Mesh";

      // Get the material sans "Particle_Mesh".
      varName = std::string(varname);
      found = varName.find_last_of("/");
      matl = varName.substr(found + 1);
    }
    else
    {
      // Get the mesh type.
      varName = std::string(varname);
      found = varName.find_last_of("/");
      varType = varName.substr(found + 1);
    }
  }

  else if( varName == "Processor" )
  {
    isInternalVar = true;
    varType = "Patch_Mesh";

    // If the machine profile is available get the sim or host name.
    if( addMachineData )
    {
      std::string hostName = std::string(varname);
      found = hostName.find_last_of("/");
      hostName = hostName.substr(found + 1);

      isMachineMeshVar = (hostName == sim->hostName);
    }
  }
  // Grid variables
  else
  {
    // For PerPatch data remove the Patch/ prefix and get the var
    // name and set the material to zero.
    if( varName == "Patch" )
    {
      // Get the var name and material sans "Patch/".
      varName = std::string(varname);
      found = varName.find("/");  
      varName = varName.substr(found + 1);

      matl = "0";

      // If the machine profile is available get the sim or host name.
      if( addMachineData )
      {
        std::string hostName = std::string(varname);
        found = hostName.find_last_of("/");
        hostName = hostName.substr(found + 1);
        
        isMachineMeshVar = (hostName == sim->hostName);
        
        // Get the var name sans the sim or host name.
        found = varName.find_last_of("/");
        varName = varName.substr(0, found);
      }
    }

    // Get the var type and check to see if it is a particle variable.
    for (int k=0; k<(int)stepInfo->varInfo.size(); ++k)
    {
      if (stepInfo->varInfo[k].name == varName)
      {
        varType = stepInfo->varInfo[k].type;

        // Check for a particle variable
        if (stepInfo->varInfo[k].type.find("ParticleVariable") !=
            std::string::npos)
        {
          isParticleVar = true;
        }

        break;
      }
    }
  }

  if( varType.empty() )
  {
    std::stringstream msg;
    msg << "Visit libsim - " << domain << "  "
        << "Uintah variable \"" << varname << "\"  "
        << "has no type.";
    
    VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);

    return varH;
  }

  // Particle variable
  if (isParticleVar)
  {
    int level, local_patch;
    GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

    int matlNo = -1;
    if (matl.compare("*") != 0)
      matlNo = atoi(matl.c_str());

    ParticleDataRaw *pd = nullptr;
    
    if( isInternalVar )
    {
      // Patch node and ranks.
      if( strncmp(varname, "Nodes/", 6) == 0 ||
          strncmp(varname, "Ranks/", 6) == 0 )
      {
        unsigned int numParticles =
          getNumberParticles(schedulerP, gridP, level, local_patch, matlNo);
        
        if( numParticles )
        {
          pd = new ParticleDataRaw;
          pd->num = numParticles;
          pd->components = 1;
          pd->data = new double[pd->num * pd->components];
          
          double val = sim->myworld->myRank();
          
          for (int i=0; i<pd->num*pd->components; ++i)
            pd->data[i] = val;
        }
      }
    }
    else
    {
      pd = getParticleData(schedulerP, gridP, level, local_patch, varName, matlNo);
      
      if( pd )
        CheckNaNs(pd->data, pd->num*pd->components, varname, level, local_patch);
    }

    // Some patches may not have particles.
    if(pd && VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT,
                                  pd->components, pd->num, pd->data);
      
      // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
      // owns the data (VISIT_OWNER_SIM - indicates the simulation
      // owns the data). However, pd needs to be deleted.
      
      // delete pd->data
      delete pd;
    }
  }
  
  // Volume data that be on the simulation mesh or the machine mesh.
  else
  {
    int level, local_patch;
    GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

    ReductionInfoMapper< RuntimeStatsEnum, double > runtimeStats =
      sim->simController->getRuntimeStats();

    LevelInfo &levelInfo = stepInfo->levelInfo[level];
    PatchInfo &patchInfo = levelInfo.patchInfo[local_patch];

    // Get the patch bounds
    int plow[3], phigh[3];
    patchInfo.getBounds(plow, phigh, varType);

    bool nodeCentered = (varType.find("NC") != std::string::npos);

    // For node based meshes add one if there is a neighbor patch.
    if( nodeCentered )
    {
      int nlow[3], nhigh[3];
      patchInfo.getBounds(nlow, nhigh, "NEIGHBORS");

      for (int i=0; i<3; i++)
        phigh[i] += nhigh[i];
    }

    GridDataRaw *gd = nullptr;

    // The data for these variables does not come from the data
    // warehouse but instead from internal structures.
    if( isInternalVar )
    {
      gd = new GridDataRaw;

      // Processor data that is on the simulation mesh or the machine
      // meash. With either there is only value to return per patch or
      // per rank.
      if( varName == "Processor" )
      {
        // Using the patch mesh on the simulation
        gd->num = 1;
        gd->components = 1;
        gd->data = new double[gd->num * gd->components];

        std::string procLevelName;
        std::string operationName;
      
        std::string tmp = std::string(varname);

        // If the machine profile is available strip off the sim or host name.
        if( addMachineData )
        {
          // Strip off the "/Sim" or machine name.
          found = tmp.find_last_of("/");
          tmp = tmp.substr(0, found);
        }

        // Strip off the "Processor/" prefix
        found = tmp.find_first_of("/");
        tmp.erase(0, found + 1);

        // Strip off the "Application/", "MPI/" or "Runtime/" prefix      
        found = tmp.find_first_of("/");
        tmp.erase(0, found + 1);
      
        // Get the actual var name and strip it off
        found = tmp.find_first_of("/");
        varName = tmp.substr(0, found);
        // The remaining bit is the procLevelName 
        procLevelName = tmp.erase(0, found + 1);

        MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
          (sim->simController->getSchedulerP().get_rep());
      
        ApplicationInterface* appInterface =
          sim->simController->getApplicationInterface();
          
        // Simulation State Runtime stats
        if( strncmp( varname, "Processor/Runtime/", 18 ) == 0 &&
            runtimeStats.exists( varName ) )
        {
          double val;
          
          if( procLevelName == "Node/Average" )
            val = runtimeStats.getNodeAverage( varName );
          else if( procLevelName == "Node/Sum" )
            val = runtimeStats.getNodeSum( varName );
          else // if( procLevelName == "Rank" )
            val = runtimeStats.getRankValue( varName );
          
          for (int i=0; i<gd->num*gd->components; ++i)
            gd->data[i] = val;
        }
        
        // MPI Scheduler Timing stats
        else if( strncmp( varname, "Processor/MPI/", 14 ) == 0 &&
                 mpiScheduler && mpiScheduler->m_mpi_info.exists(varName) )
        {
          double val;
          
          if( procLevelName == "Node/Average" )
            val = mpiScheduler->m_mpi_info.getNodeAverage( varName );
          else if( procLevelName == "Node/Sum" )
            val = mpiScheduler->m_mpi_info.getNodeSum( varName );
          else // if( procLevelName == "Rank" )
            val = mpiScheduler->m_mpi_info.getRankValue( varName );
          
          for (int i=0; i<gd->num*gd->components; ++i)
            gd->data[i] = val;
        }

        // Application Timing stats
        else if( strncmp( varname, "Processor/Application/", 14 ) == 0 &&
                 appInterface->getApplicationStats().exists(varName) )
        {
          double val;
          
          if( procLevelName == "Node/Average" )
            val = appInterface->getApplicationStats().getNodeAverage( varName );
          else if( procLevelName == "Node/Sum" )
            val = appInterface->getApplicationStats().getNodeSum( varName );
          else // if( procLevelName == "Rank" )
            val = appInterface->getApplicationStats().getRankValue( varName );
          
          for (int i=0; i<gd->num*gd->components; ++i)
            gd->data[i] = val;
        }
      }
      else if( strncmp(varname, "Patch/Id", 8) == 0 )
      {
        // Using the machine patch mesh
        if( isMachineMeshVar )
        {
          // Here the request is for per patch data that is going on the
          // machine view.  The domain is for a rank.
          const PatchSubset* myPatches =
            lb->getPerProcessorPatchSet(gridP)->getSubset( domain );

          const unsigned int nPatches = myPatches->size();

          if( nPatches == 0 )
            return VISIT_INVALID_HANDLE;

          gd->num = nPatches;
          gd->components = 1;   
          gd->data = new double[gd->num * gd->components];

          // Get the data for each patch and store it in the grid data object.
          for (unsigned int p = 0; p < nPatches; ++p)
          {
            const Patch* patch = myPatches->get(p);

            GetLevelAndLocalPatchNumber(stepInfo,
                                        patch->getGridIndex(), level, local_patch);
            
            LevelInfo &levelInfo = stepInfo->levelInfo[level];
            PatchInfo &patchInfo = levelInfo.patchInfo[local_patch];

            gd->data[p] = patchInfo.getPatchId();
          }
        }
        // Using the simulation patch mesh
        else
        {
          gd->num = 1;
          gd->components = 1;   
          gd->data = new double[gd->num * gd->components];

          double val = patchInfo.getPatchId();
          
          for (int i=0; i<gd->num*gd->components; ++i)
            gd->data[i] = val;
        }
      }

      // Patch Task Pair Communication
      else if( strncmp(varname, "Patch/Communication", 19) == 0 )
      {
        varName = std::string(varname);
        
        gd->num = 1;        // Using the simulation patch mesh
        gd->components = 1;     
        gd->data = new double[gd->num * gd->components];

        SchedulerCommon *scheduler = dynamic_cast<SchedulerCommon*>
          (sim->simController->getSchedulerP().get_rep());

        // There may be multiple communication matrices, the index will be last.
        unsigned int index = 0;

        if( scheduler->getNumTaskGraphs() > 1 ) {
          size_t found = varName.find_last_of("/");
        
          index = stoi( varName.substr(found + 1) );
          varName = varName.substr(0, found);
        }

        std::map< std::pair< std::string, std::string >,
                  MapInfoMapper< unsigned int, CommunicationStatsEnum, unsigned int > > &comm_info =
          scheduler->getTaskGraph(index)->getDetailedTasks()->getCommInfo();
      
        // Lop off the prefix
        size_t found = std::string("Patch/Communication/").size();
        std::string statName = varName.substr(found);

        // Task pair name and the stat
        found = statName.find("/");
        std::string pairName = statName.substr(0, found);
        statName = statName.substr(found + 1);

        found = pairName.find("|");
        std::string toTaskName   = pairName.substr(0, found);
        std::string fromTaskName = pairName.substr(found + 1);

        std::pair< std::string, std::string > taskPair( toTaskName, fromTaskName );

        // Get the reduction
        found = statName.find("/");
        std::string reductionType = statName.substr(found + 1);
        statName = statName.substr(0, found);

        double val = 0;

        if( reductionType == "Size" )
          val = comm_info[taskPair].size();
        else if( reductionType == "Sum" )
          val = comm_info[taskPair].getSum( statName );
        else if( reductionType == "Average" )
          val = comm_info[taskPair].getAverage( statName );
        else if( reductionType == "Minimum" )
          val = comm_info[taskPair].getMinimum( statName );
        else if( reductionType == "Maximum" )
          val = comm_info[taskPair].getMaximum( statName );
        else if( reductionType == "StdDev" )
          val = comm_info[taskPair].getStdDev( statName );

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }
      // Patch processor rank
      else if( strncmp(varname, "Patch/Rank", 10) == 0 )
      {
        gd->num = 1;        // Using the simulation patch mesh
        gd->components = 1;     
        gd->data = new double[gd->num * gd->components];
        
        double val = sim->myworld->myRank();

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }
      // Patch processor node
      else if( strncmp(varname, "Patch/Node", 10) == 0 )
      { 
        gd->num = 1;        // Using the simulation patch mesh
        gd->components = 1;     
        gd->data = new double[gd->num * gd->components];
        
        double val = sim->myworld->myNode();

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }
      // Patch task processor 
      else if( strncmp(varname, "Patch/Tasks", 11) == 0 )
      { 
        gd->num = 1;        // Using the simulation patch mesh
        gd->components = 1;     
        gd->data = new double[gd->num * gd->components];

        MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
          (sim->simController->getSchedulerP().get_rep());
        
        varName = std::string(varname);

        size_t found = varName.find_last_of("/");
        std::string statName = varName.substr(found + 1);

        varName = varName.substr(0, found);
        found = std::string("Patch/Tasks/").size();
        std::string taskName = varName.substr(found);

        double val = mpiScheduler->m_task_info[taskName].getRankValue(statName);

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }
      else if( strncmp(varname, "Patch/Bounds/Low",  16) == 0 ||
               strncmp(varname, "Patch/Bounds/High", 17) == 0)
      {
        gd->num = 1;        // Using the simulation patch mesh
        gd->components = 3; // Bounds are vectors
        gd->data = new double[gd->num * gd->components];

        // Get the bounds for this mesh as a variable (not for the grid).
        std::string meshname = std::string(varname);
        found = meshname.find_last_of("/");
        meshname = meshname.substr(found + 1);
        
        patchInfo.getBounds(plow, phigh, meshname);
        
        int *value;        
        if (strncmp(varname, "Patch/Bounds/Low", 16) == 0 )
          value = &plow[0];
        else //if( strncmp(varname, "Patch/Bounds/High", 17) == 0)
          value = &phigh[0];

        for (int c=0; c<3; c++)
          gd->data[c] = value[c];
      }
      // Patch based node ids
      else if (strncmp(varname, "Patch/Nodes", 11) == 0 )
      {
        // Using the simulation node mesh
        gd->num = ((phigh[0] - plow[0]) *
                   (phigh[1] - plow[1]) *
                   (phigh[2] - plow[2]));
        gd->components = 3; // Bounds are vectors
        gd->data = new double[gd->num * gd->components];

        int cc = 0;
        
        for( int k=plow[2]; k<phigh[2]; ++k )
        {
          for( int j=plow[1]; j<phigh[1]; ++j )
          {
            for( int i=plow[0]; i<phigh[0]; ++i )
            {
              gd->data[cc++] = i;
              gd->data[cc++] = j;
              gd->data[cc++] = k;
            }
          }
        }
      }
      else if (strncmp(varname, "Nodes/", 6) == 0 ||
               strncmp(varname, "Ranks/", 6) == 0 )
      {
        // Using the simulation patch mesh
        if (strcmp(&(varname[6]), "Patch_Mesh") == 0)
          gd->num = 1;
        // Using the simulation cell mesh
        else /* if (strcmp(&(varname[6]), "CC_Mesh") == 0 ||
                    strcmp(&(varname[6]), "NC_Mesh") == 0 ||
                    strcmp(&(varname[6]), "SFCX_Mesh") == 0 ||
                    strcmp(&(varname[6]), "SFCY_Mesh") == 0 ||
                    strcmp(&(varname[6]), "SFCZ_Mesh") == 0) */
          gd->num = ((phigh[0] - plow[0]) *
                     (phigh[1] - plow[1]) *
                     (phigh[2] - plow[2]));
        
        gd->components = 1;
        gd->data = new double[gd->num * gd->components];

        double val = sim->myworld->myRank();

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }
    }
    // Per patch data from the warehouse going on the machine mesh.
    else if( isMachineMeshVar )
    {
      // Here the request is for per patch data that is going on the
      // machine view. The domain is for a rank.
      const PatchSubset* myPatches =
        lb->getPerProcessorPatchSet(gridP)->getSubset( domain );

      const unsigned int nPatches = myPatches->size();
      
      if( nPatches == 0 )
        return VISIT_INVALID_HANDLE;
      
      gd = new GridDataRaw;
      gd->num = nPatches;
      gd->components = 1;
      gd->data = new double[gd->num * gd->components];

      // Get the data for each patch and store it in the grid data object.
      for (unsigned int p = 0; p < nPatches; ++p)
      {
        const Patch* patch = myPatches->get(p);

        GetLevelAndLocalPatchNumber(stepInfo,
                                    patch->getGridIndex(), level, local_patch);

        GridDataRaw *tmp =
          getGridData(schedulerP, gridP, level, local_patch, varName,
                      atoi(matl.c_str()), plow, phigh, NO_EXTRA_GEOMETRY );

        gd->data[p] = tmp->data[0];

        delete tmp->data;
        delete tmp;
      }
    }
    // Patch data from the warehouse going on the simulation mesh.
    else
    {
      gd = getGridData(schedulerP, gridP, level, local_patch, varName,
                       atoi(matl.c_str()), plow, phigh,
                       (nodeCentered ? NO_EXTRA_GEOMETRY : sim->loadExtraGeometry));

      if( gd )
      {
        CheckNaNs(gd->data, gd->num*gd->components, varname, level, local_patch);
      }
    }
    
    if( gd )
    {
      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT,
                                    gd->components, gd->num, gd->data);
        
        // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
        // owns the data (VISIT_OWNER_SIM - indicates the simulation
        // owns the data). However, gd needs to be deleted.
        
        // delete gd->data;
        delete gd;
      }
    }
    else
    {
      std::stringstream msg;
      msg << "Visit libsim - " << domain << "  "
          << "Uintah variable \"" << varname << "\"  "
          << "could not be processed.";
      
      VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
    }
  }

  return varH;
}

// ****************************************************************************
//  Method: visit_SimGetDomainList
//
//  Purpose:
//      Callback for processing a domain list
//
// ****************************************************************************
visit_handle visit_SimGetDomainList(const char *meshname, void *cbdata)
{
  if( Parallel::usingMPI() )
  {
    visit_simulation_data *sim = (visit_simulation_data *)cbdata;

    if( std::string(meshname).find("Communication_") == 0 ||
        std::string(meshname).find("Machine_") == 0 )
    {
      bool global = (std::string(meshname).find("Global") != std::string::npos);
      
      // Only rank 0 return the whole of the mesh.
      if( global && sim->myworld->myRank() != 0 )
        return VISIT_INVALID_HANDLE;

      // Set the cell ids for this process.
      visit_handle domainH = VISIT_INVALID_HANDLE;
      
      if(VisIt_DomainList_alloc(&domainH) == VISIT_OKAY)
      {
        visit_handle varH = VISIT_INVALID_HANDLE;
        
        if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
        {
          const unsigned int nCells = 1;
          int cells[nCells];
          cells[0]= sim->myworld->myRank();

          VisIt_VariableData_setDataI(varH, VISIT_OWNER_COPY, 1, nCells, cells);
          
          VisIt_DomainList_setDomains(domainH, sim->myworld->nRanks(), varH);
        }
      }
      
      return domainH;
    }


    SchedulerP schedulerP = sim->simController->getSchedulerP();
    GridP      gridP      = sim->gridP;
    
    TimeStepInfo* &stepInfo = sim->stepInfo;
    
    LoadBalancer* lb = sim->simController->getLoadBalancer();
    
    int cc = 0;
    int totalPatches = 0;

    int numLevels = stepInfo->levelInfo.size();

    // Storage for the patch ids that belong to this processs.
    std::vector<int> localPatches;
    
    // Get level info
    for (int l=0; l<numLevels; ++l)
    {
      LevelP level = gridP->getLevel(l);
      
      int numPatches = level->numPatches();
      
      // Resize to fit the total number of patches found so far.
      totalPatches += numPatches;
      localPatches.resize( totalPatches );
      
      // Get the patch info
      for (int p=0; p<numPatches; ++p)
      {
        const Patch* patch = level->getPatch(p);

        // Record the patch id if it belongs to this process.
        if( sim->myworld->myRank() ==
            lb->getPatchwiseProcessorAssignment(patch) )
          localPatches[cc++] = GetGlobalDomainNumber(stepInfo, l, p);
      }
    }

    // Resize to fit the actual number of patch ids stored.
    localPatches.resize( cc );
    
    // Set the patch ids for this process.
    visit_handle domainH = VISIT_INVALID_HANDLE;
    
    if(VisIt_DomainList_alloc(&domainH) == VISIT_OKAY)
    {
      visit_handle varH = VISIT_INVALID_HANDLE;
      
      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataI(varH, VISIT_OWNER_COPY, 1,
                                    localPatches.size(), localPatches.data());
        
        VisIt_DomainList_setDomains(domainH, totalPatches, varH);
      }
    }

    return domainH;
  }
  else
    return VISIT_INVALID_HANDLE;
}



// ****************************************************************************
//  Method: addRectilinearMesh
//
//  Purpose:
//      Add rectilinear mesh.
//
// ****************************************************************************
void
addRectilinearMesh( visit_handle md, std::set<std::string> &meshes_added,
                    std::string meshName, visit_simulation_data *sim )
{
  TimeStepInfo* &stepInfo = sim->stepInfo;
  GridP          gridP    = sim->gridP; 

  if (meshes_added.find(meshName) == meshes_added.end())
  {
    // Mesh meta data
    visit_handle mmd = VISIT_INVALID_HANDLE;
    
    // Set the first mesh’s properties.
    if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
    {
      int numLevels = stepInfo->levelInfo.size();

      int totalPatches = 0;
      for (int i=0; i<numLevels; ++i)
        totalPatches += stepInfo->levelInfo[i].patchInfo.size();

      double box_min[3], box_max[3];
      stepInfo->levelInfo[0].getExtents(box_min, box_max);

      int low[3], high[3];
      stepInfo->levelInfo[0].getBounds(low, high, meshName);

      int logical[3];
      for (int i=0; i<3; ++i)
        logical[i] = high[i] - low[i];
      
      // Set the mesh’s properties.
      VisIt_MeshMetaData_setName(mmd, meshName.c_str());
      if( sim->simController->getApplicationInterface()->isAMR() )
        VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_AMR);
      else
        VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_RECTILINEAR);
      
      VisIt_MeshMetaData_setTopologicalDimension(mmd, 3);
      VisIt_MeshMetaData_setSpatialDimension(mmd, 3);
      
      VisIt_MeshMetaData_setNumDomains(mmd, totalPatches);
      VisIt_MeshMetaData_setDomainTitle(mmd, "patches");
      VisIt_MeshMetaData_setDomainPieceName(mmd, "patch");
      VisIt_MeshMetaData_setNumGroups(mmd, numLevels);
      VisIt_MeshMetaData_setGroupTitle(mmd, "levels");
      VisIt_MeshMetaData_setGroupPieceName(mmd, "level");
            
      for (int p=0; p<totalPatches; ++p)
      {
        char tmpName[64];
        int level, local_patch;
        
        GetLevelAndLocalPatchNumber(stepInfo, p, level, local_patch);
        
        LevelP levelP = gridP->getLevel(level);
        // const Patch* patch = levelP->getPatch(local_patch);
        
        sprintf(tmpName,"level%d, patch%d", level, local_patch);
        
        VisIt_MeshMetaData_addGroupId(mmd, level);
        VisIt_MeshMetaData_addDomainName(mmd, tmpName);
      }
      
      // ARS - FIXME
      // VisIt_MeshMetaData_setContainsExteriorBoundaryGhosts(mmd, false);
      
      VisIt_MeshMetaData_setHasSpatialExtents(mmd, 1);
      
      double extents[6] = { box_min[0], box_max[0],
                            box_min[1], box_max[1],
                            box_min[2], box_max[2] };
      
      VisIt_MeshMetaData_setSpatialExtents(mmd, extents);
      
      VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
      VisIt_MeshMetaData_setLogicalBounds(mmd, logical);
      
      VisIt_SimulationMetaData_addMesh(md, mmd);
      
      addMeshNodeRankSIL( md, meshName, sim );

      meshes_added.insert(meshName);
      
      // std::cerr << "Calculating SimGetMetaData for "
      //        << mesh_for_this_var.c_str() << " mesh (" << mmd << ")." 
      //        << std::endl;          
    }
  }
}


// ****************************************************************************
//  Method: addParticleMesh
//
//  Purpose:
//      Add a particle mesh
//
// ****************************************************************************
void
addParticleMesh( visit_handle md, std::set<std::string> &meshes_added,
                 std::string meshName, visit_simulation_data *sim )
{
  TimeStepInfo* &stepInfo = sim->stepInfo;
  GridP          gridP    = sim->gridP; 

  if (meshes_added.find(meshName) == meshes_added.end())
  {
    // Mesh meta data
    visit_handle mmd = VISIT_INVALID_HANDLE;
    
    // Set the first mesh’s properties.
    if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
    {
      int numLevels = stepInfo->levelInfo.size();

      int totalPatches = 0;
      for (int i=0; i<numLevels; ++i)
        totalPatches += stepInfo->levelInfo[i].patchInfo.size();

      double box_min[3], box_max[3];
      stepInfo->levelInfo[0].getExtents(box_min, box_max);

      int low[3], high[3];
      stepInfo->levelInfo[0].getBounds(low, high, meshName);

      int logical[3];
      for (int i=0; i<3; ++i)
        logical[i] = high[i] - low[i];
      
      // Set the mesh’s properties.
      VisIt_MeshMetaData_setName(mmd, meshName.c_str());
      VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_POINT);
      VisIt_MeshMetaData_setTopologicalDimension(mmd, 0);
      VisIt_MeshMetaData_setSpatialDimension(mmd, 3);
        
      VisIt_MeshMetaData_setNumDomains(mmd, totalPatches);
      VisIt_MeshMetaData_setDomainTitle(mmd, "patches");
      VisIt_MeshMetaData_setDomainPieceName(mmd, "patch");
      VisIt_MeshMetaData_setNumGroups(mmd, numLevels);
      VisIt_MeshMetaData_setGroupTitle(mmd, "levels");
      VisIt_MeshMetaData_setGroupPieceName(mmd, "level");

      for (int k=0; k<totalPatches; ++k)
      {
        char tmpName[64];
        int level, local_patch;
        
        GetLevelAndLocalPatchNumber(stepInfo, k, level, local_patch);
        sprintf(tmpName,"level%d, patch%d", level, local_patch);
        
        VisIt_MeshMetaData_addGroupId(mmd, level);
        VisIt_MeshMetaData_addDomainName(mmd, tmpName);
      }
      
      VisIt_MeshMetaData_setHasSpatialExtents(mmd, 1);
      
      double extents[6] = { box_min[0], box_max[0],
                            box_min[1], box_max[1],
                            box_min[2], box_max[2] };
      
      VisIt_MeshMetaData_setSpatialExtents(mmd, extents);
            
      VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
      VisIt_MeshMetaData_setLogicalBounds(mmd, logical);
      
      VisIt_SimulationMetaData_addMesh(md, mmd);
      
      addMeshNodeRankSIL( md, meshName, sim );
      
      meshes_added.insert(meshName);
    }
  }
}

// ****************************************************************************
//  Method: addMeshNodeRankSIL
//
//  Purpose:
//      Add a SIL for subsettng via the nodes and ranks.
//
// ****************************************************************************
void addMeshNodeRankSIL( visit_handle md, std::string meshName, visit_simulation_data *sim )
{
  // Add a SIL for subsettng via the nodes and ranks.
  visit_handle smd_node = VISIT_INVALID_HANDLE;
  visit_handle smd_rank = VISIT_INVALID_HANDLE;

  if(VisIt_VariableMetaData_alloc(&smd_node) == VISIT_OKAY &&
     VisIt_VariableMetaData_alloc(&smd_rank) == VISIT_OKAY)
  {
    const unsigned int nRanks = sim->myworld->nRanks();
    const unsigned int nNodes = sim->myworld->nNodes();

    int rank_enum_id[nRanks];
    int node_enum_id[nNodes];
              
    // Node
    std::string enum_name_node = std::string("Nodes/") + meshName;
    VisIt_VariableMetaData_setName(smd_node, enum_name_node.c_str());
    VisIt_VariableMetaData_setMeshName(smd_node, meshName.c_str());
    VisIt_VariableMetaData_setCentering(smd_node, VISIT_VARCENTERING_ZONE);
    VisIt_VariableMetaData_setType(smd_node, VISIT_VARTYPE_SCALAR);
    VisIt_VariableMetaData_setNumComponents(smd_node, 1);

    VisIt_VariableMetaData_setEnumerationType(smd_node, VISIT_ENUMTYPE_BY_VALUE);
    VisIt_VariableMetaData_setHideFromGUI(smd_node, true);

    // Rank
    std::string enum_name_rank = std::string("Ranks/") + meshName;
    VisIt_VariableMetaData_setName(smd_rank, enum_name_rank.c_str());
    VisIt_VariableMetaData_setMeshName(smd_rank, meshName.c_str());
    VisIt_VariableMetaData_setCentering(smd_rank, VISIT_VARCENTERING_ZONE);
    VisIt_VariableMetaData_setType(smd_rank, VISIT_VARTYPE_SCALAR);
    VisIt_VariableMetaData_setNumComponents(smd_rank, 1);

    VisIt_VariableMetaData_setEnumerationType(smd_rank, VISIT_ENUMTYPE_BY_VALUE);
    VisIt_VariableMetaData_setHideFromGUI(smd_rank, true);

    for( unsigned int r=0; r<nRanks; ++r ) {
      char msg_node[12];
      sprintf( msg_node, "Rank_%04d", r );
      int index;
      VisIt_VariableMetaData_addEnumNameValue( smd_node, msg_node, r, &index );
      rank_enum_id[r] = index;
      
      char msg_rank[36];
      sprintf( msg_rank, "%s, Rank_%04d", sim->myworld->getNodeNameFromRank( r ).c_str(), r );
      VisIt_VariableMetaData_addEnumNameValue( smd_rank, msg_rank, r, &index );
    }
    
    if( nNodes > 1 ) {
      for( unsigned int n=0; n<nNodes; ++n ) {
        int index;
        VisIt_VariableMetaData_addEnumNameValue( smd_node, sim->myworld->getNodeName( n ).c_str(), nRanks+n, &index );
        node_enum_id[n] = index;
      }
      
      for( unsigned int r=0; r<nRanks; ++r ) {
        unsigned int n = sim->myworld->getNodeIndexFromRank( r );
        
        VisIt_VariableMetaData_addEnumGraphEdge(smd_node, node_enum_id[n], rank_enum_id[r], "Ranks" );
      }
    }
    
    // if( nSwitches > 1 ) {
    //   for( unsigned int s=0; s<nSwitches; ++s ) {
    //     int index;
    //     VisIt_VariableMetaData_addEnumNameValue( smd_switch, sim->myworld->getSwitchName( n ).c_str(), nRanks+nNodes+s, &index );
    //     switch_enum_id[n] = index;
    //   }
      
    //   for( unsigned int n=0; n<nNodes; ++n ) {
    //     unsigned int s = sim->myworld->getSwitchIndexFromNode( r );
        
    //     VisIt_VariableMetaData_addEnumGraphEdge(smd_switch, switch_enum_id[s], node_enum_id[r], "Switches" );
    //   }
    // }
    
    VisIt_SimulationMetaData_addVariable(md, smd_node);
    VisIt_SimulationMetaData_addVariable(md, smd_rank);
  }
}

// ****************************************************************************
//  Method: AddMeshVariable
//
//  Purpose:
//      Add a variable to the meta data.
//
// ****************************************************************************
void addMeshVariable( visit_handle md, std::set<std::string> &mesh_vars_added,
                      std::string varName, std::string varType,
                      std::string meshName, VisIt_VarCentering cent )
{
  // Make sure a variable is added only once.
  if (mesh_vars_added.find(meshName+varName) == mesh_vars_added.end())
  {
    mesh_vars_added.insert(meshName+varName);
    
    int nComponents;
    int type;
    
    // 3 -> point/vector dimension
    if (varType.find("Point") != std::string::npos ||
        varType.find("Vector") != std::string::npos ||
        varType.find("IntVector") != std::string::npos)
    {
      type = VISIT_VARTYPE_VECTOR;
      nComponents = 3;
    }
    // 9 -> tensor 
    else if (varType.find("Matrix3") != std::string::npos)
    {
      type = VISIT_VARTYPE_TENSOR;
      nComponents = 9;
    }
    // 7 -> vector
    else if (varType.find("Stencil7") != std::string::npos)
    {
      type = VISIT_VARTYPE_VECTOR;
      nComponents = 7;
    }
    // 4 -> vector
    else if (varType.find("Stencil4") != std::string::npos)
    {
      type = VISIT_VARTYPE_VECTOR;
      nComponents = 4;
    }
    // 1 -> scalar
    else
    {
      type = VISIT_VARTYPE_SCALAR;
      nComponents = 1;
    }
    // else
    // {
    //   std::stringstream msg;
    //   msg << "Visit libsim - "
    //    << "Uintah variable \"" << varName << "\"  "
    //    << "has an unknown grid variable type \""
    //    << varType << "\"";
      
    //   VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
      
    //   return;
    // }

    visit_handle vmd = VISIT_INVALID_HANDLE;

    if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
    {
      VisIt_VariableMetaData_setName(vmd, varName.c_str());
      VisIt_VariableMetaData_setMeshName(vmd, meshName.c_str());
      VisIt_VariableMetaData_setCentering(vmd, cent);
      
      VisIt_VariableMetaData_setType(vmd, type);
      VisIt_VariableMetaData_setNumComponents(vmd, nComponents);
      
      // ARS - FIXME
      //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
      VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
      
      VisIt_SimulationMetaData_addVariable(md, vmd);
    }
  }
}

} // End namespace Uintah
