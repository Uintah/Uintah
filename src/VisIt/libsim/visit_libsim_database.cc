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
#include "visit_libsim_database.h"
#include "visit_libsim_callbacks.h"
#include "visit_libsim_customUI.h"

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Output.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/VarLabel.h>


#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>

#include <VisIt/interfaces/warehouseInterface.h>

#include <vector>
#include <stdio.h>

namespace Uintah {

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

  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();
  SchedulerP     schedulerP = sim->simController->getSchedulerP();
  LoadBalancer * lb         = sim->simController->getLoadBalancer();
  Output       * output     = sim->simController->getOutput();
  GridP          gridP      = sim->gridP;
      
  if( !schedulerP.get_rep() || !gridP.get_rep() )
  {
    return VISIT_INVALID_HANDLE;
  }

  // int timestate = sim->cycle;
  
  LoadExtra loadExtraElements = (LoadExtra) sim->loadExtraElements;
  // bool &forceMeshReload = sim->forceMeshReload;

  if( sim->stepInfo )
    delete sim->stepInfo;
  
  sim->stepInfo = getTimeStepInfo(schedulerP, gridP, loadExtraElements);

  TimeStepInfo* &stepInfo = sim->stepInfo;
  
  visit_handle md = VISIT_INVALID_HANDLE;

  /* Create metadata with no variables. */
  if(VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY)
  {
    /* Set the simulation state. */

    /* NOTE visit_SimGetMetaData is called as a results of calling
       visit_CheckState which calls VisItTimeStepChanged at this point
       the sim->runMode will always be VISIT_SIMMODE_RUNNING. */

    // To get the "Simulation status" in the Simulation window correct
    // one needs to call VisItUI_setValueS("SIMULATION_MODE", "Stopped", 1);
    if(sim->runMode == VISIT_SIMMODE_FINISHED ||
       sim->runMode == VISIT_SIMMODE_STOPPED ||
       sim->runMode == VISIT_SIMMODE_STEP)
      VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_STOPPED);
    else if(sim->runMode == VISIT_SIMMODE_RUNNING)
      VisIt_SimulationMetaData_setMode(md, VISIT_SIMMODE_RUNNING);

    VisIt_SimulationMetaData_setCycleTime(md, sim->cycle, sim->time);

    int numLevels = stepInfo->levelInfo.size();

    int totalPatches = 0;
    for (int i=0; i<numLevels; ++i)
      totalPatches += stepInfo->levelInfo[i].patchInfo.size();
    
    // compute the bounding box of the mesh from the grid indices of
    // level 0
    LevelInfo &levelInfo = stepInfo->levelInfo[0];

    unsigned int addMachineData = (sim->switchNodeList.size() &&
                                   (int) sim->switchIndex != -1 &&
                                   (int) sim->nodeIndex   != -1 );

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

    // Get CC bounds
    int low[3], high[3];
    levelInfo.getBounds(low, high, "CC_Mesh");

    // This can be done once for everything because the spatial range is
    // the same for all meshes
    double box_min[3] = { levelInfo.anchor[0] + low[0] * levelInfo.spacing[0],
                          levelInfo.anchor[1] + low[1] * levelInfo.spacing[1],
                          levelInfo.anchor[2] + low[2] * levelInfo.spacing[2] };

    double box_max[3] = { levelInfo.anchor[0] + high[0] * levelInfo.spacing[0],
                          levelInfo.anchor[1] + high[1] * levelInfo.spacing[1],
                          levelInfo.anchor[2] + high[2] * levelInfo.spacing[2] };

    // debug5 << "box_min/max=["
    //     << box_min[0] << "," << box_min[1] << ","
    //     << box_min[2] << "] to ["
    //     << box_max[0] << "," << box_max[1] << ","
    //     << box_max[2] << "]" << std::endl;

    // int logical[3];

    // for (int i=0; i<3; ++i)
    //   logical[i] = high[i] - low[i];

    // debug5 << "logical: " << logical[0] << ", " << logical[1] << ", "
    //     << logical[2] << std::endl;

    int numVars = stepInfo->varInfo.size();

    // Do a hasty search for a node or cell mesh.
    for (int i=0; i<numVars; ++i)
    {
      if (stepInfo->varInfo[i].type.find("ParticleVariable") == std::string::npos)
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
    }

    // Loop through all vars and add them to the meta data.
    for (int i=0; i<numVars; ++i)
    {
      bool isPerPatchVar = false;
      
      if (stepInfo->varInfo[i].type.find("ParticleVariable") == std::string::npos)
      {
        std::string varname = stepInfo->varInfo[i].name;
        std::string vartype = stepInfo->varInfo[i].type;
//      int matsize         = stepInfo->varInfo[i].materials.size();

        std::string mesh_for_this_var;
        VisIt_VarCentering cent;

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
        else if (vartype.find("ReductionVariable") != std::string::npos)
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
                << "has an unknown variable type \""
                << vartype << "\"";
            
            VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
          }

          continue;
        }

        if (meshes_added.find(mesh_for_this_var) == meshes_added.end())
        {
          std::string varname = stepInfo->varInfo[i].name;
          std::string vartype = stepInfo->varInfo[i].type;
          
          // Mesh meta data
          visit_handle mmd = VISIT_INVALID_HANDLE;
          
          /* Set the first mesh’s properties.*/
          if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
          {
            /* Set the mesh’s properties.*/
            VisIt_MeshMetaData_setName(mmd, mesh_for_this_var.c_str());
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

            // VisIt_MeshMetaData_setNumDomains(mmd, sim->myworld->nRanks());
            // VisIt_MeshMetaData_setDomainTitle(mmd, "ranks");
            // VisIt_MeshMetaData_setDomainPieceName(mmd, "rank");
            // VisIt_MeshMetaData_setNumGroups(mmd, sim->myworld->nNodes());
            // VisIt_MeshMetaData_setGroupTitle(mmd, "nodes");
            // VisIt_MeshMetaData_setGroupPieceName(mmd, "node");
            
            for (int p=0; p<totalPatches; ++p)
            {
              char tmpName[64];
              int level, local_patch;
      
              GetLevelAndLocalPatchNumber(stepInfo, p, level, local_patch);

              LevelP levelP = gridP->getLevel(level);
              const Patch* patch = levelP->getPatch(local_patch);
              
              int rank = lb->getPatchwiseProcessorAssignment(patch);
              int node = sim->myworld->getNodeFromRank(rank);

              sprintf(tmpName,"level%d, patch%d, node%d, rank%d",
                      level, local_patch, node, rank);

              // sprintf(tmpName,"node%d, rank%d, level%d, patch%d",
              //         level, local_patch,
              //         sim->myworld->getNodeFromRank(rank), rank);

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

            // ARS - FIXME
            // VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
            // VisIt_MeshMetaData_logicalBounds(mmd, logical[0]);

            VisIt_SimulationMetaData_addMesh(md, mmd);

            // std::cerr << "Calculating SimGetMetaData for "
            //        << mesh_for_this_var.c_str() << " mesh (" << mmd << ")." 
            //        << std::endl;          
          }

          meshes_added.insert(mesh_for_this_var);
        }

        std::string mesh_name[2] = {mesh_for_this_var,
                                    ("Machine_" + sim->hostName + "/Patch") };

        std::string mesh_layout[2] = {"/Sim", "/"+sim->hostName};

        // If there is a machine layout then the patch data can be
        // placed on the simulation and machine mesh.
        for( int k=0; k<1+int(isPerPatchVar && addMachineData); ++k )
        {
          // Add mesh vars
          int numMaterials = stepInfo->varInfo[i].materials.size();
          
          if( numMaterials == 0 )
          {
            std::string newVarname = varname;
            
            if( isPerPatchVar )
            {
              newVarname = "Patch/" + newVarname;
              
              if( addMachineData )
                newVarname += mesh_layout[k];
            }
            else
              newVarname.append("/0");
            
            if (mesh_vars_added.find(mesh_name[k]+newVarname) ==
                mesh_vars_added.end())
            {
              mesh_vars_added.insert(mesh_name[k]+newVarname);
              
              visit_handle vmd = VISIT_INVALID_HANDLE;
              
              if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
              {
                VisIt_VariableMetaData_setName(vmd, newVarname.c_str());
                VisIt_VariableMetaData_setMeshName(vmd, mesh_name[k].c_str());
                VisIt_VariableMetaData_setCentering(vmd, cent);
                
                // 3 -> vector dimension
                if (vartype.find("Vector") != std::string::npos)
                {
                  VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                  VisIt_VariableMetaData_setNumComponents(vmd, 3);
                }
                // 9 -> tensor 
                else if (vartype.find("Matrix3") != std::string::npos)
                {
                  VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_TENSOR);
                  VisIt_VariableMetaData_setNumComponents(vmd, 9);
                }
                // 7 -> vector
                else if (vartype.find("Stencil7") != std::string::npos)
                {
                  VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                  VisIt_VariableMetaData_setNumComponents(vmd, 7);
                }
                // 4 -> vector
                else if (vartype.find("Stencil4") != std::string::npos)
                {
                  VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                  VisIt_VariableMetaData_setNumComponents(vmd, 4);
                }
                // scalar
                else // if (vartype.find("Scalar") != std::string::npos)
                {
                  VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                  VisIt_VariableMetaData_setNumComponents(vmd, 1);
                }
              // else
              // {
              //   std::stringstream msg;
              //   msg << "Visit libsim - "
              //       << "Uintah variable \"" << varname << "\"  "
              //       << "has an unknown variable type \""
              //       << vartype << "\"";
              
              //   VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
              //   continue;
              // }

                VisIt_SimulationMetaData_addVariable(md, vmd);
              }
            }
          }
          else
          {
            for (int j=0; j<numMaterials; ++j)
            {
              std::string newVarname = varname;
              
              if( isPerPatchVar )
              {
                newVarname = "Patch/" + newVarname;
                
                if( addMachineData )
                  newVarname += mesh_layout[k];
              }
              else
              {
                char buffer[128];
                sprintf(buffer, "%d", stepInfo->varInfo[i].materials[j]);
                newVarname.append("/");
                newVarname.append(buffer);
              }
              
              if (mesh_vars_added.find(mesh_name[k]+newVarname) ==
                  mesh_vars_added.end())
              {
                mesh_vars_added.insert(mesh_name[k]+newVarname);
                
                visit_handle vmd = VISIT_INVALID_HANDLE;
                
                if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
                {
                  VisIt_VariableMetaData_setName(vmd, newVarname.c_str());
                  VisIt_VariableMetaData_setMeshName(vmd, mesh_name[k].c_str());
                  VisIt_VariableMetaData_setCentering(vmd, cent);
                  
                  // 3 -> vector dimension
                  if (vartype.find("Vector") != std::string::npos)
                  {
                    VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                    VisIt_VariableMetaData_setNumComponents(vmd, 3);
                  }
                  // 9 -> tensor 
                  else if (vartype.find("Matrix3") != std::string::npos)
                  {
                    VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_TENSOR);
                    VisIt_VariableMetaData_setNumComponents(vmd, 9);
                  }
                  // 7 -> vector
                  else if (vartype.find("Stencil7") != std::string::npos)
                  {
                    VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                    VisIt_VariableMetaData_setNumComponents(vmd, 7);
                  }
                  // 4 -> vector
                  else if (vartype.find("Stencil4") != std::string::npos)
                  {
                    VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                    VisIt_VariableMetaData_setNumComponents(vmd, 4);
                  }
                  // scalar
                  else // if (vartype.find("Scalar") != std::string::npos)
                  {
                    VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                    VisIt_VariableMetaData_setNumComponents(vmd, 1);
                  }
                // else
                // {
                //   std::stringstream msg;
                //   msg << "Visit libsim - "
                //       << "Uintah variable \"" << varname << "\"  "
                //       << "has an unknown variable type \""
                //       << vartype << "\"";
                
                //   VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
                //   continue;
                // }
                
                  VisIt_SimulationMetaData_addVariable(md, vmd);
                }
              }
            }
          }
        }
      }   
    }

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
      // Mesh meta data
      visit_handle mmd = VISIT_INVALID_HANDLE;
      
      /* Set the first mesh’s properties.*/
      if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
      {
        /* Set the mesh’s properties.*/
        VisIt_MeshMetaData_setName(mmd, "Patch_Mesh");
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

        for (int k=0; k<totalPatches; ++k)
        {
          char tmpName[64];
          int level, local_patch;
      
          GetLevelAndLocalPatchNumber(stepInfo, k, level, local_patch);
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
        
        // ARS - FIXME
        // VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
        // VisIt_MeshMetaData_logicalBounds(mmd, logical[0]);
        
        VisIt_SimulationMetaData_addMesh(md, mmd);
      }
      
      visit_handle vmd = VISIT_INVALID_HANDLE;

      int cent = VISIT_VARCENTERING_ZONE;

      std::string mesh_for_this_var = "Patch_Mesh";

      std::string mesh_name[2] = {mesh_for_this_var,
                                  ("Machine_" + sim->hostName + "/Patch") };

      std::string mesh_layout[2] = {"/Sim", "/"+sim->hostName};

      const char *patch_names[3] =
        { "Patch/Id", "Patch/ProcRank", "Patch/ProcNode" };

      // If there is a machine layout then the performance data can be
      // placed on the simulation and machine mesh.

      // Patch id, rank, and node on both the sim and machine 
      for( unsigned k=0; k<1+addMachineData; ++k )
      {
        for( unsigned int i=0; i<3; ++i )
        {
          if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
          {
            std::string tmp_name = patch_names[i];

            if( addMachineData )
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
        }
      }

      // Bounds for node and cell based patch variables on just the sim
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
      
      mesh_name[1] = "Machine_" + sim->hostName + "/Local";

      const unsigned int nProcLevels = 3;
      std::string proc_level[nProcLevels] =
        {"/Rank", "/Node/Average", "/Node/Sum"};

      // If there is a machine layout then the performance data can be
      // placed on the simulation and machine mesh.

      // Runtime data on both the sim and machine.
      for( unsigned k=0; k<1+addMachineData; ++k )
      {
        // There is performance on a per node and per core basis.
        for( unsigned j=0; j<nProcLevels; ++j )
        {
          unsigned int nStats = sim->simController->getRuntimeStats().size();
          
          // Add in the processor runtime stats.
          for( unsigned int i=0; i<nStats; ++i )
          {
            visit_handle vmd = VISIT_INVALID_HANDLE;
            
            if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
            {
              std::string tmp_name = std::string("Processor/Runtime/") +
                sim->simController->getRuntimeStats().getName( i ) +
                proc_level[j];
              
              // If there is a machine layout then the performance
              // data can be placed on the the simulation and machine
              // mesh.
              if( addMachineData )
                tmp_name += mesh_layout[k];
              
              std::string units =
                sim->simController->getRuntimeStats().getUnits( i );
              
              VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
              VisIt_VariableMetaData_setMeshName(vmd, mesh_name[k].c_str());
              VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
              VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
              VisIt_VariableMetaData_setNumComponents(vmd, 1);
              VisIt_VariableMetaData_setUnits(vmd, units.c_str());
              
              // ARS - FIXME
              //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
              VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
              VisIt_SimulationMetaData_addVariable(md, vmd);
            }
          }
          
          MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
            (sim->simController->getSchedulerP().get_rep());
          
          // Add in the mpi runtime stats.
          if( mpiScheduler )
          {
            for( unsigned int i=0; i<mpiScheduler->mpi_info_.size(); ++i )
            {
              visit_handle vmd = VISIT_INVALID_HANDLE;
              
              if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
              {
                std::string tmp_name = std::string("Processor/MPI/") + 
                  mpiScheduler->mpi_info_.getName( i ) + proc_level[j];
                
                if( addMachineData )
                  tmp_name += mesh_layout[k];
                
                std::string units = mpiScheduler->mpi_info_.getUnits( i );
                
                VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
                VisIt_VariableMetaData_setMeshName(vmd, mesh_name[k].c_str());
                VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
                VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                VisIt_VariableMetaData_setNumComponents(vmd, 1);
                VisIt_VariableMetaData_setUnits(vmd, units.c_str());
                
                // ARS - FIXME
                //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
                VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
                VisIt_SimulationMetaData_addVariable(md, vmd);
              }
            }
          }

          ApplicationInterface* appInterface =
            sim->simController->getApplicationInterface();
          
          nStats = appInterface->getApplicationStats().size();
          
          // Add in the application stats.
          for( unsigned int i=0; i<nStats; ++i )
          {
            visit_handle vmd = VISIT_INVALID_HANDLE;
            
            if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
            {
              std::string tmp_name = std::string("Processor/Application/") +
                appInterface->getApplicationStats().getName( i ) +
                proc_level[j];
              
              if( addMachineData )
                tmp_name += mesh_layout[k];
                
              std::string units =
                appInterface->getApplicationStats().getUnits( i );
              
              VisIt_VariableMetaData_setName(vmd, tmp_name.c_str());
              VisIt_VariableMetaData_setMeshName(vmd, mesh_name[k].c_str());
              VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
              VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
              VisIt_VariableMetaData_setNumComponents(vmd, 1);
              VisIt_VariableMetaData_setUnits(vmd, units.c_str());
              
              // ARS - FIXME
              //      VisIt_VariableMetaData_setHasDataExtents(vmd, false);
              VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
              VisIt_SimulationMetaData_addVariable(md, vmd);
            }
          }          
        }
      }
    }

    // Nothing needs to be modifed for particle data, as they exist only
    // on a single level
    for (int i=0; i<numVars; ++i)
    {
      if (stepInfo->varInfo[i].type.find("ParticleVariable") != std::string::npos)
      {
        std::string varname = stepInfo->varInfo[i].name;
        std::string vartype = stepInfo->varInfo[i].type;
        
        // j=-1 -> all materials (*)
        int numMaterials = stepInfo->varInfo[i].materials.size();

        for (int j=-1; j<numMaterials; ++j)
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

              // ARS - FIXME
              // VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
              // VisIt_MeshMetaData_seteLogicalBounds(mmd, logical[0]);

              VisIt_SimulationMetaData_addMesh(md, mmd);
            }

            meshes_added.insert(mesh_for_this_var);
          }

          if (mesh_vars_added.find(mesh_for_this_var+newVarname) ==
              mesh_vars_added.end())
          {
            mesh_vars_added.insert(mesh_for_this_var+newVarname);
            
            VisIt_VarCentering cent = VISIT_VARCENTERING_NODE;

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
              }
              // 9 -> tensor 
              else if (vartype.find("Matrix3") != std::string::npos)
              {
                VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_TENSOR);
                VisIt_VariableMetaData_setNumComponents(vmd, 9);
              }
              // 7 -> vector
              else if (vartype.find("Stencil7") != std::string::npos)
              {
                VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                VisIt_VariableMetaData_setNumComponents(vmd, 7);
              }
              // 4 -> vector
              else if (vartype.find("Stencil4") != std::string::npos)
              {
                VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_VECTOR);
                VisIt_VariableMetaData_setNumComponents(vmd, 4);
              }
              // scalar
              else if (vartype.find("Scalar") != std::string::npos)
              {
                VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                VisIt_VariableMetaData_setNumComponents(vmd, 1);
              }
              else
              {
                std::stringstream msg;
                msg << "Visit libsim - "
                    << "Uintah variable \"" << varname << "\"  "
                    << "has an unknown variable type \""
                    << vartype << "\"";
            
                VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
                continue;
              }
              
              VisIt_SimulationMetaData_addVariable(md, vmd);
            }
          }
        }
      }   
    }

    // If there is a machine layout then the performance data can be
    // placed on the machine mesh.
    if( addMachineData )
    {
      // If there is a machine layout then there is a global, local,
      // and patch machine mesh. The global is all of the nodes and
      // cores. The local is the nodes and cores actually used. The
      // patch is patches on the each core.
      for( unsigned int i=0; i<3; ++i )
      {
        visit_handle mmd = VISIT_INVALID_HANDLE;
        
        /* Set the first mesh’s properties.*/
        if(VisIt_MeshMetaData_alloc(&mmd) == VISIT_OKAY)
        {
          /* Set the mesh’s properties.*/
          std::string meshName = "Machine_" + sim->hostName;

          if( i == 0 ) // Global mesh
            VisIt_MeshMetaData_setName(mmd, (meshName + "/Global").c_str());
          else if( i == 1 ) // Local mesh
            VisIt_MeshMetaData_setName(mmd, (meshName + "/Local").c_str());
          else if( i == 2 ) // Patch mesh
            VisIt_MeshMetaData_setName(mmd, (meshName + "/Patch").c_str());

          VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_UNSTRUCTURED);
          VisIt_MeshMetaData_setTopologicalDimension(mmd, 2);
          VisIt_MeshMetaData_setSpatialDimension(mmd, 2);
          VisIt_MeshMetaData_setXLabel(mmd, "Switches");
          VisIt_MeshMetaData_setYLabel(mmd, "Nodes");

          // For the global view there is only one domain. For the
          // local and patch view there is one domain per rank.
          VisIt_MeshMetaData_setNumDomains(mmd,
                             (i == 0 ? 1 : sim->myworld->nRanks()));

          if( i > 0 )
          {
            VisIt_MeshMetaData_setDomainTitle(mmd, "ranks");
            VisIt_MeshMetaData_setDomainPieceName(mmd, "rank");
            VisIt_MeshMetaData_setNumGroups(mmd, sim->myworld->nNodes());
            VisIt_MeshMetaData_setGroupTitle(mmd, "nodes");
            VisIt_MeshMetaData_setGroupPieceName(mmd, "node");

            for (int rank=0; rank<sim->myworld->nRanks(); ++rank)
            {
              int node = sim->myworld->getNodeFromRank( rank );
              char tmpName[64];       
              sprintf(tmpName,"node%d, rank%d", node, rank);
              
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

          // ARS - FIXME
          // VisIt_MeshMetaData_setHasLogicalBounds(mmd, 1);
          // VisIt_MeshMetaData_logicalBounds(mmd, logical[0]);
        
          VisIt_SimulationMetaData_addMesh(md, mmd);
        }
      }

      const int nVars = 4;
      
      std::string vars[nVars] = {"Node/Number",
                                 "MPI/Comm/Node",
                                 "MPI/Comm/Rank",
                                 "MPI/Rank"};

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
          if( vars[i] ==  "Number" )
            VisIt_VariableMetaData_setUnits(vmd, sim->hostName.c_str());
          else
            VisIt_VariableMetaData_setUnits(vmd, "");
              
          // ARS - FIXME
          // VisIt_VariableMetaData_setHasDataExtents(vmd, false);
          VisIt_VariableMetaData_setTreatAsASCII(vmd, false);
          VisIt_SimulationMetaData_addVariable(md, vmd);
        }
      }
    }
    
    // ARS - FIXME
    // md->AddGroupInformation(numLevels, totalPatches, groupIds);
    // md->AddDefaultSILRestrictionDescription(std::string("!TurnOnAll"));

    /* Add some commands. */
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

  // int timestate = sim->cycle;  
  // LoadExtra loadExtraElements = (LoadExtra) sim->loadExtraElements;
  // bool &forceMeshReload = sim->forceMeshReload;
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

      int extents[6] = { plow[0], phigh[0],
                         plow[1], phigh[1],
                         plow[2], phigh[2] };
        
      // std::cerr << "\trdb->SetIndicesForPatch(" << patch << ","
      //        << my_level << ", " << local_patch << ", <"
      //        << extents[0] << "," << extents[2] << "," << extents[4]
      //        << "> to <"
      //        << extents[1] << "," << extents[3] << "," << extents[5] << ">)"
      //        << std::endl;

      VisIt_DomainBoundaries_set_amrIndices(rdb, patch, my_level, extents);
      //      VisIt_DomainBoundaries_finish(rdb, patch);
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

  // int timestate = sim->cycle;  
  // LoadExtra loadExtraElements = (LoadExtra) sim->loadExtraElements;
  // bool &forceMeshReload = sim->forceMeshReload;
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
        
      // std::cerr << "\tdn->SetLevelRefinementRatios(" << level << ", <"
      //                << rr[0] << "," << rr[1] << "," << rr[2] << ">)\n";

      VisIt_DomainNesting_set_levelRefinement(dn, level, rr);
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
        
      // For node based meshes add one if there is a neighbor patch.
      if( meshname.find("NC_") == 0 )
      {
        int nlow[3], nhigh[3];
        patchInfo.getBounds(nlow, nhigh, "NEIGHBORS");
          
        for (int i=0; i<3; i++)
          phigh[i] += nhigh[i];
      }

      int extents[6];

      for (int i=0; i<3; ++i)
      {
        extents[i+0] = plow[i];
        extents[i+3] = phigh[i] - 1;
      }

      // std::cerr << "\tdn->SetNestingForDomain("
      //                << p << "," << my_level << ") <"
      //                << extents[0] << "," << extents[1] << "," << extents[2] << "> to <"
      //                << extents[3] << "," << extents[4] << "," << extents[5] << ">";

      // std::cerr << "\t children patches <";
        
      // for (int i=0; i<childPatches[p].size(); ++i)
      //        std::cerr << childPatches[p][i] << ",  ";

      // std::cerr << ">" << std::endl;;

      VisIt_DomainNesting_set_nestingForPatch(dn, p, my_level,
                                              &(childPatches[p][0]),
                                              childPatches[p].size(),
                                              extents);
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

  if( std::string(meshname).find("Machine_") == 0 )
  {
    bool global = (std::string(meshname).find("/Global") != std::string::npos);
    bool local  = (std::string(meshname).find("/Local" ) != std::string::npos);
    bool patch  = (std::string(meshname).find("/Patch" ) != std::string::npos);

    // Only rank 0 return the whole of the mesh.
    if( global && sim->myworld->myRank() != 0 )
      return VISIT_INVALID_HANDLE;

    // nConnections are for quads so the type plus four points.
    const unsigned int nQuadVals = 5;
    unsigned int nConnections = 0;
    int* connections = nullptr;

    unsigned int nPts = 0;
    float* xPts = nullptr;
    float* yPts = nullptr;
    // float* zPts = nullptr;

    if( global )
    {    
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

    else if( local )
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

    else if( patch )
    {
      const PatchSubset* myPatches =
        lb->getPerProcessorPatchSet(gridP)->getSubset( domain );

      const unsigned int nPatches = myPatches->size();

      // Total size of the layout. Try to make rectangles.
      unsigned int xMax = sqrt(nPatches);
      unsigned int yMax = xMax;

      // Make sure to cover all the patches - may be blank areas.
      while( xMax * yMax < nPatches )
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
      connections = new int[ nPatches * nQuadVals ];

      for (unsigned int  p = 0; p < nPatches; ++p) {

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
 
    if(VisIt_UnstructuredMesh_alloc(&meshH) != VISIT_ERROR)
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

  // LoadExtra loadExtraElements = (LoadExtra) sim->loadExtraElements;
  // bool &forceMeshReload = sim->forceMeshReload;
  TimeStepInfo* &stepInfo = sim->stepInfo;

  // int timestate = sim->cycle;

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
      getParticleData(schedulerP, gridP, level, local_patch, vars, matlNo);

    visit_handle cordsH = VISIT_INVALID_HANDLE;

    if(VisIt_VariableData_alloc(&cordsH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataD(cordsH, VISIT_OWNER_VISIT,
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

    // for(int i = 0; i < pd->num; ++i)
    // {
    //   onevertex = i; 
    //   ugrid->InsertNextCell(VTK_VERTEX, 1, &onevertex); 
    // } 
    
    // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
    // owns the data (VISIT_OWNER_SIM - indicates the simulation owns
    // the data). However pd needs to be delted.
    
    // delete pd->data
    delete pd;

#ifdef COMMENTOUT_FOR_NOW
    //try to retrieve existing cache ref
    void_ref_ptr vrTmp =
      cache->GetVoidRef(meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
                        timestate, domain);

    vtkDataArray *pID = nullptr;

    if (*vrTmp == nullptr)
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

      ParticleDataRaw *pd = nullptr;

      //debug5<<"\t(*getParticleData)...\n";
      //todo: this returns an array of doubles. Need to return
      //expected datatype to avoid unnecessary conversion.
      pd = getParticleData(schedulerP, gridP, level, local_patch,
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
                             timestate, domain, vr );

        //make sure it worked
        void_ref_ptr vrTmp =
          cache->GetVoidRef(meshname, AUXILIARY_DATA_GLOBAL_NODE_IDS,
                            timestate, domain);

        if (*vrTmp == nullptr || *vrTmp != *vr)
          throw InvalidFilesException("failed to register uda particle global node");
      }
    }

    return ugrid;
#endif
  }

  // Volume data
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

        // std::cerr << "Patch " << array[0] << "  " << array[1] << std::endl;
          
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
      /* Fill in the attributes of the RectilinearMesh. */
      VisIt_RectilinearMesh_setCoordsXYZ(meshH, cordH[0], cordH[1], cordH[2]);
      VisIt_RectilinearMesh_setRealIndices(meshH, base, dims);
      VisIt_RectilinearMesh_setBaseIndex(meshH, base);

      // VisIt_RectilinearMesh_setGhostCells(meshH, visit_handle gz);
      // VisIt_RectilinearMesh_setGhostNodes(meshH, visit_handle gn);
    }
  }
  // Volume data
  else //if (meshName.find("Particle_Mesh") == std::string::npos)
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

  std::string varName(varname);

  if( varName.find("Processor/Machine/") == 0 )
  {
    // At the present time all vars are local.
    // bool global = (varName.find("Global") != std::string::npos);
    // bool local  = (varName.find("Local" ) != std::string::npos);

    bool global = false;
    bool local  = true;

    // Only rank 0 return the whole of the mesh.
    if( global && sim->myworld->myRank() != 0 )
      return VISIT_INVALID_HANDLE;

    unsigned int totalCores = 0;
    unsigned int nValues = 0;
    int* values = nullptr;

    if( global )
    {
      totalCores = sim->switchNodeList.size() * sim->maxNodes * sim->maxCores;

      nValues = 0;
      values = new int[ totalCores ];

      for( unsigned int i=0; i<totalCores; ++i)
        values[i] = 0;
    
      // Loop through each switch.
      for( unsigned int s=0; s<sim->switchNodeList.size(); ++s )
      {
        unsigned int nCores = 0;

        // Loop through each node.
        for( unsigned int n=0; n<sim->switchNodeList[s].size(); ++n )
        {
          for( unsigned int i=0; i<sim->nodeCores.size(); ++i )
          {
            if( sim->nodeStart[i] <= sim->switchNodeList[s][n] &&
                sim->switchNodeList[s][n] <= sim->nodeStop[i] )
            {
              nCores  = sim->nodeCores[i];
              break;
            }
          }

          // Loop through each core.
          for( unsigned int i=0; i<nCores; ++i )
          {
            if( varName.find("Processor/Machine/Node/Number") == 0 )
              values[nValues++] = atoi(sim->hostNode.c_str());
            else if( varName.find("Processor/Machine/MPI/Comm/Node") == 0 )
              values[nValues++] = sim->myworld->myNode();
            else if( varName.find("Processor/Machine/MPI/Comm/Rank") == 0 )
              values[nValues++] = sim->myworld->myNode_myRank();
            else if( varName.find("Processor/Machine/MPI/Rank") == 0 )
              values[nValues++] = sim->myworld->myRank();
          }
        }
      }
    }

    else if( local )
    {
      totalCores = 1;

      nValues = 0;
      values = new int[ totalCores ];

      for( unsigned int i=0; i<totalCores; ++i)
        values[i] = 0;
    
      // Indexes of the switch, node, and core.
      // unsigned int s = sim->switchIndex;
      // unsigned int n = sim->nodeIndex;
      // unsigned int c = sim->myworld->myNode_myRank();

      if( varName.find("Processor/Machine/Node/Number") == 0 )
        values[nValues++] = atoi(sim->hostNode.c_str());
      else if( varName.find("Processor/Machine/MPI/Comm/Node") == 0 )
        values[nValues++] = sim->myworld->myNode();
      else if( varName.find("Processor/Machine/MPI/Comm/Rank") == 0 )
        values[nValues++] = sim->myworld->myNode_myRank();
      else if( varName.find("Processor/Machine/MPI/Rank") == 0 )
        values[nValues++] = sim->myworld->myRank();
    }

    visit_handle varH = VISIT_INVALID_HANDLE;
    
    if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataI(varH, VISIT_OWNER_VISIT, 1, nValues, values);

      // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
      // owns the data (VISIT_OWNER_SIM - indicates the simulation
      // owns the data).
    }

    return varH;
  }  

  SchedulerP schedulerP = sim->simController->getSchedulerP();
  LoadBalancer * lb     = sim->simController->getLoadBalancer();
  GridP gridP           = sim->gridP;

  LoadExtra loadExtraElements = (LoadExtra) sim->loadExtraElements;
  // bool &forceMeshReload = sim->forceMeshReload;
  TimeStepInfo* &stepInfo = sim->stepInfo;

  // int timestate = sim->cycle;

  bool isParticleVar = false;
  bool isInternalVar = false;
  bool isMachineMeshVar = false;

  bool isPerPatchVar = false;

  // Get the var name sans the material. If a patch or processor
  // variable then the var name will be either "Patch" or "Processor".
  size_t found = varName.find("/");
  std::string matl = varName.substr(found + 1);
  varName = varName.substr(0, found);

  // Get the varType except for processor and patch based data which
  // does not come from the data warehouse but instead from internal
  // structures.
  std::string varType("");

  if( strncmp(varname, "Patch/Nodes", 11) == 0 )
  {
    isInternalVar = true;

    varType = "NC_Mesh";
  }
  else if( varName == "Processor" )
  {
    isInternalVar = true;

    varType = "CC_Mesh";

    // If the machine profile is available get the sim or host name.
    if( sim->switchNodeList.size() )
    {
      std::string hostName = std::string(varname);
      found = hostName.find_last_of("/");
      hostName = hostName.substr(found + 1);

      isMachineMeshVar = (hostName == sim->hostName);
    }
  }
  else if( strncmp(varname, "Patch/Id", 8) == 0 ||
           strncmp(varname, "Patch/ProcRank", 14) == 0 ||
           strncmp(varname, "Patch/ProcNode", 14) == 0 )
  {
    isInternalVar = true;

    // If the machine profile is available get the sim or host name.
    if( sim->switchNodeList.size() )
    {
      // If a the per patch is on the machine view (i.e. not the sim view)
      isPerPatchVar = (varName.find("/Sim") == std::string::npos);
    }

    varType = "CC_Mesh";
  }

  else if( strncmp(varname, "Patch/Bounds/Low",  16) == 0 ||
           strncmp(varname, "Patch/Bounds/High", 17) == 0 )
  {
    isInternalVar = true;

    varType = "CC_Mesh";
  }
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
      if( sim->switchNodeList.size() )
      {
        // If a the per patch is on the machine view (i.e. not the sim view)
        isPerPatchVar = (varName.find("/Sim") == std::string::npos);

        // Get the var name sans the sim or host name.
        found = varName.find_last_of("/");
        varName = varName.substr(0, found);
      }
    }

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
          break;
        }
      }
    }
  }

  if( varType.empty() )
  {
    std::stringstream msg;
    msg << "Visit libsim - "
        << "Uintah variable \"" << varname << "\"  "
        << "has no type.";
    
    VisItUI_setValueS("SIMULATION_MESSAGE_ERROR", msg.str().c_str(), 1);

    return varH;
  }

  if (isMachineMeshVar)
  {
    MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
      (sim->simController->getSchedulerP().get_rep());

    GridDataRaw *gd = new GridDataRaw;

    gd->num = 1;
    gd->components = 1;

    gd->data = new double[gd->num * gd->components];

    std::string procLevelName;
      
    if( varName == "Processor" )
    {
      std::string tmp = std::string(varname);

      // Strip off the "Processor/Application/", "Processor/MPI/" or
      // "Processor/Runtime/" prefix
      found = tmp.find_first_of("/");
      tmp.erase(0, found + 1);
      found = tmp.find_first_of("/");
      tmp.erase(0, found + 1);
      
      // Get the actual var name and strip it off
      found = tmp.find_first_of("/");
      varName = tmp.substr(0, found);
      tmp.erase(0, found + 1);

      // Get the procLevelName and strip it off
      found = tmp.find_first_of("/");
      procLevelName = tmp.substr(0, found);
      tmp.erase(0, found + 1);

      if( procLevelName == "Node" )
      {
        // Get the operation and strip it off
        found = tmp.find_first_of("/");
        procLevelName += "/" + tmp.substr(0, found);
        tmp.erase(0, found + 1);
      }

      // All that is left is profile which is really the mesh and that
      // is already known.
      // std::string profileName = tmp;
    }

    // Simulation Runtime stats
    if( strncmp( varname, "Processor/Runtime/", 18 ) == 0 &&
        sim->simController->getRuntimeStats().exists( varName ) )
    {
      double val;
      
      if( procLevelName == "Node/Average" )
        val = sim->simController->getRuntimeStats().getNodeAverage( varName );
      else if( procLevelName == "Node/Sum" )
        val = sim->simController->getRuntimeStats().getNodeSum( varName );
      else // if( procLevelName == "Rank" )
        val = sim->simController->getRuntimeStats().getRankValue( varName );
      
      for (int i=0; i<gd->num*gd->components; ++i)
        gd->data[i] = val;
    }

    // MPI Scheduler Timing stats
    else if( strncmp( varname, "Processor/MPI/", 14 ) == 0 &&
             mpiScheduler && mpiScheduler->mpi_info_.exists(varName) )
    {
      double val;
      
      if( procLevelName == "Node/Average" )
        val = mpiScheduler->mpi_info_.getNodeAverage( varName );
      else if( procLevelName == "Node/Sum" )
        val = mpiScheduler->mpi_info_.getNodeSum( varName );
      else // if( procLevelName == "Rank" )
        val = mpiScheduler->mpi_info_.getRankValue( varName );
      
      for (int i=0; i<gd->num*gd->components; ++i)
        gd->data[i] = val;
    }

    // Application stats
    else if( strncmp( varname, "Processor/Application/", 16 ) == 0 &&
        sim->simController->getApplicationInterface()->getApplicationStats().exists( varName ) )
    {
      double val;
      
      if( procLevelName == "Node/Average" )
        val = sim->simController->getApplicationInterface()->getApplicationStats().getNodeAverage( varName );
      else if( procLevelName == "Node/Sum" )
        val = sim->simController->getApplicationInterface()->getApplicationStats().getNodeSum( varName );
      else // if( procLevelName == "Rank" )
        val = sim->simController->getApplicationInterface()->getApplicationStats().getRankValue( varName );
      
      for (int i=0; i<gd->num*gd->components; ++i)
        gd->data[i] = val;
    }

    // This should never be reached.
    else
    {
      std::stringstream msg;
      msg << "Visit libsim - "
          << "Uintah internal variable \"" << varname << "\"  "
          << "could not be processed.";
            
      VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
      
      for (int i=0; i<gd->num*gd->components; ++i)
        gd->data[i] = 0;
    }

    if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, gd->components,
                                  gd->num*gd->components, gd->data);

      // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
      // owns the data (VISIT_OWNER_SIM - indicates the simulation
      // owns the data). However, gd needs to be deleted.
      
      // delete gd->data
      delete gd;
    }
  }
  
  // particle data
  else if (isParticleVar)
  {
    int level, local_patch;
    GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

    int matlNo = -1;
    if (matl.compare("*") != 0)
      matlNo = atoi(matl.c_str());
      
    ParticleDataRaw *pd = 
      getParticleData(schedulerP, gridP, level, local_patch, varName, matlNo);

    CheckNaNs(pd->data, pd->num*pd->components, varname, level, local_patch);

    if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, pd->components,
                                  pd->num * pd->components, pd->data);
      
      // vtkDoubleArray *rv = vtkDoubleArray::New();
      // rv->SetNumberOfComponents(pd->components);
      // rv->SetArray(pd->data, pd->num*pd->components, 0);
      
      // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
      // owns the data (VISIT_OWNER_SIM - indicates the simulation
      // owns the data). However, pd needs to be deleted.
      
      // delete pd->data
      delete pd;
    }
  }

  // Volume data
  else
  {
    int level, local_patch;
    GetLevelAndLocalPatchNumber(stepInfo, domain, level, local_patch);

    MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
      (sim->simController->getSchedulerP().get_rep());

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

      if( varName == "Processor" )
      {
        // Using the patch mesh on the simulation
        gd->num = 1;
        gd->components = 1;
      }
      else if( strncmp(varname, "Patch/Id", 8) == 0 ||
               strncmp(varname, "Patch/ProcRank", 14) == 0 ||
               strncmp(varname, "Patch/ProcNode", 14) == 0 )
      {
        // Using the patch mesh on the machine layout
        if( isPerPatchVar )
          gd->num =
            lb->getPerProcessorPatchSet(gridP)->getSubset( domain )->size();
        // Using the patch mesh on the simulation
        else
          gd->num = 1;

        gd->components = 1;
      }
      else if (strncmp(varname, "Patch/Nodes", 11) == 0 )
      {
        // Using the node mesh on the simulation
        gd->num = ((phigh[0] - plow[0]) *
                   (phigh[1] - plow[1]) *
                   (phigh[2] - plow[2]));
        // Bounds are vectors
        gd->components = 3;
      }
      else if( strncmp(varname, "Patch/Bounds/Low",  16) == 0 ||
               strncmp(varname, "Patch/Bounds/High", 17) == 0)
      {
        // Using the patch mesh on the simulation
        gd->num = 1;
        // Bounds are vectors
        gd->components = 3;
      }
      // This section should never be reached.
      else
      {
        gd->num = 1;
        gd->components = 1;

        std::stringstream msg;
        msg << "Visit libsim - "
            << "Uintah internal variable \"" << varname << "\"  "
            << "could not be processed.";
            
        VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = 0;      
      }

      gd->data = new double[gd->num * gd->components];

      std::string procLevelName;
      
      // Strip off the "Processor/Runtime/" or "Processor/MPI/" prefix
      // and the rank or node postfix.
      if( varName == "Processor" )
      {
        std::string tmp = std::string(varname);

        // Strip off the "Processor/Application/", "Processor/MPI/" or
        // "Processor/Runtime/" prefix
        found = tmp.find_first_of("/");
        tmp.erase(0, found + 1);
        found = tmp.find_first_of("/");
        tmp.erase(0, found + 1);
      
        // Get the actual var name and strip it off
        found = tmp.find_first_of("/");
        varName = tmp.substr(0, found);
        tmp.erase(0, found + 1);

        // Get the procLevelName and strip it off
        found = tmp.find_first_of("/");
        procLevelName = tmp.substr(0, found);
        tmp.erase(0, found + 1);

        if( procLevelName == "Node" )
        {
          // Get the operation and strip it off
          found = tmp.find_first_of("/");
          procLevelName += "/" + tmp.substr(0, found);
          tmp.erase(0, found + 1);
        }

        // All that is left is profile which is really the mesh and
        // that is already known.
        // std::string profileName = tmp;
      }

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
               mpiScheduler && mpiScheduler->mpi_info_.exists(varName) )
      {
        double val;

        if( procLevelName == "Node/Average" )
          val = mpiScheduler->mpi_info_.getNodeAverage( varName );
        else if( procLevelName == "Node/Sum" )
          val = mpiScheduler->mpi_info_.getNodeSum( varName );
        else // if( procLevelName == "Rank" )
          val = mpiScheduler->mpi_info_.getRankValue( varName );

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }

      // Patch Id
      else if( strncmp(varname, "Patch/Id", 8) == 0 )
      {
        if( isPerPatchVar )
        {
          // Here the request is for per patch data that is going on the
          // machine view.  The domain is for a rank.
          const PatchSubset* myPatches =
            lb->getPerProcessorPatchSet(gridP)->getSubset( domain );

          const unsigned int nPatches = myPatches->size();
      
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
        else
        {
          double val = patchInfo.getPatchId();
          
          for (int i=0; i<gd->num*gd->components; ++i)
            gd->data[i] = val;
        }
      }
      // Patch processor rank
      else if( strncmp(varname, "Patch/ProcRank", 14) == 0 )
      {
        double val = sim->myworld->myRank();

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }
      // Patch processor node
      else if( strncmp(varname, "Patch/ProcNode", 14) == 0 )
      { 
        double val = sim->myworld->myNode();

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = val;
      }
      // Patch node ids
      else if (strncmp(varname, "Patch/Nodes", 11) == 0 )
      {
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
      // Patch bounds
      else if( strncmp(varname, "Patch/Bounds/Low",  16) == 0 ||
               strncmp(varname, "Patch/Bounds/High", 17) == 0 )
      {
        // Get the bounds for this mesh as a variable (not for the grid).
        std::string meshname = std::string(varname);
        found = meshname.find_last_of("/");
        meshname = meshname.substr(found + 1);
        
        patchInfo.getBounds(plow, phigh, meshname);
        
        int *value;
        
        if (strncmp(varname, "Patch/Bounds/Low", 16) == 0 )
          value = &plow[0];
        else // if( strncmp(varname, "Patch/Bounds/High", 17) == 0)
          value = &phigh[0];

        for (int i=0; i<gd->num; i++)
          for (int c=0; c<3; c++)
            gd->data[i*gd->components+c] = value[c];
      }
      // This section should never be reached.
      else
      {
        std::stringstream msg;
        msg << "Visit libsim - "
            << "Uintah internal variable \"" << varname << "\"  "
            << "could not be processed.";
            
        VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = 0;
      }
    }
    else if( isPerPatchVar )
    {
      // Here the request is for per patch data that is going on the
      // machine view. The domain is for a rank.
      const PatchSubset* myPatches =
        lb->getPerProcessorPatchSet(gridP)->getSubset( domain );

      const unsigned int nPatches = myPatches->size();
      
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
                      atoi(matl.c_str()), plow, phigh, (LoadExtra) 0 );

        gd->data[p] = tmp->data[0];

        delete tmp->data;
        delete tmp;
      }
    }
    // Patch data from the warehouse
    else
    {
      gd = getGridData(schedulerP, gridP, level, local_patch, varName,
                       atoi(matl.c_str()), plow, phigh,
                       (LoadExtra) (nodeCentered ? 0 : loadExtraElements));

      if( gd )
      {
        CheckNaNs(gd->data, gd->num*gd->components, varname, level, local_patch);
      }
      // This section should never be reached ... but ...
      else
      {
        std::stringstream msg;
        msg << "Visit libsim - "
            << "Uintah variable \"" << varname << "\"  "
            << "could not be processed.";
            
        VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);


        gd = new GridDataRaw;

        gd->components = 1;
        gd->num = ((phigh[0] - plow[0]) *
                   (phigh[1] - plow[1]) *
                   (phigh[2] - plow[2]));

        int numVars = stepInfo->varInfo.size();
        
        for (int i=0; i<numVars; ++i)
        {
          std::string varname = stepInfo->varInfo[i].name;
          std::string vartype = stepInfo->varInfo[i].type;

          if( varname == varName )
          {
            // 3 -> vector 
            if (vartype.find("Vector") != std::string::npos)
              gd->components = 3;
            // 9 -> tensor 
            else if (vartype.find("Matrix3") != std::string::npos)
              gd->components = 9;
            // 7 -> vector
            else if (vartype.find("Stencil7") != std::string::npos)
              gd->components = 7;
            // 4 -> vector
            else if (vartype.find("Stencil4") != std::string::npos)
              gd->components = 4;
            // PerPatch
            else if (vartype.find("PerPatch") != std::string::npos)
              gd->num = 1;
          }
        }

        gd->data = new double[gd->num * gd->components];

        for (int i=0; i<gd->num*gd->components; ++i)
          gd->data[i] = 0;
      }
    }

    if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataD(varH, VISIT_OWNER_VISIT, gd->components,
                                  gd->num*gd->components, gd->data);

      // No need to delete as the flag is VISIT_OWNER_VISIT so VisIt
      // owns the data (VISIT_OWNER_SIM - indicates the simulation
      // owns the data). However, gd needs to be deleted.
      
      // delete gd->data;
      delete gd;
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

    if( std::string(meshname).find("Machine_") == 0 )
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

} // End namespace Uintah
