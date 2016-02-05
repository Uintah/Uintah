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
#include "visit_libsim_callbacks.h"
#include "visit_libsim_customUI.h"

#include "VisItControlInterface_V2.h"
#include "VisItDataInterface_V2.h"

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Regridder.h>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/VarLabel.h>


#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>

#include <Core/Util/DebugStream.h>

#include "StandAlone/tools/uda2vis/udaData.h"
#include "StandAlone/tools/uda2vis/uda2vis.h"

#include <stdio.h>

namespace Uintah {

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

  SchedulerP      schedulerP = sim->simController->getSchedulerP();
  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  GridP           gridP      = sim->gridP;
      
  if( !schedulerP.get_rep() || !gridP.get_rep() )
  {
    return VISIT_INVALID_HANDLE;
  }

  int timestate = sim->cycle;

  sim->useExtraCells = true;
  bool &useExtraCells = sim->useExtraCells;

  sim->forceMeshReload = true;
  // bool &forceMeshReload = sim->forceMeshReload;

  sim->nodeCentered = false;
  // bool &nodeCentered = sim->nodeCentered;

  sim->stepInfo =
    getTimeStepInfo2(schedulerP, gridP, timestate, useExtraCells);

  TimeStepInfo* &stepInfo = sim->stepInfo;

  visit_handle md = VISIT_INVALID_HANDLE;

  /* Create metadata with no variables. */
  if(VisIt_SimulationMetaData_alloc(&md) == VISIT_OKAY)
  {
    /* Set the simulation state. */

    /* NOTE visit_ReadMetaData is called as a results of calling
       visit_CheckState which calls VisItTimeStepChanged at this point
       the sim->runMode will always be VISIT_SIMMODE_RUNNING. */
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

    // don't add proc id unless CC_Mesh or NC_Mesh exists (some only
    // have SFCk_MESH)
    bool addProcId = false;
    std::string mesh_for_procid;

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
        
    for (int i=0; i<numVars; ++i)
    {
      if (stepInfo->varInfo[i].type.find("ParticleVariable") == std::string::npos)
      {
        std::string varname = stepInfo->varInfo[i].name;
        std::string vartype = stepInfo->varInfo[i].type;
//      int matsize         = stepInfo->varInfo[i].materials.size();

        std::string mesh_for_this_var;
        VisIt_VarCentering cent = VISIT_VARCENTERING_ZONE;

        if (vartype.find("NC") != std::string::npos)
        {
          cent = VISIT_VARCENTERING_NODE;
          mesh_for_this_var.assign("NC_Mesh"); 
          addProcId = true;
          mesh_for_procid = mesh_for_this_var;
        }
        else if (vartype.find("CC") != std::string::npos)
        {
          cent = VISIT_VARCENTERING_ZONE;
          mesh_for_this_var.assign("CC_Mesh");
          addProcId = true;
          mesh_for_procid = mesh_for_this_var;
        }
        else if (vartype.find("SFC") != std::string::npos)
        { 
          cent = VISIT_VARCENTERING_ZONE;

          if (vartype.find("SFCX") != std::string::npos)               
            mesh_for_this_var.assign("SFCX_Mesh");
          else if (vartype.find("SFCY") != std::string::npos)          
            mesh_for_this_var.assign("SFCY_Mesh");
          else if (vartype.find("SFCZ") != std::string::npos)          
            mesh_for_this_var.assign("SFCZ_Mesh");
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
            if( sim->simController->doAMR() )
              VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_AMR);
            else
              VisIt_MeshMetaData_setMeshType(mmd, VISIT_MESHTYPE_AMR);

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
//        VisIt_MeshMetaData_setContainsExteriorBoundaryGhosts(mmd, false);

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

          // md->Add(mesh);
          meshes_added.insert(mesh_for_this_var);
        }

        // Add mesh vars
        int numMaterials = stepInfo->varInfo[i].materials.size();

        for (int j=0; j<numMaterials; ++j)
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
      // Add in the processor runtime stats.
      for( unsigned int i=0; i<simStateP->d_runTimeStats.size(); ++i )
      {
        visit_handle vmd = VISIT_INVALID_HANDLE;
          
        if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
        {
          std::string stat = std::string("processor/runtime/") + 
            simStateP->d_runTimeStats.getName( (SimulationState::RunTimeStat) i );

          std::string units = 
            simStateP->d_runTimeStats.getUnits( (SimulationState::RunTimeStat) i );

          VisIt_VariableMetaData_setName(vmd, stat.c_str());
          VisIt_VariableMetaData_setMeshName(vmd, mesh_for_procid.c_str());
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

      // Add in the mpi run time stats.
      if( mpiScheduler )
      {
        for( unsigned int i=0; i<mpiScheduler->mpi_info_.size(); ++i )
        {
          visit_handle vmd = VISIT_INVALID_HANDLE;
          
          if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
          {
            std::string stat = std::string("processor/mpi/") + 
              mpiScheduler->mpi_info_.getName( (MPIScheduler::TimingStat)i );

            std::string units = 
              simStateP->d_runTimeStats.getUnits( (SimulationState::RunTimeStat) i );

            VisIt_VariableMetaData_setName(vmd, stat.c_str());
            VisIt_VariableMetaData_setMeshName(vmd, mesh_for_procid.c_str());
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

      {
        visit_handle vmd = VISIT_INVALID_HANDLE;
          
        if(VisIt_VariableMetaData_alloc(&vmd) == VISIT_OKAY)
        {
          VisIt_VariableMetaData_setName(vmd, "processor/id");
          VisIt_VariableMetaData_setMeshName(vmd, mesh_for_procid.c_str());
          VisIt_VariableMetaData_setCentering(vmd, VISIT_VARCENTERING_ZONE);
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
              else
              {
                VisIt_VariableMetaData_setType(vmd, VISIT_VARTYPE_SCALAR);
                VisIt_VariableMetaData_setNumComponents(vmd, 1);
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

    /* Add some commands. */
    const char *cmd_names[] = {"Stop", "Step", "Run",
                               "Save", "Checkpoint", "Regrid",
                               "Finish", "Terminate", "Abort"};

    int numNames = sizeof(cmd_names) / sizeof(const char *);

    for (int i=0; i<numNames; ++i)
    {
      bool enabled;

      // Check for regrid, save, and checkpoint option.
      if(strcmp( "Regrid", cmd_names[i] ) == 0 )
        enabled = sim->simController->doAMR();

      else if(strcmp( "Save", cmd_names[i] ) == 0 )
        enabled = !sim->simController->getOutput()->isOutputTimestep();

      else if(strcmp( "Checkpoint", cmd_names[i] ) == 0 )
        enabled = !sim->simController->getOutput()->isCheckpointTimestep();

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

    // Setup the custom UI time values.
    visit_GetTimeVars( sim );
    
    // Setup the custom UI optional UPS variable table
    visit_GetUPSVars( sim );

    // Setup the custom UI output variable table
    visit_GetOutputIntervals( sim );

    // Setup the custom UI optional min/max variable table
    visit_GetAnalysisVars( sim );

    // Setup the custom UI Grid Info
    visit_GetGridInfo( sim );

    // Setup the custom UI Runtime Stats
    visit_GetRuntimeStats( sim );

    // Setup the custom UI MPI Stats
    visit_GetMPIStats( sim );

    // Setup the custom UI Image variables
    visit_GetImageVars( sim );

    // if( sim->message.size() )
    // {
    //   visit_handle msg = VISIT_INVALID_HANDLE;
      
    //   if(VisIt_MessageMetaData_alloc(&msg) == VISIT_OKAY)
    //   {
    //  VisIt_MessageMetaData_setName(msg, sim->message.c_str());
    //  VisIt_SimulationMetaData_addMessage(md, msg);
    //   }
    // }
  }

  return md;
}


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
//  Method: visit_CalculateDomainNesting
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

      // debug5 << "Calculating avtRectilinearDomainBoundaries for "
      //           << meshname << " mesh (" << rdb << ")." << std::endl;

      // avtRectilinearDomainBoundaries *rdb =
      //        new avtRectilinearDomainBoundaries(true);
      // rdb->SetNumDomains(totalPatches);

      for (int patch=0; patch<totalPatches; ++patch)
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
        // debug5 << "\trdb->SetIndicesForPatch(" << patch << ","
        //           << my_level << ", <" << e[0] << "," << e[2] << "," << e[4]
        //           << "> to <" << e[1] << "," << e[3] << "," << e[5] << ">)"
        //             << std::endl;

        VisIt_DomainBoundaries_set_amrIndices(rdb, patch, my_level, e);
//      VisIt_DomainBoundaries_finish(rdb, patch);
        // rdb->SetIndicesForPatch(patch, my_level, e);
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
      VisIt_DomainNesting_set_dimensions(dn, totalPatches, numLevels, 3);

      // avtStructuredDomainNesting *dn =
      //        new avtStructuredDomainNesting(totalPatches, numLevels);
      // dn->SetNumDimensions(3);

      //debug5 << "Calculating avtStructuredDomainNesting for "
      //       << meshname << " mesh (" << dn << ")." << std::endl;
      
      //
      // Calculate what the refinement ratio is from one level to the next.
      //
      for (int level=0; level<numLevels; ++level)
      {
        // SetLevelRefinementRatios requires data as a vector<int>
        int rr[3];

        for (int i=0; i<3; ++i)
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
      
      for (int level = numLevels-1 ; level > 0 ; level--)
      {
        int prev_level = level-1;
        LevelInfo &levelInfoParent = stepInfo->levelInfo[prev_level];
        LevelInfo &levelInfoChild = stepInfo->levelInfo[level];
        
        for (int child=0; child<(int)levelInfoChild.patchInfo.size(); ++child)
        {
          PatchInfo &childPatchInfo = levelInfoChild.patchInfo[child];
          int child_low[3],child_high[3];
          childPatchInfo.getBounds(child_low,child_high,meshname);
          
          for (int parent = 0;
               parent<(int)levelInfoParent.patchInfo.size(); ++parent)
          {
            PatchInfo &parentPatchInfo = levelInfoParent.patchInfo[parent];
            int parent_low[3],parent_high[3];
            parentPatchInfo.getBounds(parent_low,parent_high,meshname);
            
            int mins[3], maxs[3];

            for (int i=0; i<3; ++i)
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
      for (int p=0; p<totalPatches ; ++p)
      {
        int my_level, local_patch;
        GetLevelAndLocalPatchNumber(stepInfo, p, my_level, local_patch);
        
        PatchInfo &patchInfo =
          stepInfo->levelInfo[my_level].patchInfo[local_patch];
        int low[3],high[3];
        patchInfo.getBounds(low,high,meshname);
        
        int e[6];

        for (int i=0; i<3; ++i)
        {
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
          
          for (int i=0; i<3; ++i)
          {
            cp[i] = childPatches[p][i];
            
            VisIt_DomainNesting_set_nestingForPatch(dn, p, my_level,
                                                    cp, childPatches[p].size(),
                                                    e);
//          delete cp;
            
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
  //                  AUXILIARY_DATA_DOMAIN_BOUNDARY_INFORMATION,
  //                  timestate, -1);
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

  SchedulerP schedulerP = sim->simController->getSchedulerP();
  GridP      gridP      = sim->gridP;

  // bool &useExtraCells   = sim->useExtraCells;
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

    // for(int i = 0; i < pd->num; ++i)
    // {
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

    for (int i=0; i<3; ++i) 
    {
      int offset = 1; // always one for non-node-centered

      if (nodeCentered && high[i] == ghigh[i]) 
        offset = 0; // nodeCentered and patch end is on high boundary

      dims[i] = high[i]-low[i]+offset;
    }

    // debug5 << "Calculating vtkRectilinearGrid mesh for "
    //     << meshName << " mesh (" << rgrid << ").\n";

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
    for (int c=0; c<3; ++c)
    {
      // vtkFloatArray *coords = vtkFloatArray::New(); 
      // coords->SetNumberOfTuples(dims[c]); 
      // float *array = (float *) coords->GetVoidPointer(0);

      float *array = new float[ dims[c] ];

      if(VisIt_VariableData_alloc(&cordH[c]) == VISIT_OKAY)
      {
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
  visit_handle varH = VISIT_INVALID_HANDLE;
    
  visit_simulation_data *sim = (visit_simulation_data *)cbdata;

  SchedulerP schedulerP = sim->simController->getSchedulerP();
  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  GridP      gridP      = sim->gridP;
  
  // bool &useExtraCells   = sim->useExtraCells;
  // bool &forceMeshReload = sim->forceMeshReload;
  bool &nodeCentered    = sim->nodeCentered;

  TimeStepInfo* &stepInfo = sim->stepInfo;

  int timestate = sim->cycle;

  std::string varName(varname);
  bool isParticleVar = false;
  
  size_t found = varName.find("/");  
  std::string matl = varName.substr(found + 1);
  varName = varName.substr(0, found);

  std::string varType( "CC_Mesh" );

  // Check for a particle variable
  if (strcmp(varname, "processor") != 0)
  {
    for (int k=0; k<(int)stepInfo->varInfo.size(); ++k)
    {
      if (stepInfo->varInfo[k].name == varName)
      {
        varType = stepInfo->varInfo[k].type;

        if (stepInfo->varInfo[k].type.find("ParticleVariable") !=
            std::string::npos)
        {
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

    for (int i=0; i<stepInfo->levelInfo.size(); ++i)
      totalPatches += stepInfo->levelInfo[i].patchInfo.size();

    // calculate which process we should wait for a message from
    // if we're processing doiman 0 don't wait for anyone else
    int prev = (rank+numProcs-1) % numProcs;
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

    if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
    {
      VisIt_VariableData_setDataD(varH, VISIT_OWNER_SIM, pd->components,
                                  pd->num * pd->components, pd->data);
      
      // vtkDoubleArray *rv = vtkDoubleArray::New();
      // rv->SetNumberOfComponents(pd->components);
      // rv->SetArray(pd->data, pd->num*pd->components, 0);
      
      // ARS - FIX ME - CHECK FOR LEAKS
      // don't delete pd->data - vtk owns it now!
      // delete pd;
    }
  }

  // volume data
  else //if (!isParticleVar)
  {
    MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
      (sim->simController->getSchedulerP().get_rep());
        

    LevelInfo &levelInfo = stepInfo->levelInfo[level];
    PatchInfo &patchInfo = levelInfo.patchInfo[local_patch];

    // The region we're going to ask uintah for (from qlow to qhigh-1)      
    int qlow[3], qhigh[3];
    patchInfo.getBounds(qlow,qhigh,varType);
      
    GridDataRaw *gd = NULL;

    if( varName == "processor" )
    {
      varName = std::string(varname);
      found = varName.find_last_of("/");
      varName = varName.substr(found + 1);

      double val;

      // Processor Id
      if( strcmp(varname, "processor/id") == 0 )
      {
          val = patchInfo.getProcId();
      }
      // Simulation State Runtime stats
      else if( strncmp( varname, "processor/runtime/", 18 ) == 0 &&
               simStateP->d_runTimeStats.exists(varName) )
      {
        val = simStateP->d_runTimeStats.getValue( varName );
      }

      // MPI Scheduler Timing stats
      else if( strncmp( varname, "processor/mpi/", 14 ) == 0 &&
               mpiScheduler && mpiScheduler->mpi_info_.exists(varName) )
      {
        val = mpiScheduler->mpi_info_.getValue( varName );
      }

      // Create at new grid data for the values.
      gd = new GridDataRaw;

      for (int i=0; i<3; ++i)
      {
        gd->low[i ] = qlow[i];
        gd->high[i] = qhigh[i];
      }

      gd->components = 1;

      int ncells = (qhigh[0]-qlow[0])*(qhigh[1]-qlow[1])*(qhigh[2]-qlow[2]);
      gd->data = new double[ncells];

      for (int i=0; i<ncells; ++i)
        gd->data[i] = val;
    }
    else
    {
      if (nodeCentered == true)
      {
        int glow[3], ghigh[3];
        getBounds(glow,ghigh,varType, levelInfo);
        patchInfo.getBounds(qlow,qhigh,varType);
          
        for (int j=0; j<3; ++j)
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
      for (int i=0; i<stepInfo->levelInfo.size(); ++i)
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
      int ncells = (qhigh[0]-qlow[0])*(qhigh[1]-qlow[1])*(qhigh[2]-qlow[2]);
      
      CheckNaNs(ncells*gd->components, gd->data, level, local_patch);
      
      if(VisIt_VariableData_alloc(&varH) == VISIT_OKAY)
      {
        VisIt_VariableData_setDataD(varH, VISIT_OWNER_SIM, gd->components,
                                    ncells * gd->components, gd->data);
        
        // vtkDoubleArray *rv = vtkDoubleArray::New();
        // rv->SetNumberOfComponents(gd->components);
        // rv->SetArray(gd->data, ncells*gd->components, 0);
          
        // ARS - FIX ME - CHECK FOR LEAKS
        // don't delete gd->data - vtk owns it now!
          // delete gd;
      }
    }
    else
    {
      // Create at new grid data for the values.
      gd = new GridDataRaw;

      for (int i=0; i<3; ++i)
      {
        gd->low[i ] = qlow[i];
        gd->high[i] = qhigh[i];
      }

      gd->components = 1;

      int ncells = (qhigh[0]-qlow[0])*(qhigh[1]-qlow[1])*(qhigh[2]-qlow[2]);
      gd->data = new double[ncells];

      for (int i=0; i<ncells; ++i)
        gd->data[i] = 0;
    }
  }

  return varH;
}

//---------------------------------------------------------------------
// visit_SimGetDomainList
//     Callback for processing a domain list
//---------------------------------------------------------------------
visit_handle visit_SimGetDomainList(const char *name, void *cbdata)
{
  if( Parallel::usingMPI() )
  {
    int par_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);

    visit_simulation_data *sim = (visit_simulation_data *)cbdata;

    SchedulerP schedulerP = sim->simController->getSchedulerP();
    GridP      gridP      = sim->gridP;
    
    TimeStepInfo* &stepInfo = sim->stepInfo;
    
    LoadBalancer* lb = schedulerP->getLoadBalancer();
    
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
        if( par_rank == lb->getPatchwiseProcessorAssignment(patch) )
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
