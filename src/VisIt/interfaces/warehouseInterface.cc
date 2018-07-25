/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

/*
 *  insituUtils.cc: Provides an interface between the Uintah data warehouse 
 *                  and VisIt's libsim in situ interface.
 *
 *  Written by:
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2018
 *
 */

#include <VisIt/interfaces/warehouseInterface.h>
#include <VisIt/interfaces/utils.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace Uintah;

namespace {
  // Dout  dbgOut("VisItWarehouseInterface", "VisIt", "Data warehoue interface to VisIt", false);
}

/////////////////////////////////////////////////////////////////////
// Interface between VisIt's libsim and Uintah
namespace Uintah {

/////////////////////////////////////////////////////////////////////
// Get all the information that may be needed for the current timestep,
// including variable/material info, and level/patch info
// This uses the scheduler for in-situ.
TimeStepInfo* getTimeStepInfo(SchedulerP schedulerP,
                              GridP gridP,
                              LoadExtra loadExtraElements)
{
  DataWarehouse* dw = schedulerP->getLastDW();
  LoadBalancer * lb = schedulerP->getLoadBalancer();

  int numLevels = gridP->numLevels();
  TimeStepInfo *stepInfo = new TimeStepInfo();
  stepInfo->levelInfo.resize(numLevels);

  // Get the material information from the scheduler.
  Scheduler::VarLabelMaterialMap* pLabelMatlMap =
    schedulerP->makeVarLabelMaterialMap();

  std::set<const VarLabel*, VarLabel::Compare> varLabels;
  std::set<const VarLabel*, VarLabel::Compare>::iterator varIter;
  
  // Loop through all of the required and computed variables.
  for (int i=0; i<2; ++i )
  {
    if( i == 0 )
        varLabels = schedulerP->getInitialRequiredVars();
    else
        varLabels = schedulerP->getComputedVars();

    for (varIter = varLabels.begin(); varIter != varLabels.end(); ++varIter )
    {      
      const VarLabel *varLabel = *varIter;

      VariableInfo varInfo;
      varInfo.name = varLabel->getName();
      varInfo.type = varLabel->typeDescription()->getName();

      // Loop through all of the materials for this variable
      Scheduler::VarLabelMaterialMap::iterator matMapIter =
        pLabelMatlMap->find( varInfo.name );

      if( matMapIter != pLabelMatlMap->end() )
      {
        std::list< int > &materials = matMapIter->second;
        std::list< int >::iterator matIter;

        for (matIter = materials.begin(); matIter != materials.end(); ++matIter)
        {
          const int material = *matIter;

          // Check to make sure the variable exists on at least one patch
          // for at least one level.
          bool exists = false;
          
          for (int l=0; l<numLevels; ++l)
          {
            LevelP level = gridP->getLevel(l);
            int numPatches = level->numPatches();
            
            for (int p=0; p<numPatches; ++p)
            {
              const Patch* patch = level->getPatch(p);
              
              if( dw->exists( varLabel, material, patch ) )
              {
                // The variable exists on this level and patch.
                varInfo.materials.push_back( material );
                exists = true;
                break;
              }
            }
            
            if( exists == true )
              break;
          }
        }
        
        stepInfo->varInfo.push_back( varInfo );
      }
    }
  }
  
  // Get the level information
  for (int l=0; l<numLevels; ++l)
  {
    LevelInfo &levelInfo = stepInfo->levelInfo[l];
    LevelP level = gridP->getLevel(l);

    copyIntVector(levelInfo.refinementRatio, level->getRefinementRatio());
    copyVector(   levelInfo.spacing,         level->dCell());
    copyVector(   levelInfo.anchor,          level->getAnchor());
    copyIntVector(levelInfo.periodic,        level->getPeriodicBoundaries());

    // Patch info
    int numPatches = level->numPatches();
    levelInfo.patchInfo.resize(numPatches);

    for (int p=0; p<numPatches; ++p)
    {
      const Patch* patch = level->getPatch(p);
      PatchInfo &patchInfo = levelInfo.patchInfo[p];

      // If the user wants to see extra cells, just include them and
      // let VisIt believe they are part of the original data. This is
      // accomplished by setting <meshtype>_low and <meshtype>_high to
      // the extra cell boundaries so that VisIt is none the wiser.
      if (loadExtraElements == NONE)
      {
        patchInfo.setBounds(&patch->getCellLowIndex()[0],
                            &patch->getCellHighIndex()[0], "CC_Mesh");
        patchInfo.setBounds(&patch->getNodeLowIndex()[0],
                            &patch->getNodeHighIndex()[0], "NC_Mesh");
        patchInfo.setBounds(&patch->getSFCXLowIndex()[0],
                            &patch->getSFCXHighIndex()[0], "SFCX_Mesh");
        patchInfo.setBounds(&patch->getSFCYLowIndex()[0],
                            &patch->getSFCYHighIndex()[0], "SFCY_Mesh");
        patchInfo.setBounds(&patch->getSFCZLowIndex()[0],
                            &patch->getSFCZHighIndex()[0], "SFCZ_Mesh");
      }
      else if (loadExtraElements == CELLS)
      {
        patchInfo.setBounds(&patch->getExtraCellLowIndex()[0],
                            &patch->getExtraCellHighIndex()[0], "CC_Mesh");
        patchInfo.setBounds(&patch->getExtraNodeLowIndex()[0],
                            &patch->getExtraNodeHighIndex()[0], "NC_Mesh");
        patchInfo.setBounds(&patch->getExtraSFCXLowIndex()[0],
                            &patch->getExtraSFCXHighIndex()[0], "SFCX_Mesh");
        patchInfo.setBounds(&patch->getExtraSFCYLowIndex()[0],
                            &patch->getExtraSFCYHighIndex()[0], "SFCY_Mesh");
        patchInfo.setBounds(&patch->getExtraSFCZLowIndex()[0],
                            &patch->getExtraSFCZHighIndex()[0], "SFCZ_Mesh");
      }
      else if (loadExtraElements == PATCHES)
      {
        IntVector ilow(patch->getCellLowIndex()[0],
                       patch->getCellLowIndex()[1],
                       patch->getCellLowIndex()[2]);

        IntVector ihigh(patch->getCellHighIndex()[0],
                        patch->getCellHighIndex()[1],
                        patch->getCellHighIndex()[2]);

        // Test code to extend the patch when extra cells are
        // present. In this case if extra cells are present instead of
        // just adding the extra cell add in additional cells to make
        // up a complete patch.
        for (int i=0; i<3; ++i) {

          // Check for extra cells.
          if( patch->getExtraCellLowIndex()[i] != patch->getCellLowIndex()[i] )
            ilow[i] -= level->getRefinementRatio()[i];

          if( patch->getExtraCellHighIndex()[i] != patch->getCellHighIndex()[i] )
            ihigh[i] -= level->getRefinementRatio()[i];
        }

        patchInfo.setBounds(&ilow[0], &ihigh[0], "CC_Mesh");

        // Test code not implemented for these meshes.        
        patchInfo.setBounds(&patch->getExtraNodeLowIndex()[0],
                            &patch->getExtraNodeHighIndex()[0], "NC_Mesh");
        patchInfo.setBounds(&patch->getExtraSFCXLowIndex()[0],
                            &patch->getExtraSFCXHighIndex()[0], "SFCX_Mesh");
        patchInfo.setBounds(&patch->getExtraSFCYLowIndex()[0],
                            &patch->getExtraSFCYHighIndex()[0], "SFCY_Mesh");
        patchInfo.setBounds(&patch->getExtraSFCZLowIndex()[0],
                            &patch->getExtraSFCZHighIndex()[0], "SFCZ_Mesh");
      }

      patchInfo.setBounds(&patch->neighborsLow()[0],
                          &patch->neighborsHigh()[0], "NEIGHBORS");

      // Set the patch id
      patchInfo.setPatchId(patch->getID());

      // Set the processor id
      patchInfo.setProcId( lb->getPatchwiseProcessorAssignment(patch) );
      
      // Set the number of nodes
      patchInfo.setNumNodes( 0 );
    }
  }

  delete pLabelMatlMap;
  
  return stepInfo;
}


// ****************************************************************************
//  Method: GetLevelAndLocalPatchNumber
//
//  Purpose:
//      Translates the global patch identifier to a refinement level and patch
//      number local to that refinement level.
//  
//  Programmer: sshankar, taken from implementation of the plugin, CHOMBO
//  Creation:   May 20, 2008
//
// ****************************************************************************
void GetLevelAndLocalPatchNumber(TimeStepInfo* stepInfo,
                                 int global_patch, 
                                 int &level, int &local_patch)
{
  int num_levels = stepInfo->levelInfo.size();
  int num_patches = 0;
  int tmp = global_patch;
  level = 0;

  while (level < num_levels)
  {
    num_patches = stepInfo->levelInfo[level].patchInfo.size();

    if (tmp < num_patches)
      break;

    tmp -= num_patches;
    level++;
  }

  local_patch = tmp;
}


// ****************************************************************************
//  Method: GetGlobalDomainNumber
//
//  Purpose:
//      Translates the level and local patch number into a global patch id.
//  
// ****************************************************************************
int GetGlobalDomainNumber(TimeStepInfo* stepInfo,
                          int level, int local_patch)
{
  int g = 0;

  for (int l=0; l<level; l++)
    g += stepInfo->levelInfo[l].patchInfo.size();
  g += local_patch;

  return g;
}


// ****************************************************************************
//  Method: CheckNaNs
//
//  Purpose:
//      Check for and warn about NaN values in the file.
//
//  Arguments:
//      num        data size
//      data       data
//      level      level that contains this patch
//      patch      patch that contains these cells
//
//  Returns:    none
//
//  Programmer: cchriste
//  Creation:   06.02.2012
//
//  Modifications:
const double NAN_REPLACE_VAL = 1.0E9;

void CheckNaNs(double *data, const int num,
               const char* varname, const int level, const int patch)
{
  // replace nan's with a large negative number
  std::vector<int> nanCells;

  for (int i=0; i<num; ++i) 
  {
    if (std::isnan(data[i]))
    {
      data[i] = NAN_REPLACE_VAL;
      nanCells.push_back(i);
    }
  }

  if (!nanCells.empty())
  {
    std::stringstream sstr;
    sstr << "NaNs exist for variable " << varname
         << " in patch " << patch << " of level " << level
         << " and " << nanCells.size() << "/" << num
         << " cells have been replaced by the value "
         <<  NAN_REPLACE_VAL << ".";

    // if ((int)nanCells.size()>40)
    // {
    //   sstr << std::endl << "First 20: ";

    //   for (int i=0;i<(int)nanCells.size() && i<20;++i)
    //     sstr << nanCells[i] << ",";

    //   sstr << std::endl << "Last 20: ";

    //   for (int i=(int)nanCells.size()-21;i<(int)nanCells.size();++i)
    //     sstr << nanCells[i] << ",";
    // }
    // else
    // {
    //   for (int i=0;i<(int)nanCells.size();++i)
    //     sstr << nanCells[i] << ((int)nanCells.size()!=(i+1)?",":".");
    // }

    std::cerr << "Uintah/VisIt Libsim warning : " << sstr.str() << std::endl;
  }
}


/////////////////////////////////////////////////////////////////////
// Read the grid data for the given index range
// This uses the scheduler for in-situ.
template<template <typename> class VAR, typename T>
static GridDataRaw* readGridData(DataWarehouse *dw,
                                 const Patch *patch,
                                 const LevelP level,
                                 const VarLabel *varLabel,
                                 int material,
                                 int low[3],
                                 int high[3],
                                 LoadExtra loadExtraElements)
{
  if( dw->exists( varLabel, material, patch ) )
  {
    GridDataRaw *gd = new GridDataRaw;
    gd->components = numComponents<T>();
    
    int dims[3];
    for (int i=0; i<3; ++i) {
      gd->low[i]  =  low[i];
      gd->high[i] = high[i];
      
      dims[i] = high[i] - low[i];
    }
    
    gd->num = dims[0] * dims[1] * dims[2];
    gd->data = new double[gd->num*gd->components];
    
    VAR<T> var;

    // This queries just the patch
    if( loadExtraElements == NONE )
    {
      IntVector ilow(  low[0],  low[1],  low[2]);
      IntVector ihigh(high[0], high[1], high[2]);

      dw->getRegion( var, varLabel, material, level.get_rep(), ilow, ihigh );
    }
    // This queries the entire patch, including extra cells and boundary cells
    else if( loadExtraElements == CELLS )
    {
      dw->get( var, varLabel, material, patch, Ghost::None, 0 );
    }
    else if( loadExtraElements == PATCHES )
    {
      // This call does not work properly as it will return garbage
      // where the cells do not exists.
      
      // IntVector ilow(  low[0],  low[1],  low[2]);
      // IntVector ihigh(high[0], high[1], high[2]);

      // dw->getRegion( var, varLabel, material, level.get_rep(), ilow, ihigh );

      // This queries the entire patch, including extra cells and
      // boundary cells which is smaller than the requeste region.
      dw->get( var, varLabel, material, patch, Ghost::None, 0 );
    }
    
    const T *p = var.getPointer();

    IntVector tmplow = var.getLowIndex();
    IntVector tmphigh = var.getHighIndex();
    IntVector size;

    for (int i=0; i<3; ++i) {
      size[i] = tmphigh[i] - tmplow[i];
    }

    // Fail safe option if the data returned does match the data requested.
    if(  low[0] !=  tmplow[0] ||  low[1] !=  tmplow[1] ||  low[2] !=  tmplow[2] ||
        high[0] != tmphigh[0] || high[1] != tmphigh[1] || high[2] != tmphigh[2] )
    {
      // std::cerr << __LINE__ << "  " << varLabel->getName() << "  "
      //           << dims[0] << "  " << dims[1] << "  " << dims[2] << "     "
      //           << size[0] << "  " << size[1] << "  " << size[2] << "     "
      
      //           << low[0] << "  " << tmplow[0] << "  "
      //           << low[1] << "  " << tmplow[1] << "  "
      //           << low[2] << "  " << tmplow[2] << "    "
      
      //           << high[0] << "  " << tmphigh[0] << "  "
      //           << high[1] << "  " << tmphigh[1] << "  "
      //           << high[2] << "  " << tmphigh[2] << "  "
      //           << std::endl;
      
      for (int i=0; i<gd->num*gd->components; ++i)
        gd->data[i] = 0;
      
      int kd = 0, jd = 0, id;
      int ks = 0, js = 0, is;
      
      for (int k=low[2]; k<high[2]; ++k)
      {
        if( tmplow[2] <= k && k < tmphigh[2] )
        {
          kd = (k-   low[2]) * dims[1] * dims[0];
          ks = (k-tmplow[2]) * size[1] * size[0];
        
          for (int j=low[1]; j<high[1]; ++j)
          {
            if( tmplow[1] <= j && j < tmphigh[1] )
            {
              jd = kd + (j-   low[1]) * dims[0];
              js = ks + (j-tmplow[1]) * size[0];
          
              for (int i=low[0]; i<high[0]; ++i)
              {
                if( tmplow[0] <= i && i < tmphigh[0] )
                {
                  id = jd + (i-   low[0]);
                  is = js + (i-tmplow[0]);
            
                  copyComponents<T>(&gd->data[id*gd->components], p[is]);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      for (int i=0; i<gd->num; ++i)
        copyComponents<T>(&gd->data[i*gd->components], p[i]);
    }
  
    return gd;
  }
  else
  {
    return nullptr;
  }
}

/////////////////////////////////////////////////////////////////////
// Read the grid data for the given index range
// This uses the scheduler for in-situ.
template<template <typename> class VAR, typename T>
static GridDataRaw* readPatchData(DataWarehouse *dw,
                                  const Patch *patch,
                                  const LevelP level,
                                  const VarLabel *varLabel,
                                  int material)
{
  if( dw->exists( varLabel, material, patch ) )
  {
    GridDataRaw *gd = new GridDataRaw;
    gd->components = numComponents<T>();

    gd->num = 1;
    gd->data = new double[gd->num*gd->components];
    
    if (varLabel->getName() == "refinePatchFlag")
    {
      VAR< PatchFlagP > refinePatchFlag;

      // This queries the entire patch, including extra cells and boundary cells
      dw->get(refinePatchFlag, varLabel, material, patch);
      
      const T p = refinePatchFlag.get().get_rep()->flag;

      for (int i=0; i<gd->num; ++i)
        copyComponents<T>(&gd->data[i*gd->components], p);
    }

    else if (varLabel->getName().find("FileInfo") == 0 ||
             varLabel->getName().find("CellInformation") == 0 ||
             varLabel->getName().find("CutCellInfo") == 0)
    {
      for (int i=0; i<gd->num*gd->components; ++i)
        gd->data[i] = 0;
    }
    else
    {
      VAR<T> var;
      PerPatchBase* patchVar = dynamic_cast<PerPatchBase*>(&var);

      // This queries the entire patch, including extra cells and boundary cells
      dw->get(*patchVar, varLabel, material, patch);

      const T *p = (T*) patchVar->getBasePointer();

      for (int i=0; i<gd->num; ++i)
        copyComponents<T>(&gd->data[i*gd->components], *p);

      // std::cerr << *p << " ************* "
      // 		<< gd->data[0] << " ************* " 
      // 		<< std::endl;
    }
  
    return gd;
  }
  else
  {
    return nullptr;
  }
}


/////////////////////////////////////////////////////////////////////
// Read the grid data for a given patch.
// This uses the scheduler for in-situ.
template<template<typename> class VAR>
GridDataRaw* getGridDataMainType(DataWarehouse *dw,
                                 const Patch *patch,
                                 const LevelP level,
                                 const VarLabel *varLabel,
                                 int material,
                                 int low[3],
                                 int high[3],
                                 LoadExtra loadExtraElements,
                                 const TypeDescription *subtype)
{  
  switch (subtype->getType())
  {
  case TypeDescription::double_type:
    return readGridData<VAR, double>(dw, patch, level, varLabel,
                                     material, low, high, loadExtraElements);
  case TypeDescription::float_type:
    return readGridData<VAR, float>(dw, patch, level, varLabel,
                                    material, low, high, loadExtraElements);
  case TypeDescription::int_type:
    return readGridData<VAR, int>(dw, patch, level, varLabel,
                                  material, low, high, loadExtraElements);
  case TypeDescription::Vector:
    return readGridData<VAR, Vector>(dw, patch, level, varLabel,
                                     material, low, high, loadExtraElements);
  case TypeDescription::Stencil7:
    return readGridData<VAR, Stencil7>(dw, patch, level, varLabel,
                                       material, low, high, loadExtraElements);
  case TypeDescription::Stencil4:
    return readGridData<VAR, Stencil4>(dw, patch, level, varLabel,
                                       material, low, high, loadExtraElements);
  case TypeDescription::Matrix3:
    return readGridData<VAR, Matrix3>(dw, patch, level, varLabel,
                                      material, low, high, loadExtraElements);
  case TypeDescription::bool_type:
  case TypeDescription::short_int_type:
  case TypeDescription::long_type:
  case TypeDescription::long64_type:
    std::cerr << "Uintah/VisIt Libsim getGridDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is not implemented." << std::endl;
    return nullptr;
  default:
    std::cerr << "Uintah/VisIt Libsim getGridDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is unkwown." << std::endl;
    return nullptr;
  }
}

/////////////////////////////////////////////////////////////////////
// Read the grid data for a given patch.
// This uses the scheduler for in-situ.
template<template<typename> class VAR>
GridDataRaw* getPatchDataMainType(DataWarehouse *dw,
                                  const Patch *patch,
                                  const LevelP level,
                                  const VarLabel *varLabel,
                                  int material,
                                  const TypeDescription *subtype)
{
  // printTask( patch, dbgOut, "getPatchDataMainType" );
  
  switch (subtype->getType())
  {
  case TypeDescription::double_type:
    return readPatchData<VAR, double>(dw, patch, level, varLabel, material);
  case TypeDescription::float_type:
    return readPatchData<VAR,  float>(dw, patch, level, varLabel, material);
  case TypeDescription::int_type:
    return readPatchData<VAR,    int>(dw, patch, level, varLabel, material);
  case TypeDescription::Vector:
  case TypeDescription::Stencil7:
  case TypeDescription::Stencil4:
  case TypeDescription::Matrix3:
  case TypeDescription::bool_type:
  case TypeDescription::short_int_type:
  case TypeDescription::long_type:
  case TypeDescription::long64_type:
    std::cerr << "Uintah/VisIt Libsim getPatchDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is not implemented." << std::endl;
    return nullptr;
  default:
    std::cerr << "Uintah/VisIt Libsim getPatchDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is unkwown." << std::endl;
    return nullptr;
  }
}


/////////////////////////////////////////////////////////////////////
// Read the grid data for a given patch.
// This uses the scheduler for in-situ.
GridDataRaw* getGridData(SchedulerP schedulerP,
                         GridP gridP,
                         int level_i,
                         int patch_i,
                         std::string variable_name,
                         int material,
                         int low[3],
                         int high[3],
                         LoadExtra loadExtraElements)
{
  DataWarehouse *dw = schedulerP->getLastDW();
  
  LevelP level = gridP->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // Get variable type from the scheduler.
  const VarLabel* varLabel                = nullptr;
  const TypeDescription* maintype = nullptr;
  const TypeDescription* subtype  = nullptr;

  std::set<const VarLabel*, VarLabel::Compare> varLabels;
  std::set<const VarLabel*, VarLabel::Compare>::iterator varIter;

  // Loop through all of the required and computed variables.
  for (int i=0; i<2; ++i )
  {
    if( i == 0 )
        varLabels = schedulerP->getInitialRequiredVars();
    else
        varLabels = schedulerP->getComputedVars();
        
    for (varIter = varLabels.begin(); varIter != varLabels.end(); ++varIter )
    {      
      varLabel = *varIter;
    
      if (varLabel->getName() == variable_name)
      {
        maintype = varLabel->typeDescription();
        subtype = varLabel->typeDescription()->getSubType();
        break;
      }
    }
  }

  if (!maintype || !subtype)
  {
    std::cerr << "Uintah/VisIt Libsim Error: couldn't find variable type "
              << variable_name << "  "
              << (maintype ? maintype->getName() : " no main type" ) << "  "
              << ( subtype ?  subtype->getName() : " no subtype" ) << "  "
              << std::endl;
    return nullptr;
  }

  switch(maintype->getType())
  {
  case TypeDescription::CCVariable:
    return getGridDataMainType<constCCVariable>(dw, patch, level,
                                                varLabel, material,
                                                low, high, loadExtraElements, subtype);
  case TypeDescription::NCVariable:
    return getGridDataMainType<constNCVariable>(dw, patch, level,
                                                varLabel, material,
                                                low, high, loadExtraElements, subtype);
  case TypeDescription::SFCXVariable:
    return getGridDataMainType<constSFCXVariable>(dw, patch, level,
                                                  varLabel, material,
                                                  low, high, loadExtraElements, subtype);
  case TypeDescription::SFCYVariable:
    return getGridDataMainType<constSFCYVariable>(dw, patch, level,
                                                  varLabel, material,
                                                  low, high, loadExtraElements, subtype);
  case TypeDescription::SFCZVariable:
    return getGridDataMainType<constSFCZVariable>(dw, patch, level,
                                                  varLabel, material,
                                                  low, high, loadExtraElements, subtype);
  case TypeDescription::PerPatch:
    return getPatchDataMainType<PerPatch>(dw, patch, level,
                                          varLabel, material, subtype);
  default:
    std::cerr << "Uintah/VisIt Libsim Error: unknown type: "
              << maintype->getName() << " for variable: "
              << variable_name << std::endl;
    return nullptr;
  }
}


/////////////////////////////////////////////////////////////////////
// Read all the particle data for a given patch.
// This uses the scheduler for in-situ.
template<typename T>
ParticleDataRaw* readParticleData(SchedulerP schedulerP,
                                  const Patch *patch,
                                  const VarLabel *varLabel,
                                  int material)
{
  DataWarehouse *dw = schedulerP->getLastDW();

  std::string variable_name = varLabel->getName();

  ParticleDataRaw *pd = new ParticleDataRaw;
  pd->components = numComponents<T>();
  pd->num = 0;

  // get the material information for all variables
  Scheduler::VarLabelMaterialMap* pLabelMatlMap =
    schedulerP->makeVarLabelMaterialMap();

  // get the materials for this variable
  Scheduler::VarLabelMaterialMap::iterator matMapIter =
    pLabelMatlMap->find( variable_name );

  // figure out which material we're interested in
  std::list< int > &allMatls = matMapIter->second;
  std::list< int > matlsForVar;

  if (material < 0)
  {
    matlsForVar = allMatls;
  }
  else
  {
    // make sure the patch has the variable - use empty material set
    // if it doesn't
    for (std::list< int >::iterator matIter = allMatls.begin();
         matIter != allMatls.end(); matIter++)
    {
      if( *matIter == material )
      {
        matlsForVar.push_back(material);
        break;
      }
    }
  }

  // first get all the particle subsets so that we know how many total
  // particles we'll have
  std::vector<constParticleVariable<T>*> particle_vars;

  for( std::list< int >::iterator matIter = matlsForVar.begin();
       matIter != matlsForVar.end(); matIter++ )
  {
    const int material = *matIter;

    constParticleVariable<T> *var = new constParticleVariable<T>;

    if( dw->exists( varLabel, material, patch ) )
      dw->get( *var, varLabel, material, patch);

    //archive->query(*var, variable_name, matl, patch, timestep);

    particle_vars.push_back(var);
    pd->num += var->getParticleSubset()->numParticles();
  }

  // figure out which material we're interested in
  // ConsecutiveRangeSet allMatls =
  //   archive->queryMaterials(variable_name, patch, timestep);

  // ConsecutiveRangeSet matlsForVar;

  // if (material < 0)
  // {
  //   matlsForVar = allMatls;
  // }
  // else
  // {
       // make sure the patch has the variable - use empty material set
       // if it doesn't
  //   if (0 < allMatls.size() && allMatls.find(material) != allMatls.end())
  //     matlsForVar.addInOrder(material);
  // }

  // first get all the particle subsets so that we know how many total
  // particles we'll have
  // std::vector<ParticleVariable<T>*> particle_vars;

  // for( ConsecutiveRangeSet::iterator matlIter = matlsForVar.begin();
  //      matlIter != matlsForVar.end(); matlIter++ )
  // {
  //   int matl = *matlIter;

  //   ParticleVariable<T> *var = new ParticleVariable<T>;
  //   archive->query(*var, variable_name, matl, patch, timestep);

  //   particle_vars.push_back(var);
  //   pd->num += var->getParticleSubset()->numParticles();
  // }

  // copy all the data
  int pi = 0;

  pd->data = new double[pd->components * pd->num];

  for (unsigned int i=0; i<particle_vars.size(); ++i)
  {
    ParticleSubset *pSubset = particle_vars[i]->getParticleSubset();

    for (ParticleSubset::iterator p = pSubset->begin();
         p != pSubset->end(); ++p)
    {

      //TODO: need to be able to read data as array of longs for
      //particle id, but copyComponents always reads double
      copyComponents<T>(&pd->data[pi*pd->components],
                        (*particle_vars[i])[*p]);
      ++pi;
    }
  }

  // cleanup
  for (unsigned int i=0; i<particle_vars.size(); ++i)
    delete particle_vars[i];

  delete pLabelMatlMap;

  return pd;
}


/////////////////////////////////////////////////////////////////////
// Read all the particle data for a given patch.
// This uses the scheduler for in-situ.
ParticleDataRaw* getParticleData(SchedulerP schedulerP,
                                 GridP gridP,
                                 int level_i,
                                 int patch_i,
                                 std::string variable_name,
                                 int material)
{
  LevelP level = gridP->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // printTask( patch, dbgOut, "getParticleData variable: " + variable_name );
  
  // get the variable information
  const VarLabel* varLabel                = nullptr;
  const TypeDescription* maintype = nullptr;
  const TypeDescription* subtype  = nullptr;

  std::set<const VarLabel*, VarLabel::Compare> varLabels;
  std::set<const VarLabel*, VarLabel::Compare>::iterator varIter;

  // Loop through all of the required and computed variables
  for (int i=0; i<2; ++i )
  {
    if( i == 0 )
        varLabels = schedulerP->getInitialRequiredVars();
    else
        varLabels = schedulerP->getComputedVars();

    for (varIter = varLabels.begin(); varIter != varLabels.end(); ++varIter)
    {
      varLabel = *varIter;
      
      if (varLabel->getName() == variable_name)
      {
        maintype = varLabel->typeDescription();
        subtype = varLabel->typeDescription()->getSubType();
        
        break;
      }
    }
  }

  if (!maintype || !subtype) {
    std::cerr << "Uintah/VisIt Libsim Error: couldn't find variable type"
              << variable_name << "  "
              << (maintype ? maintype->getName() : " no main type" ) << "  "
              << ( subtype ?  subtype->getName() : " no subtype" ) << "  "
              << std::endl;
    return nullptr;
  }

  switch (subtype->getType()) {
  case TypeDescription::double_type:
    return readParticleData<double>(schedulerP, patch, varLabel, material);
  case TypeDescription::float_type:
    return readParticleData<float>(schedulerP, patch, varLabel, material);
  case TypeDescription::int_type:
    return readParticleData<int>(schedulerP, patch, varLabel, material);
  case TypeDescription::long64_type:
    return readParticleData<long64>(schedulerP, patch, varLabel, material);
  case TypeDescription::Point:
    return readParticleData<Point>(schedulerP, patch, varLabel, material);
  case TypeDescription::Vector:
    return readParticleData<Vector>(schedulerP, patch, varLabel, material);
  case TypeDescription::IntVector:
    return readParticleData<IntVector>(schedulerP, patch, varLabel, material);
  case TypeDescription::Stencil7:
    return readParticleData<Stencil7>(schedulerP, patch, varLabel, material);
  case TypeDescription::Stencil4:
    return readParticleData<Stencil4>(schedulerP, patch, varLabel, material);
  case TypeDescription::Matrix3:
    return readParticleData<Matrix3>(schedulerP, patch, varLabel, material);
  default:
    std::cerr << "Uintah/VisIt Libsim Error: " 
              << "unknown subtype for particle data: " << subtype->getName()
              << " for particle vairable: " << variable_name << std::endl;
    return nullptr;
  }
}

}
