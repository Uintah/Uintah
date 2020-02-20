/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
 *  warehouseInterface.cc: Provides an interface between Uintah's data
 *                         warehouse and VisIt's libsim in situ interface.
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
                              LoadExtraGeometry loadExtraGeometry,
                              LoadVariables     loadVariables)
{
  DataWarehouse* dw = schedulerP->getLastDW();
  LoadBalancer * lb = schedulerP->getLoadBalancer();
  Output       * output = schedulerP->getOutput();

  int numLevels = gridP->numLevels();
  TimeStepInfo *stepInfo = new TimeStepInfo();
  stepInfo->levelInfo.resize(numLevels);

  // Get the material information from the scheduler.
  Scheduler::VarLabelMaterialMap* pLabelMatlMap =
    schedulerP->makeVarLabelMaterialMap();

  std::set<const VarLabel*, VarLabel::Compare> varLabels;
  std::set<const VarLabel*, VarLabel::Compare>::iterator varIter;

  // Loop through all of the required or computed variables.
  if( schedulerP->isRestartInitTimestep() ||
      loadVariables == LOAD_CHECKPOINT_VARIABLES )
    varLabels = schedulerP->getInitialRequiredVars();
  else
    varLabels = schedulerP->getComputedVars();

  for (varIter = varLabels.begin(); varIter != varLabels.end(); ++varIter )
  {
    const VarLabel *varLabel = *varIter;

    if( // Toss out variables not being saved.
        (loadVariables == LOAD_OUTPUT_VARIABLES &&
         !output->isLabelSaved( varLabel->getName() )) ||

        // Toss out variables not being checkpointed.
        (loadVariables == LOAD_CHECKPOINT_VARIABLES &&
         schedulerP->getNotCheckPointVars().find( varLabel->getName() ) !=
         schedulerP->getNotCheckPointVars().end() ) )
      continue;

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
          const LevelP &level = gridP->getLevel(l);
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
  
  delete pLabelMatlMap;
  
  const std::string meshTypes[5] = { "NC_MESH", "CC_MESH", 
                                     "SFCX_MESH", "SFCY_MESH", "SFCZ_MESH" };
  
  const Patch::VariableBasis basis[5] = { Patch::NodeBased,
                                          Patch::CellBased,
                                          Patch::XFaceBased,
                                          Patch::YFaceBased,
                                          Patch::ZFaceBased };

  std::map< std::string, std::pair< IntVector, IntVector > > extraPatches;

  // Get the level information
  for (int l=0; l<numLevels; ++l)
  {
    LevelInfo &levelInfo = stepInfo->levelInfo[l];
    const LevelP &level = gridP->getLevel(l);

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

      for( unsigned int m=0; m<5; ++m )
      {
        IntVector iLow, iHigh;

        // If the user wants to see extra cells, just include them and
        // let VisIt believe they are part of the original data. This is
        // accomplished by setting <meshtype>_low and <meshtype>_high to
        // the extra cell boundaries so that VisIt is none the wiser.
        if (loadExtraGeometry == CELLS)
        {
          iLow  = patch->getExtraLowIndex (basis[m], IntVector(0,0,0));
          iHigh = patch->getExtraHighIndex(basis[m], IntVector(0,0,0));
        }
        else if (loadExtraGeometry == PATCHES)
        {
          iLow  = patch->getLowIndex (basis[m]);
          iHigh = patch->getHighIndex(basis[m]);

          IntVector iExtraLow  = patch->getExtraLowIndex (basis[m], IntVector(0,0,0));
          IntVector iExtraHigh = patch->getExtraHighIndex(basis[m], IntVector(0,0,0));

          // if( m == 1 & l == 0 && p == 0 )
          //   std::cerr << iLow << "    " << iHigh << "    "
          //          << iExtraLow << "    " << iExtraHigh << std::endl;

          // IntVector iLowBase, iHighBase;
          
          // for( int k=-1; k<2; ++k )
          // {
          //   if( (k == -1 && iLow[2]  == iExtraLow[2]) ||
          //    (k ==  1 && iHigh[2] == iExtraHigh[2]) )
          //     continue;

          //   if( k == -1 ) {
          //     iLowBase[2] = iLow[2] - level->getRefinementRatio()[2];
          //     iHighBase[2] = iLow[2];
          //   } else if( k == 0 ) {
          //     iLowBase[2] = iLow[2];
          //     iHighBase[2] = iHigh[2];
          //   } else if( k == 1 ) {
          //     iLowBase[2] = iHigh[2];
          //     iHighBase[2] = iHigh[2] + level->getRefinementRatio()[2];
          //   }
            
          //   for( int j=-1; j<2; ++j )
          //   {
          //     if( (j == -1 && iLow[1]  == iExtraLow[1]) ||
          //      (j ==  1 && iHigh[1] == iExtraHigh[1]) )
          //    continue;
              
          //     if( j == -1 ) {
          //    iLowBase[1] = iLow[1] - level->getRefinementRatio()[1];
          //    iHighBase[1] = iLow[1];
          //     } else if( j == 0 ) {
          //    iLowBase[1] = iLow[1];
          //    iHighBase[1] = iHigh[1];
          //     } else if( j == 1 ) {
          //    iLowBase[1] = iHigh[1];
          //    iHighBase[1] = iHigh[1] + level->getRefinementRatio()[1];
          //     }
              
          //     for( int i=-1; i<2; ++i )
          //     {
          //    if( (i == -1 && iLow[0]  == iExtraLow[0]) ||
          //        (i ==  1 && iHigh[0] == iExtraHigh[0]) )
          //      continue;

          //    if( k == 0 && j == 0 && i == 0 )
          //      continue;

          //    if( i == -1 ) {
          //      iLowBase[0] = iLow[0] - level->getRefinementRatio()[0];
          //      iHighBase[0] = iLow[0];
          //    } else if( i == 0 ) {
          //      iLowBase[0] = iLow[0];
          //      iHighBase[0] = iHigh[0];
          //    } else if( i == 1 ) {
          //      iLowBase[0] = iHigh[0];
          //      iHighBase[0] = iHigh[0] + level->getRefinementRatio()[0];
          //    }
                  
          //    std::stringstream hash;
          //    hash << iLowBase[0] << iLowBase[1] << iLowBase[2] << iHighBase[0] << iHighBase[1] << iHighBase[2];

          //    extraPatches[hash.str()] = std::pair< IntVector, IntVector >( iLowBase, iHighBase );

          //    if( m == 1 & l == 0 && p == 0 )
          //      std::cerr << iLowBase << "    " << iHighBase << "    " << i << "  " << j << "  " << k << std::endl;
          //     }
          //   }
          // }

          // Extend the patch when extra elements are present. In this
          // case if extra elements are present instead of just adding
          // the extra element add in additional elements to make up a
          // complete patch.
          for (int i=0; i<3; ++i) {
            
            if( iLow[i] != iExtraLow[i] )
              iLow[i] -= level->getRefinementRatio()[i];
            
            if( iHigh[i] != iExtraHigh[i] )
              iHigh[i] += level->getRefinementRatio()[i];
          }

          // Clamp: don't exceed the limits
          // IntVector lLow, lHigh;
        
          // if( basis[m] == Patch::NodeBased ) {         
          //   level->findNodeIndexRange( lLow, lHigh );
          // }
          // else if( basis[m] == Patch::CellBased ) {
          //   level->findCellIndexRange( lLow, lHigh );
          // }
          // else {
          //   lLow  = iLow;
          //   lHigh = iHigh;
          // }
        
          // iLow  = Uintah::Max(lLow,  iLow);
          // iHigh = Uintah::Min(lHigh, iHigh);
        }
        else //if (loadExtraGeometry == NO_EXTRA_GEOMETRY)
        {
          iLow  = patch->getLowIndex (basis[m]);
          iHigh = patch->getHighIndex(basis[m]);
        }
        
        patchInfo.setBounds(&iLow[0], &iHigh[0], meshTypes[m]);

        // if( m == 1 && l == 0 && p == 0 )
        //   for( const auto & extraPatch : extraPatches )
        //     std::cerr << extraPatch.second.first << "  " << extraPatch.second.second << std::endl;
      }

      patchInfo.setBounds(&patch->neighborsLow()[0],
                          &patch->neighborsHigh()[0], "NEIGHBORS");

      // Set the patch id
      patchInfo.setPatchId(patch->getID());

      // Set the processor id
      patchInfo.setProcId( lb->getPatchwiseProcessorAssignment(patch) );
    }
  }

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
//  Method: mapIndexToCoarser
//
//  Purpose: Map a cell from the current mesh to the coarse mesh
//      
//  
// ****************************************************************************
IntVector
mapIndexToCoarser( const IntVector & idx, const IntVector & refinementRatio )
{
  // If the fine cell index is negative it must be made positive first,
  // then an offset must be subtracted to get the right coarse cell.
  IntVector index = idx;;
  IntVector offset(0, 0, 0);

  for( unsigned int i=0; i<3; ++i ) {
    while( index[i] < 0 ) {
      index[i] += refinementRatio[i];
      ++offset[i];
    }
  }

  return index / refinementRatio - offset;
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
                                 const VarLabel *varLabel,
                                 int material,
                                 int low[3],
                                 int high[3],
                                 LoadExtraGeometry loadExtraGeometry)
{
  if( !dw->exists( varLabel, material, patch ) )
    return nullptr;
  
  GridDataRaw *gd = new GridDataRaw;
  gd->components = numComponents<T>();
  
  IntVector ilow(  low[0],  low[1],  low[2]);
  IntVector ihigh(high[0], high[1], high[2]);
  IntVector dims = ihigh - ilow;
  
  for (int i=0; i<3; ++i) {
    gd->low[i]  =  low[i];
    gd->high[i] = high[i];
  }
  
  gd->num = dims[0] * dims[1] * dims[2];
  gd->data = new double[gd->num*gd->components];
  
  // This queries just the patch
  if( loadExtraGeometry == NO_EXTRA_GEOMETRY )
  {
    VAR<T> var;  
    dw->getRegion( var, varLabel, material, patch->getLevel(), ilow, ihigh );
    const T *p = var.getPointer();
    
    for (int i=0; i<gd->num; ++i)
      copyComponents<T>(&gd->data[i*gd->components], p[i]);
  }
  // This queries the entire patch, including extra cells and boundary cells
  else if( loadExtraGeometry == CELLS )
  {
    VAR<T> var;
    dw->get( var, varLabel, material, patch, Ghost::None, 0 );
    const T *p = var.getPointer();

    for (int i=0; i<gd->num; ++i)
      copyComponents<T>(&gd->data[i*gd->components], p[i]);
  }
  else if( loadExtraGeometry == PATCHES )
  {
    for (int i=0; i<gd->num*gd->components; ++i)
      gd->data[i] = 0;

    // This queries the entire patch, including extra cells and
    // boundary cells which may be smaller than the requested
    // region. But the missing cells will get filled in below.
    VAR<T> var;
    dw->get( var, varLabel, material, patch, Ghost::None, 0 );
    const T *p = var.getPointer();

    IntVector varlow  = var.getLowIndex();
    IntVector varhigh = var.getHighIndex();
    IntVector vardims = varhigh - varlow;

    VAR<T> cvar;
    const T *cp = nullptr;
    IntVector clow, chigh, cvardims(0);

    const Level *level = patch->getLevel();
    const IntVector rRatio = level->getRefinementRatio();
    
    // If cells are missing get the values from the coarser level
    if( varlow != ilow || varhigh != ihigh )
    {
      const Level *coarserLevel = level->getCoarserLevel().get_rep();
      
      clow  = mapIndexToCoarser( ilow,  rRatio );
      chigh = mapIndexToCoarser( ihigh, rRatio );

      // Clamp: don't exceed coarse level limits
      IntVector lLow, lHigh;
      coarserLevel->findCellIndexRange( lLow, lHigh );
      
      clow  = Uintah::Max(lLow,  clow);
      chigh = Uintah::Min(lHigh, chigh); 

      // Get the data from the coarser level.
      dw->getRegion( cvar, varLabel, material, coarserLevel,
                     clow, chigh, true );
      cp = cvar.getPointer();
        
      cvardims = cvar.getHighIndex() - cvar.getLowIndex();
      
      // Copy the coarse level data to all points on the fine level.
      // for (int k=low[2]; k<high[2]; ++k) {

      //   int kd = (k-low[2]) * dims[1] * dims[0];
      
      //   for (int j=low[1]; j<high[1]; ++j) {

      //     int jd = kd + (j-low[1]) * dims[0];
            
      //     for (int i=low[0]; i<high[0]; ++i) {

      //       int id = jd + (i-low[0]);

      //       IntVector tmp = mapIndexToCoarser( IntVector( i, j, k ), rRatio );

      //       int kv =      (tmp[2]-clow[2]) * cvardims[1] * cvardims[0];
      //       int jv = kv + (tmp[1]-clow[1]) * cvardims[0];
      //       int iv = jv + (tmp[0]-clow[0]);

      //       if( clow <= tmp && tmp < chigh )
      //         copyComponents<T>(&gd->data[id*gd->components], cp[iv]);
      //     }
      //   }
      // }

    }

    // Copy the coarse and fine level data
    for (int k=low[2]; k<high[2]; ++k)
    {
      int kd = (k-low[2]) * dims[1] * dims[0];
      
      if( varlow[2] <= k && k < varhigh[2] )
      {
        int kv = (k-varlow[2]) * vardims[1] * vardims[0];
        
        for (int j=low[1]; j<high[1]; ++j)
        {
          int jd = kd + (j-low[1]) * dims[0];
          
          if( varlow[1] <= j && j < varhigh[1] )
          {
            int jv = kv + (j-varlow[1]) * vardims[0];
            
            for (int i=low[0]; i<high[0]; ++i)
            {
              int id = jd + (i-low[0]);

              // Copy the fine level data to a point on the fine level.
              if( varlow[0] <= i && i < varhigh[0] )
              {
                int iv = jv + (i-varlow[0]);
            
                copyComponents<T>(&gd->data[id*gd->components], p[iv]);
              }
              // Copy the coarse level data to a point on the fine level.
              else
              {
                IntVector tmp = mapIndexToCoarser( IntVector( i, j, k ), rRatio );

                int kv =      (tmp[2]-clow[2]) * cvardims[1] * cvardims[0];
                int jv = kv + (tmp[1]-clow[1]) * cvardims[0];
                int iv = jv + (tmp[0]-clow[0]);
            
                if( clow <= tmp && tmp < chigh )
                  copyComponents<T>(&gd->data[id*gd->components], cp[iv]);
              }
            }
          }
          // Copy the coarse level data to each point on the fine level.
          else
          {
            for (int i=low[0]; i<high[0]; ++i)
            {
              int id = jd + (i-low[0]);

              IntVector tmp = mapIndexToCoarser( IntVector( i, j, k ), rRatio );

              int kv =      (tmp[2]-clow[2]) * cvardims[1] * cvardims[0];
              int jv = kv + (tmp[1]-clow[1]) * cvardims[0];
              int iv = jv + (tmp[0]-clow[0]);
            
              if( clow <= tmp && tmp < chigh )
                copyComponents<T>(&gd->data[id*gd->components], cp[iv]);
            }
          }
        }
      }
      // Copy the coarse level data to each point on the fine level.
      else
      {
        for (int j=low[1]; j<high[1]; ++j) {

          int jd = kd + (j-low[1]) * dims[0];
            
          for (int i=low[0]; i<high[0]; ++i) {

            int id = jd + (i-low[0]);

            IntVector tmp = mapIndexToCoarser( IntVector( i, j, k ), rRatio );

            int kv =      (tmp[2]-clow[2]) * cvardims[1] * cvardims[0];
            int jv = kv + (tmp[1]-clow[1]) * cvardims[0];
            int iv = jv + (tmp[0]-clow[0]);
            
            if( clow <= tmp && tmp < chigh )
              copyComponents<T>(&gd->data[id*gd->components], cp[iv]);
          }
        }
      }
    }
  }
  
  return gd;
}

/////////////////////////////////////////////////////////////////////
// Read the grid data for the given index range
// This uses the scheduler for in-situ.
template<template <typename> class VAR, typename T>
static GridDataRaw* readPatchData(DataWarehouse *dw,
                                  const Patch *patch,
                                  const VarLabel *varLabel,
                                  int material)
{
  if( !dw->exists( varLabel, material, patch ) )
    return nullptr;
  
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
    //                << gd->data[0] << " ************* " 
    //                << std::endl;
  }
  
  return gd;
}


/////////////////////////////////////////////////////////////////////
// Read the grid data for a given patch.
// This uses the scheduler for in-situ.
template<template<typename> class VAR>
GridDataRaw* getGridDataMainType(DataWarehouse *dw,
                                 const Patch *patch,
                                 const VarLabel *varLabel,
                                 int material,
                                 int low[3],
                                 int high[3],
                                 LoadExtraGeometry loadExtraGeometry,
                                 const TypeDescription *subtype)
{  
  switch (subtype->getType())
  {
  case TypeDescription::double_type:
    return readGridData<VAR, double>(dw, patch, varLabel,
                                     material, low, high, loadExtraGeometry);
  case TypeDescription::float_type:
    return readGridData<VAR, float>(dw, patch, varLabel,
                                    material, low, high, loadExtraGeometry);
  case TypeDescription::int_type:
    return readGridData<VAR, int>(dw, patch, varLabel,
                                  material, low, high, loadExtraGeometry);
  case TypeDescription::Vector:
    return readGridData<VAR, Vector>(dw, patch, varLabel,
                                     material, low, high, loadExtraGeometry);
  case TypeDescription::Stencil7:
    return readGridData<VAR, Stencil7>(dw, patch, varLabel,
                                       material, low, high, loadExtraGeometry);
  case TypeDescription::Stencil4:
    return readGridData<VAR, Stencil4>(dw, patch, varLabel,
                                       material, low, high, loadExtraGeometry);
  case TypeDescription::Matrix3:
    return readGridData<VAR, Matrix3>(dw, patch, varLabel,
                                      material, low, high, loadExtraGeometry);
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
                                  const VarLabel *varLabel,
                                  int material,
                                  const TypeDescription *subtype)
{
  switch (subtype->getType())
  {
  case TypeDescription::double_type:
    return readPatchData<VAR, double>(dw, patch, varLabel, material);
  case TypeDescription::float_type:
    return readPatchData<VAR,  float>(dw, patch, varLabel, material);
  case TypeDescription::int_type:
    return readPatchData<VAR,    int>(dw, patch, varLabel, material);
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
                         LoadExtraGeometry loadExtraGeometry)
{
  DataWarehouse *dw = schedulerP->getLastDW();
  
  const LevelP &level = gridP->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // Get variable type from the scheduler.
  const VarLabel* varLabel        = nullptr;
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
    return getGridDataMainType<constCCVariable>(dw, patch, varLabel, material,
                                                low, high, loadExtraGeometry, subtype);
  case TypeDescription::NCVariable:
    return getGridDataMainType<constNCVariable>(dw, patch, varLabel, material,
                                                low, high, loadExtraGeometry, subtype);
  case TypeDescription::SFCXVariable:
    return getGridDataMainType<constSFCXVariable>(dw, patch, varLabel, material,
                                                  low, high, loadExtraGeometry, subtype);
  case TypeDescription::SFCYVariable:
    return getGridDataMainType<constSFCYVariable>(dw, patch, varLabel, material,
                                                  low, high, loadExtraGeometry, subtype);
  case TypeDescription::SFCZVariable:
    return getGridDataMainType<constSFCZVariable>(dw, patch, varLabel, material,
                                                  low, high, loadExtraGeometry, subtype);
  case TypeDescription::PerPatch:
    return getPatchDataMainType<PerPatch>(dw, patch, varLabel, material, subtype);
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
unsigned int getNumberParticles(SchedulerP schedulerP,
                                GridP gridP,
                                int level_i,
                                int patch_i,
                                int material)
{
  DataWarehouse *dw = schedulerP->getLastDW();

  const LevelP &level = gridP->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  const std::string &variable_name =
    Uintah::VarLabel::getParticlePositionName();
  
  // get the variable information
  const VarLabel* varLabel = nullptr;
  
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
        break;
      }
    }
  }

  // if( dw->exists( varLabel, material, patch ) )
  //   return 0;
  
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
    
  // Get all the particle subsets and the total number of particles.
  unsigned int numParticles = 0;
  
  for( std::list< int >::iterator matIter = matlsForVar.begin();
       matIter != matlsForVar.end(); matIter++ )
  {
    const int material = *matIter;
      
    constParticleVariable<Point> *var = new constParticleVariable<Point>;
      
    dw->get( *var, varLabel, material, patch);

    numParticles += var->getParticleSubset()->numParticles();

    delete var;
  }

  // cleanup
  delete pLabelMatlMap;
  
  return numParticles;
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

  // if( !dw->exists( varLabel, material, patch ) )
  //   return nullptr;
  
  std::string variable_name = varLabel->getName();
        
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
 
  // Get all the particle subsets and the total number of particles.
  std::vector<constParticleVariable<T>*> particle_vars;
    
  unsigned int numParticles = 0;
  
  for( std::list< int >::iterator matIter = matlsForVar.begin();
       matIter != matlsForVar.end(); matIter++ )
  {
    const int material = *matIter;
      
    constParticleVariable<T> *var = new constParticleVariable<T>;
      
    dw->get( *var, varLabel, material, patch);

    particle_vars.push_back(var);
    numParticles += var->getParticleSubset()->numParticles();
  }

  ParticleDataRaw *pd = nullptr;

  // Copy all the data
  if( numParticles )
  {
    pd = new ParticleDataRaw;
    pd->components = numComponents<T>();
    pd->num = numParticles;
    pd->data = new double[pd->components * pd->num];

    int pi = 0;

    for (unsigned int i=0; i<particle_vars.size(); ++i)
    {
        ParticleSubset *pSubset = particle_vars[i]->getParticleSubset();
        
        for (ParticleSubset::iterator p = pSubset->begin();
             p != pSubset->end(); ++p)
        {       
          // TODO: need to be able to read data as array of longs for
          // particle id, but copyComponents always reads double
          copyComponents<T>(&pd->data[pi*pd->components],
                            (*particle_vars[i])[*p]);
          ++pi;
        }
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
  const LevelP &level = gridP->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // get the variable information
  const VarLabel* varLabel        = nullptr;
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
