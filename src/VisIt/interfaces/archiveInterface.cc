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
 *  archiveUtils.cc: Provides an interface between Uintah's data
 *                   archiver and VisIt's database reader.
 *
 *  Written by:
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2018
 *
 */

#include <VisIt/interfaces/datatypes.h>
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
  // Dout  dbgOut("VisItArchiveInterface", "VisIt", "Data archive interface to VisIt", false);
}

/////////////////////////////////////////////////////////////////////
// Open a data archive.
extern "C"
DataArchive* openDataArchive(const std::string& input_uda_name)
{
  // DOUT(dbgOut, "openDataArchive" );
  
  DataArchive *archive = scinew DataArchive(input_uda_name);

  return archive;
}


/////////////////////////////////////////////////////////////////////
// Close a data archive - the visit plugin itself doesn't know about
// DataArchive::~DataArchive().
extern "C"
void closeDataArchive(DataArchive *archive)
{
  // DOUT(dbgOut, "closeDataArchive" );

  delete archive;
}


/////////////////////////////////////////////////////////////////////
// Get the grid for the current timestep, so we don't have to query
// it over and over.  We return a pointer to the GridP since the 
// visit plugin doesn't actually know about Grid's (or GridP's), and
// so the handle doesn't get destructed.
extern "C"
GridP* getGrid(DataArchive *archive, int timeStepNo)
{
  std::ostringstream msg;
  msg << std::left<< std::setw(50) << "getGrid "<< std::right <<" timestep: " << timeStepNo;
  // DOUT(dbgOut, msg.str() );
  
  GridP *grid = new GridP(archive->queryGrid(timeStepNo));
  return grid;
}


/////////////////////////////////////////////////////////////////////
// Destruct the GridP, which will decrement the reference count.
extern "C"
void releaseGrid(GridP *grid)
{
  // DOUT(dbgOut, "releaseGrid" );

  delete grid;
}


/////////////////////////////////////////////////////////////////////
// Get the time for each cycle.
extern "C"
std::vector<double> getCycleTimes(DataArchive *archive)
{
  // DOUT(dbgOut, "getCycleTimes" );

  // Get the times and indices.
  std::vector<int> index;
  std::vector<double> times;

  // query time info from dataarchive
  archive->queryTimesteps(index, times);

  return times;
} 


/////////////////////////////////////////////////////////////////////
// Get the time for each cycle.
extern "C"
unsigned int queryProcessors(DataArchive *archive)
{
  // DOUT(dbgOut, "queryProcessors" );

  // query number of processors used from dataarchive
  unsigned int nProcs;
  
  archive->queryProcessors( nProcs );

  return nProcs;
} 


/////////////////////////////////////////////////////////////////////
// Get all the information that may be needed for the current timestep,
// including variable/material info, and level/patch info
// This function uses the archive for file reading.
extern "C"
TimeStepInfo* getTimeStepInfo(DataArchive *archive,
                              GridP *grid,
                              int timestep,
                              LoadExtraGeometry loadExtraGeometry)
{
  std::ostringstream msg;
  msg << std::left<< std::setw(50) << "getTimeStepInfo "<< std::right <<" timestep: " << timestep;
  // DOUT(dbgOut, msg.str() );
  
  int numLevels = (*grid)->numLevels();
  TimeStepInfo *stepInfo = new TimeStepInfo();
  stepInfo->levelInfo.resize(numLevels);

  // Get the variable information from the archive.
  std::vector<std::string>                    vars;
  std::vector<const Uintah::TypeDescription*> types;
  std::vector<int>                            numMatls;
  archive->queryVariables( vars, numMatls, types );
  stepInfo->varInfo.resize( vars.size() );

  for (unsigned int i=0; i<vars.size(); ++i) {
    VariableInfo &varInfo = stepInfo->varInfo[i];

    varInfo.name = vars[i];
    varInfo.type = types[i]->getName();

    // Query each level for material info until materials are found.
    for (int l=0; l<numLevels; l++) {
      const LevelP &level = (*grid)->getLevel(l);
      const Patch* patch = *(level->patchesBegin());

      ConsecutiveRangeSet matls =
        archive->queryMaterials(vars[i], patch, timestep);

      if (matls.size() > 0) {

        // Copy the list of materials
        for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
             matlIter != matls.end(); matlIter++)
          varInfo.materials.push_back(*matlIter);

        // Don't query on any more levels.
        break;
      }
    }
  }

  const std::string meshTypes[5] = { "NC_MESH", "CC_MESH", 
                                     "SFCX_MESH", "SFCY_MESH", "SFCZ_MESH" };
  
  const Patch::VariableBasis basis[5] = { Patch::NodeBased,
                                          Patch::CellBased,
                                          Patch::XFaceBased,
                                          Patch::YFaceBased,
                                          Patch::ZFaceBased };
  
  // Get level information
  for (int l=0; l<numLevels; ++l)
  {
    LevelInfo &levelInfo = stepInfo->levelInfo[l];
    const LevelP &level = (*grid)->getLevel(l);

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

          // // Clamp: don't exceed the limits
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
      }

      patchInfo.setBounds(&patch->neighborsLow()[0],
                          &patch->neighborsHigh()[0], "NEIGHBORS");

      // Get the patch id
      patchInfo.setPatchId(patch->getID());
      
      // Get the processor id
      patchInfo.setProcId(archive->queryPatchwiseProcessor(patch, timestep));
    }
  }

  return stepInfo;
}

/////////////////////////////////////////////////////////////////////
// Map a cell from the current mesh to the coarse mesh
//  
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

/////////////////////////////////////////////////////////////////////
// Read the grid data for the given index range
// This function uses the archive for file reading.
template<template <typename> class VAR, typename T>
static GridDataRaw* readGridData(DataArchive *archive,
                                 const Patch *patch,
                                 std::string variable_name,
                                 int material,
                                 int timestep,
                                 int low[3],
                                 int high[3],
                                 LoadExtraGeometry loadExtraGeometry)
{
  if( !archive->exists( variable_name, patch, timestep ) )
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
    archive->queryRegion(var, variable_name, material,
                         patch->getLevel(), timestep, ilow, ihigh);
    const T *p = var.getPointer();

    for (int i=0; i<gd->num; ++i)
      copyComponents<T>(&gd->data[i*gd->components], p[i]);
  }
  // This queries the entire patch, including extra cells and boundary cells
  else if( loadExtraGeometry == CELLS )
  {
    VAR<T> var;
    archive->query(var, variable_name, material, patch, timestep);
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
    archive->query(var, variable_name, material, patch, timestep);
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
      archive->queryRegion(cvar, variable_name, material,
                           coarserLevel, timestep, clow, chigh);
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
// This function uses the archive for file reading.
template<template <typename> class VAR, typename T>
static GridDataRaw* readPatchData(DataArchive *archive,
                                  const Patch *patch,
                                  std::string variable_name,
                                  int material,
                                  int timestep)
{
  if( !archive->exists( variable_name, patch, timestep ) )
    return nullptr;
  
  GridDataRaw *gd = new GridDataRaw;
  gd->components = numComponents<T>();
  
  gd->num = 1;
  gd->data = new double[gd->num*gd->components];
  
  if (variable_name == "refinePatchFlag")
  {
    VAR< PatchFlagP > refinePatchFlag;
    
    // This queries the entire patch, including extra cells and boundary cells
    archive->query(refinePatchFlag, variable_name, material, patch, timestep);
    
    const T p = refinePatchFlag.get().get_rep()->flag;
    
    for (int i=0; i<gd->num; ++i)
      copyComponents<T>(&gd->data[i*gd->components], p);
  }
  else if (variable_name.find("FileInfo") == 0 ||
           variable_name.find("CellInformation") == 0 ||
           variable_name.find("CutCellInfo") == 0)
  {
    for (int i=0; i<gd->num*gd->components; ++i)
      gd->data[i] = 0;
  }
  else
  {
    VAR<T> var;
    PerPatchBase* patchVar = dynamic_cast<PerPatchBase*>(&var);

    // This queries the entire patch, including extra cells and boundary cells
    archive->query(*patchVar, variable_name, material, patch, timestep);

    const T *p = (T*) patchVar->getBasePointer();

    for (int i=0; i<gd->num; ++i)
      copyComponents<T>(&gd->data[i*gd->components], *p);
  }

  return gd;
}


/////////////////////////////////////////////////////////////////////
// Read the grid data based on the type
// This function uses the archive for file reading.
template<template<typename> class VAR>
GridDataRaw* getGridDataMainType(DataArchive *archive,
                                 const Patch *patch,
                                 std::string variable_name,
                                 int material,
                                 int timestep,
                                 int low[3],
                                 int high[3],
                                 LoadExtraGeometry loadExtraGeometry,
                                 const Uintah::TypeDescription *subtype)
{
  switch (subtype->getType()) {
  case Uintah::TypeDescription::double_type:
    return readGridData<VAR, double>(archive, patch, variable_name,
                                     material, timestep, low, high, loadExtraGeometry);
  case Uintah::TypeDescription::float_type:
    return readGridData<VAR, float>(archive, patch, variable_name,
                                    material, timestep, low, high, loadExtraGeometry);
  case Uintah::TypeDescription::int_type:
    return readGridData<VAR, int>(archive, patch, variable_name,
                                  material, timestep, low, high, loadExtraGeometry);
  case Uintah::TypeDescription::Vector:
    return readGridData<VAR, Vector>(archive, patch, variable_name,
                                     material, timestep, low, high, loadExtraGeometry);
  case Uintah::TypeDescription::Stencil7:
    return readGridData<VAR, Stencil7>(archive, patch, variable_name,
                                       material, timestep, low, high, loadExtraGeometry);
  case Uintah::TypeDescription::Stencil4:
    return readGridData<VAR, Stencil4>(archive, patch, variable_name,
                                       material, timestep, low, high, loadExtraGeometry);
  case Uintah::TypeDescription::Matrix3:
    return readGridData<VAR, Matrix3>(archive, patch, variable_name,
                                      material, timestep, low, high, loadExtraGeometry);
  case Uintah::TypeDescription::bool_type:
  case Uintah::TypeDescription::short_int_type:
  case Uintah::TypeDescription::long_type:
  case Uintah::TypeDescription::long64_type:
    std::cerr << "Uintah::archiveUtils::getGridDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is not implemented." << std::endl;
    return nullptr;
  default:
    std::cerr << "Uintah::archiveUtils::getGridDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is unkwown." << std::endl;
    return nullptr;
  }
}


/////////////////////////////////////////////////////////////////////
// Read the grid data for a given patch.
// This function uses the archive for file reading.
template<template<typename> class VAR>
GridDataRaw* getPatchDataMainType(DataArchive *archive,
                                  const Patch *patch,
                                  std::string variable_name,
                                  int material,
                                  int timestep,
                                  const Uintah::TypeDescription *subtype)
{
  switch (subtype->getType())
  {
  case Uintah::TypeDescription::double_type:
    return readPatchData<VAR, double>(archive, patch, variable_name,
                                      material, timestep);
  case Uintah::TypeDescription::float_type:
    return readPatchData<VAR, float>(archive, patch, variable_name,
                                     material, timestep);
  case Uintah::TypeDescription::int_type:
    return readPatchData<VAR, int>(archive, patch, variable_name,
                                   material, timestep);
  case Uintah::TypeDescription::Vector:
  case Uintah::TypeDescription::Stencil7:
  case Uintah::TypeDescription::Stencil4:
  case Uintah::TypeDescription::Matrix3:
  case Uintah::TypeDescription::bool_type:
  case Uintah::TypeDescription::short_int_type:
  case Uintah::TypeDescription::long_type:
  case Uintah::TypeDescription::long64_type:
    std::cerr << "Uintah::archiveUtils::getPatchDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is not implemented." << std::endl;
    return nullptr;
  default:
    std::cerr << "Uintah::archiveUtils::getPatchDataMainType Error: "
              << "Subtype " << subtype->getType() << "  for variable: "
              << subtype->getName() << " is unkwown." << std::endl;
    return nullptr;
  }
}


/////////////////////////////////////////////////////////////////////
// Read the grid data
// This function uses the archive for file reading.
extern "C"
GridDataRaw* getGridData(DataArchive *archive,
                         GridP *grid,
                         int level_i,
                         int patch_i,
                         std::string variable_name,
                         int material,
                         int timestep,
                         int low[3],
                         int high[3],
                         LoadExtraGeometry loadExtraGeometry)
{
  const LevelP &level = (*grid)->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);
  
  // Get variable type from the archive.
  std::vector<std::string>                    vars;
  std::vector<const Uintah::TypeDescription*> types;
  std::vector<int>                            numMatls;
  archive->queryVariables( vars, numMatls, types );

  const Uintah::TypeDescription* maintype = nullptr;
  const Uintah::TypeDescription* subtype = nullptr;

  // Loop through all of the variables in archive.
  for (unsigned int i=0; i<vars.size(); ++i)
  {
    if (vars[i] == variable_name) {
      maintype = types[i];
      subtype = maintype->getSubType();
      break;
    }
  }

  if (!maintype || !subtype)
  {
    std::cerr << "Uintah::archiveUtils::getGridData couldn't find variable type "
              << variable_name << "  "
              << (maintype ? maintype->getName() : " no main type" ) << "  "
              << ( subtype ?  subtype->getName() : " no subtype" ) << "  "
              << std::endl;
    return nullptr;
  }

  switch(maintype->getType())
  {
  case Uintah::TypeDescription::CCVariable:
    return getGridDataMainType<CCVariable>(archive, patch,
                                           variable_name, material, timestep,
                                           low, high, loadExtraGeometry, subtype);
  case Uintah::TypeDescription::NCVariable:
    return getGridDataMainType<NCVariable>(archive, patch,
                                           variable_name, material, timestep,
                                           low, high, loadExtraGeometry, subtype);
  case Uintah::TypeDescription::SFCXVariable:
    return getGridDataMainType<SFCXVariable>(archive, patch,
                                             variable_name, material, timestep,
                                             low, high, loadExtraGeometry, subtype);
  case Uintah::TypeDescription::SFCYVariable:
    return getGridDataMainType<SFCYVariable>(archive, patch,
                                             variable_name, material, timestep,
                                             low, high, loadExtraGeometry, subtype);
  case Uintah::TypeDescription::SFCZVariable:
    return getGridDataMainType<SFCZVariable>(archive, patch,
                                             variable_name, material, timestep,
                                             low, high, loadExtraGeometry, subtype);
  case Uintah::TypeDescription::PerPatch:
    return getPatchDataMainType<PerPatch>(archive, patch,
                                          variable_name, material, timestep, subtype);
  default:
    std::cerr << "Uintah::archiveUtils::getGridData :"
              << "unknown subtype: " << subtype->getName()
              << " for volume variable: " << variable_name << std::endl;
    return nullptr;
  }
}


/////////////////////////////////////////////////////////////////////
// Check to see if a variable exists.
// This function uses the archive for file reading.
extern "C"
bool variableExists(DataArchive *archive,
                    std::string variable_name)
{
  // DOUT(dbgOut, "  variableExists (" << variable_name << ")"  );
  
  // figure out what the type of the variable we're querying is
  std::vector<std::string>                    vars;
  std::vector<const Uintah::TypeDescription*> types;
  std::vector<int>                            numMatls;
  archive->queryVariables( vars, numMatls, types );

  const Uintah::TypeDescription* maintype = nullptr;
  const Uintah::TypeDescription* subtype  = nullptr;

  for (unsigned int i=0; i<vars.size(); ++i) {
    if (vars[i] == variable_name) {
      maintype = types[i];
      subtype = maintype->getSubType();
    }
  }

  return (maintype && subtype);
}


/////////////////////////////////////////////////////////////////////
// Read all the particle data for a given patch.
// This function uses the archive for file reading.
extern "C"
unsigned int getNumberParticles(DataArchive *archive,
                                GridP *grid,
                                int level_i,
                                int patch_i,
                                int material,
                                int timestep)
{
  const LevelP &level = (*grid)->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  const std::string &variable_name =
    Uintah::VarLabel::getParticlePositionName();
  
  if( !archive->exists( variable_name, patch, timestep ) )
    return 0;

  // Figure out which material we're interested in
  ConsecutiveRangeSet allMatls =
    archive->queryMaterials(variable_name, patch, timestep);

  ConsecutiveRangeSet matlsForVar;
  if (material < 0) {
    matlsForVar = allMatls;
  }
  else {
    // Make sure the patch has the variable - use empty material set
    // if it doesn't
    if (allMatls.size() > 0 && allMatls.find(material) != allMatls.end())
      matlsForVar.addInOrder(material);
  }

  // Get all the particle subsets and the total number of particles.
  unsigned int numParticles = 0;

  for( ConsecutiveRangeSet::iterator matlIter =
         matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ )
  {
    int matl = *matlIter;

    ParticleVariable<long64> *var = new ParticleVariable<long64>;
    archive->query(*var, variable_name, matl, patch, timestep);

    numParticles += var->getParticleSubset()->numParticles();
  }

  return numParticles;
}


/////////////////////////////////////////////////////////////////////
// Read all the particle data for a given patch.
// This function uses the archive for file reading.
template<typename T>
ParticleDataRaw* readParticleData(DataArchive *archive,
                                  const Patch *patch,
                                  std::string variable_name,
                                  int material,
                                  int timestep)
{
  if( !archive->exists( variable_name, patch, timestep ) )
    return nullptr;
  
  // Figure out which material we're interested in
  ConsecutiveRangeSet allMatls =
    archive->queryMaterials(variable_name, patch, timestep);

  ConsecutiveRangeSet matlsForVar;
  if (material<0) {
    matlsForVar = allMatls;
  }
  else {
    // Make sure the patch has the variable - use empty material set
    // if it doesn't
    if (allMatls.size()>0 && allMatls.find(material) != allMatls.end())
      matlsForVar.addInOrder(material);
  }

  // Get all the particle subsets and the total number of particles.
  std::vector<ParticleVariable<T>*> particle_vars;

  unsigned int numParticles = 0;
  
  for( ConsecutiveRangeSet::iterator matlIter =
         matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ )
  {
    const int matl = *matlIter;

    ParticleVariable<T> *var = new ParticleVariable<T>;

    archive->query(*var, variable_name, matl, patch, timestep);
    
    particle_vars.push_back(var);
    numParticles += var->getParticleSubset()->numParticles();
  }

  ParticleDataRaw *pd = nullptr;

  // Copy all the data
  if( numParticles ) {
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

  // Cleanup
  for (unsigned int i=0; i<particle_vars.size(); ++i)
    delete particle_vars[i];

  return pd;
}


/////////////////////////////////////////////////////////////////////
// Read the particle data
// This function uses the archive for file reading.
extern "C"
ParticleDataRaw* getParticleData(DataArchive *archive,
                                 GridP *grid,
                                 int level_i,
                                 int patch_i,
                                 std::string variable_name,
                                 int material,
                                 int timestep)
{
  const LevelP &level = (*grid)->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // figure out what the type of the variable we're querying is
  std::vector<std::string>                    vars;
  std::vector<const Uintah::TypeDescription*> types;
  std::vector<int>                            numMatls;
  archive->queryVariables( vars, numMatls, types );

  const Uintah::TypeDescription* maintype = nullptr;
  const Uintah::TypeDescription* subtype  = nullptr;

  for (unsigned int i=0; i<vars.size(); ++i) {
    if (vars[i] == variable_name) {
      maintype = types[i];
      subtype = maintype->getSubType();
    }
  }

  if (!maintype || !subtype) {
    std::cerr << "Uintah::archiveUtils::getGridData couldn't find variable type "
              << variable_name << "  "
              << (maintype ? maintype->getName() : " no main type" ) << "  "
              << ( subtype ?  subtype->getName() : " no subtype" ) << "  "
              << std::endl;
    return nullptr;
  }

  switch (subtype->getType()) {
  case Uintah::TypeDescription::double_type:
    return readParticleData<double>(archive, patch, variable_name,
                                    material, timestep);
  case Uintah::TypeDescription::float_type:
    return readParticleData<float>(archive, patch, variable_name,
                                   material, timestep);
  case Uintah::TypeDescription::int_type:
    return readParticleData<int>(archive, patch, variable_name,
                                 material, timestep);
  case Uintah::TypeDescription::long64_type:
    return readParticleData<long64>(archive, patch, variable_name,
                                    material, timestep);
  case Uintah::TypeDescription::Point:
    return readParticleData<Point>(archive, patch, variable_name,
                                   material, timestep);
  case Uintah::TypeDescription::Vector:
    return readParticleData<Vector>(archive, patch, variable_name,
                                    material, timestep);
  case Uintah::TypeDescription::IntVector:
    return readParticleData<IntVector>(archive, patch, variable_name,
                                       material, timestep);
  case Uintah::TypeDescription::Stencil7:
    return readParticleData<Stencil7>(archive, patch, variable_name,
                                      material, timestep);
  case Uintah::TypeDescription::Stencil4:
    return readParticleData<Stencil4>(archive, patch, variable_name,
                                      material, timestep);
  case Uintah::TypeDescription::Matrix3:
    return readParticleData<Matrix3>(archive, patch, variable_name,
                                     material, timestep);
  default:
    std::cerr << "Uintah::archiveUtils::getGridData :"
              << "unknown subtype: " << subtype->getName()
              << " for particle variable: " << variable_name << std::endl;
    return nullptr;
  }
}

/////////////////////////////////////////////////////////////////////
// Read the particle position name
// This function uses the archive for file reading.
extern "C"
std::string getParticlePositionName(DataArchive *archive)
{
  return archive->getParticlePositionName();
}





#if 0
#include <Core/Grid/AMR.h>
//______________________________________________________________________
//
void allocateTemporary( GridVariableBase& var,
                        const Patch*      patch,
                        Ghost::GhostType  gtype,
                        const int        numGhostCells )
{
  IntVector boundaryLayer(0, 0, 0); // Is this right?
  IntVector lowIndex, highIndex;
  IntVector lowOffset, highOffset;
  Patch::VariableBasis basis = Patch::translateTypeToBasis(var.virtualGetTypeDescription()->getType(), false);
  Patch::getGhostOffsets(var.virtualGetTypeDescription()->getType(), gtype, numGhostCells, lowOffset, highOffset);

  patch->computeExtents(basis, boundaryLayer, lowOffset, highOffset,lowIndex, highIndex);

  var.allocate(lowIndex, highIndex);
}

/*___________________________________________________________________
 Function~  setFineLevelPatchExtraCells-- 
_____________________________________________________________________*/
template<template<typename> class VAR, typename T>
void setFineLevelPatchExtraCells(DataArchive *archive,
                                 const Patch* finePatch,
                                 VAR<T>& Q_fineLevel,          
                                 std::string variable_name,
                                 int material,
                                 int timestep)                        
{
  const Level* fineLevel = finePatch->getLevel();
  const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();

  // cout_dbg << *finePatch << " ";
  // finePatch->printPatchBCs(cout_dbg);

  IntVector refineRatio = fineLevel->getRefinementRatio();
  int order_CFI_Interpolation = 0;
  
  //__________________________________
  // Iterate over coarsefine interface faces
  std::vector<Patch::FaceType> cf;
  finePatch->getCoarseFaces(cf);
  
  std::vector<Patch::FaceType>::const_iterator iter;
  for (iter  = cf.begin(); iter != cf.end(); ++iter){
    Patch::FaceType face = *iter;

    //__________________________________
    // Get fine level hi & lo cell iter limits and coarselevel hi and low index
    IntVector cl, ch, fl, fh;
    getCoarseFineFaceRange(finePatch, coarseLevel, face, Patch::ExtraPlusEdgeCells, 
                           order_CFI_Interpolation, cl, ch, fl, fh);
                           
    //__________________________________
    // enlarge the finelevel foot print by refineRatio (R)
    // x-           x+        y-       y+       z-        z+
    // (-1,0,0)  (1,0,0)  (0,-1,0)  (0,1,0)  (0,0,-1)  (0,0,1)
    IntVector dir = finePatch->getFaceAxes(face);        // face axes
    int pDir      = dir[0];  // principal direction

    if( face == Patch::xminus || face == Patch::yminus || face == Patch::zminus) {
      fl[ pDir ] -= refineRatio[ pDir ] + 1;
    }
    if( face == Patch::xplus  || face == Patch::yplus  || face == Patch::zplus) {
      fh[ pDir ] += refineRatio[ pDir ] - 1;
    } 
    
    // Clamp: don't exceed coarse level limits
    IntVector cL_l, cL_h;
    fineLevel->findCellIndexRange( cL_l, cL_h );
    
    cl = Uintah::Max(cl, cL_l);
    ch = Uintah::Min(ch, cL_h); 
    
    // DOUT(dbgOut, " face " << face << " refineRatio "<< refineRatio
    //     << " BC type " << finePatch->getBCType(face)
    //     << " FineLevel iterator" << fl << " " << fh 
    //     << " \t coarseLevel iterator " << cl << " " << ch << "\n" );

    //__________________________________
    // Pull coarse level data from archive
    VAR<T> Q_CL;
    archive->queryRegion(Q_CL, variable_name, material,
                         coarseLevel, timestep, cl, ch);

    //__________________________________
    // populate fine level cells with coarse level data
    for(CellIterator iter(fl,fh); !iter.done(); iter++){
      IntVector f_cell = *iter;
      IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
      Q_fineLevel[f_cell] = Q_CL[c_cell];
    }

    //____ B U L L E T   P R O O F I N G_______ 
    // All values must be initialized at this point
    // Note only check patches that aren't on the edge of the domain
#if 0
    IntVector badCell;
    CellIterator iter = finePatch->getExtraCellIterator();
    if( isEqual<varType>( varType(d_EVIL_NUM), iter, Q, badCell) ){
      std::ostringstream warn;
      warn <<"ERROR refine_CF_interfaceOperator "
           << "detected an uninitialized variable: "
           << Q_name << ", cell " << badCell
           << " Q_CC " << Q[badCell] 
           << " Patch " << finePatch->getID() << " Level idx "
           <<fineLevel->getIndex()<<"\n\n";
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
#endif

  }  // face loop
}

#endif
