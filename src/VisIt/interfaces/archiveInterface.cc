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
 *  archiveUtils.cc: Provides an interface between the Uintah archive and VisIt.
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
                              LoadExtra loadExtraElements)
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
      LevelP level = (*grid)->getLevel(l);
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

  // Get level information
  for (int l=0; l<numLevels; ++l)
  {
    LevelInfo &levelInfo = stepInfo->levelInfo[l];
    LevelP level = (*grid)->getLevel(l);

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

      // Get the patch id
      patchInfo.setPatchId(patch->getID());
      
      // Get the processor id
      patchInfo.setProcId(archive->queryPatchwiseProcessor(patch, timestep));
    }
  }

  return stepInfo;
}


/////////////////////////////////////////////////////////////////////
// Read the grid data for the given index range
// This function uses the archive for file reading.
template<template <typename> class VAR, typename T>
static GridDataRaw* readGridData(DataArchive *archive,
                                 const Patch *patch,
                                 const LevelP level,
                                 std::string variable_name,
                                 int material,
                                 int timestep,
                                 int low[3],
                                 int high[3],
                                 LoadExtra loadExtraElements)
{
  if( archive->exists( variable_name, patch, timestep ) )
  {
    // printTask( patch, dbgOut, "    readGridData", timestep, material, variable_name );
   
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
      IntVector ilow(low[0], low[1], low[2]);
      IntVector ihigh(high[0], high[1], high[2]);
      
      archive->queryRegion(var, variable_name, material,
                           level.get_rep(), timestep, ilow, ihigh);
    }
    // This queries the entire patch, including extra cells and boundary cells
    else if( loadExtraElements == CELLS )
    {
      archive->query(var, variable_name, material, patch, timestep);
    }
    else if( loadExtraElements == PATCHES )
    {
      // This call does not work properly as it will return garbage
      // where the cells do not exists.
      
      // IntVector ilow(low[0], low[1], low[2]);
      // IntVector ihigh(high[0], high[1], high[2]);
      
      // archive->queryRegion(var, variable_name, material,
      //                      level.get_rep(), timestep, ilow, ihigh);

      // This queries the entire patch, including extra cells and
      // boundary cells which is smaller than the requeste region.
      archive->query(var, variable_name, material, patch, timestep);
    }
  
    T *p = var.getPointer();

    IntVector tmplow;
    IntVector tmphigh;
    IntVector size;
    
    var.getSizes(tmplow, tmphigh, size);
    
    // Fail safe option if the data returned does match the data requested.
    if(  low[0] !=  tmplow[0] ||  low[1] !=  tmplow[1] ||  low[2] !=  tmplow[2] ||
        high[0] != tmphigh[0] || high[1] != tmphigh[1] || high[2] != tmphigh[2] )
    {
      // std::cerr << __LINE__ << "  " << variable_name << "  "
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
// This function uses the archive for file reading.
template<template <typename> class VAR, typename T>
static GridDataRaw* readPatchData(DataArchive *archive,
                                  const Patch *patch,
                                  const LevelP level,
                                  std::string variable_name,
                                  int material,
                                  int timestep)
{
  if( archive->exists( variable_name, patch, timestep ) )
  {
    // printTask( patch, dbgOut, "    readPatchData ", timestep, material, variable_name );
    
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
  else
  {
    return nullptr;
  }
}


/////////////////////////////////////////////////////////////////////
// Read the grid data based on the type
// This function uses the archive for file reading.
template<template<typename> class VAR>
GridDataRaw* getGridDataMainType(DataArchive *archive,
                                 const Patch *patch,
                                 const LevelP level,
                                 std::string variable_name,
                                 int material,
                                 int timestep,
                                 int low[3],
                                 int high[3],
                                 LoadExtra loadExtraElements,
                                 const Uintah::TypeDescription *subtype)
{
  // printTask( patch, dbgOut, "  getGridDataMainType", timestep, material, variable_name );

  switch (subtype->getType()) {
  case Uintah::TypeDescription::double_type:
    return readGridData<VAR, double>(archive, patch, level, variable_name,
                                     material, timestep, low, high, loadExtraElements);
  case Uintah::TypeDescription::float_type:
    return readGridData<VAR, float>(archive, patch, level, variable_name,
                                    material, timestep, low, high, loadExtraElements);
  case Uintah::TypeDescription::int_type:
    return readGridData<VAR, int>(archive, patch, level, variable_name,
                                  material, timestep, low, high, loadExtraElements);
  case Uintah::TypeDescription::Vector:
    return readGridData<VAR, Vector>(archive, patch, level, variable_name,
                                     material, timestep, low, high, loadExtraElements);
  case Uintah::TypeDescription::Stencil7:
    return readGridData<VAR, Stencil7>(archive, patch, level, variable_name,
                                       material, timestep, low, high, loadExtraElements);
  case Uintah::TypeDescription::Stencil4:
    return readGridData<VAR, Stencil4>(archive, patch, level, variable_name,
                                       material, timestep, low, high, loadExtraElements);
  case Uintah::TypeDescription::Matrix3:
    return readGridData<VAR, Matrix3>(archive, patch, level, variable_name,
                                      material, timestep, low, high, loadExtraElements);
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
                                  const LevelP level,
                                  std::string variable_name,
                                  int material,
                                  int timestep,
                                  const Uintah::TypeDescription *subtype)
{

  // printTask( patch, dbgOut, "    getPatchDataMainType ", timestep, material, variable_name );
  
  switch (subtype->getType())
  {
  case Uintah::TypeDescription::double_type:
    return readPatchData<VAR, double>(archive, patch, level, variable_name,
                                      material, timestep);
  case Uintah::TypeDescription::float_type:
    return readPatchData<VAR, float>(archive, patch, level, variable_name,
                                     material, timestep);
  case Uintah::TypeDescription::int_type:
    return readPatchData<VAR, int>(archive, patch, level, variable_name,
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
                         LoadExtra loadExtraElements)
{
  LevelP level = (*grid)->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);
  
  // printTask( patch, dbgOut, "getGridData ", timestep, material, variable_name );

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
    return getGridDataMainType<CCVariable>(archive, patch, level,
                                           variable_name, material, timestep,
                                           low, high, loadExtraElements, subtype);
  case Uintah::TypeDescription::NCVariable:
    return getGridDataMainType<NCVariable>(archive, patch, level,
                                           variable_name, material, timestep,
                                           low, high, loadExtraElements, subtype);
  case Uintah::TypeDescription::SFCXVariable:
    return getGridDataMainType<SFCXVariable>(archive, patch, level,
                                             variable_name, material, timestep,
                                             low, high, loadExtraElements, subtype);
  case Uintah::TypeDescription::SFCYVariable:
    return getGridDataMainType<SFCYVariable>(archive, patch, level,
                                             variable_name, material, timestep,
                                             low, high, loadExtraElements, subtype);
  case Uintah::TypeDescription::SFCZVariable:
    return getGridDataMainType<SFCZVariable>(archive, patch, level,
                                             variable_name, material, timestep,
                                             low, high, loadExtraElements, subtype);
  case Uintah::TypeDescription::PerPatch:
    return getPatchDataMainType<PerPatch>(archive, patch, level,
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
template<typename T>
ParticleDataRaw* readParticleData(DataArchive *archive,
                                  const Patch *patch,
                                  std::string variable_name,
                                  int material,
                                  int timestep)
{

  // printTask( patch, dbgOut, "  readParticleData", timestep, material, variable_name );

  ParticleDataRaw *pd = new ParticleDataRaw;
  pd->components = numComponents<T>();
  pd->num = 0;

  // figure out which material we're interested in
  ConsecutiveRangeSet allMatls =
    archive->queryMaterials(variable_name, patch, timestep);

  ConsecutiveRangeSet matlsForVar;
  if (material<0) {
    matlsForVar = allMatls;
  }
  else {
    // make sure the patch has the variable - use empty material set
    // if it doesn't
    if (allMatls.size()>0 && allMatls.find(material) != allMatls.end())
      matlsForVar.addInOrder(material);
  }

  // first get all the particle subsets so that we know how many total
  // particles we'll have
  std::vector<ParticleVariable<T>*> particle_vars;
  for( ConsecutiveRangeSet::iterator matlIter =
         matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ )
  {
    int matl = *matlIter;

    ParticleVariable<T> *var = new ParticleVariable<T>;
    archive->query(*var, variable_name, matl, patch, timestep);

    particle_vars.push_back(var);
    pd->num += var->getParticleSubset()->numParticles();
  }

  // copy all the data
  int pi = 0;
  pd->data = new double[pd->components * pd->num];
  for (unsigned int i=0; i<particle_vars.size(); ++i)
  {
    ParticleSubset::iterator p;

    for (p = particle_vars[i]->getParticleSubset()->begin();
         p != particle_vars[i]->getParticleSubset()->end(); ++p)
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
  LevelP level = (*grid)->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // printTask( patch, dbgOut, "getParticleData", timestep, material, variable_name );
  
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
void setFineLevelPatchExtraCells(const Patch* finePatch, 
                                 const Level* fineLevel,          
                                 const Level* coarseLevel,        
                                 varType& Q_fineLevel,          
                                 std::string Q_name,           
                                 int matl)                        
{
  cout_dbg << *finePatch << " ";
  finePatch->printPatchBCs(cout_dbg);
  IntVector refineRatio = fineLevel->getRefinementRatio();
  int order_CFI_Interpolation = 0
  
  //__________________________________
  // Iterate over coarsefine interface faces
  std::vector<Patch::FaceType> cf;
  finePatch->getCoarseFaces(cf);
  
  std::vector<Patch::FaceType>::const_iterator iter;
  for (iter  = cf.begin(); iter != cf.end(); ++iter){
    Patch::FaceType face = *iter;

    //__________________________________
    // Get fine level hi & lo cell iter limits
    //  and coarselevel hi and low index


    IntVector cl, ch, fl, fh;
    getCoarseFineFaceRange(finePatch, coarseLevel, face, Patch::ExtraPlusEdgeCells, 
                           order_CFI_Interpolation, cl, ch, fl, fh);
                           
    //__________________________________
    // enlarge the finelevel foot print by refineRatio (R)
    // x-           x+        y-       y+       z-        z+
    // (-1,0,0)  (1,0,0)  (0,-1,0)  (0,1,0)  (0,0,-1)  (0,0,1)
    IntVector dir = finePatch->getFaceAxes(patchFace);        // face axes
    int pDir      = dir[0];  // principal direction

    if( face == Patch::xminus || face == Patch::yminus || face == Patch::zminus) {
      fl[ pDir ] -= refineRatio[ pDir ] + 1;
    }
    if( face == Patch::xplus  || face == Patch::yplus  || face == Patch::zplus) {
      fh[ pDir ] += refineRatio[ pDir ] - 1;
    } 
    
    // clamp: don't exceed fine level limits
    IntVector fL_l, fL_h;
    fineLevel->findCellIndexRange( fL_l, fL_h );
    
    fl = Uintah::Max(fl, fL_l);
    fh = Uintah::Min(fh, fL_h); 
    
    // DOUT(dbgOut, " face " << face << " refineRatio "<< refineRatio
    //     << " BC type " << finePatch->getBCType(face)
    //     << " FineLevel iterator" << fl << " " << fh 
    //     << " \t coarseLevel iterator " << cl << " " << ch << "\n" );

    //__________________________________
    // Pull coarse level data from archive
    VAR<T> Q_CL;
    archive->queryRegion(Q_CL, Q_name, matl, coarseLevel.get_rep(), timestep, cl, ch);

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
