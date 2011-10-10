/*

  The MIT License

  Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
  Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
  University of Utah.

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a 
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation 
  the rights to use, copy, modify, merge, publish, distribute, sublicense, 
  and/or sell copies of the Software, and to permit persons to whom the 
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included 
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
  DEALINGS IN THE SOFTWARE.

*/


/*
 *  uda2nrrd.cc: Provides an interface between VisIt and Uintah.
 *
 *  Written by:
 *   Department of Computer Science
 *   University of Utah
 *   April 2003-2007
 *
 *  Copyright (C) 2003-2007 U of U
 */

#include <StandAlone/tools/uda2vis/udaData.h>

#include <Core/DataArchive/DataArchive.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <algorithm>

using namespace std;
using namespace Uintah;


/////////////////////////////////////////////////////////////////////
// Utility functions for copying data from Uintah structures into
// simple arrays.
void copyIntVector(int to[3], const IntVector &from) {
  to[0]=from[0];  to[1]=from[1];  to[2]=from[2];
}

void copyVector(double to[3], const Vector &from) {
  to[0]=from[0];  to[1]=from[1];  to[2]=from[2];
}

void copyVector(double to[3], const Point &from) {
  to[0]=from.x();  to[1]=from.y();  to[2]=from.z();
}


/////////////////////////////////////////////////////////////////////
// Utility functions for serializing Uintah data structures into
// a simple array for visit.
template <typename T>
int numComponents() {
  return 1;
}

template <>
int numComponents<Vector>() {
  return 3;
}

template <>
int numComponents<Point>() {
  return 3;
}

template <>
int numComponents<Matrix3>() {
  return 9;
}

template <typename T>
void copyComponents(double *dest, const T &src) {
  (*dest) = (double)src;
}

template <>
void copyComponents<Vector>(double *dest, const Vector &src) {
  dest[0] = (double)src[0];
  dest[1] = (double)src[1];
  dest[2] = (double)src[2];
}

template <>
void copyComponents<Point>(double *dest, const Point &src) {
  dest[0] = (double)src.x();
  dest[1] = (double)src.y();
  dest[2] = (double)src.z();
}

template <>
void copyComponents<Matrix3>(double *dest, const Matrix3 &src) {
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      dest[i*3+j] = (double)src(i,j);
    }
  }
}


/////////////////////////////////////////////////////////////////////
// Open a data archive.
extern "C"
DataArchive*
openDataArchive(const string& input_uda_name) {

  DataArchive *archive = scinew DataArchive(input_uda_name);

  return archive;
}


/////////////////////////////////////////////////////////////////////
// Close a data archive - the visit plugin itself doesn't know about
// DataArchive::~DataArchive().
extern "C"
void
closeDataArchive(DataArchive *archive) {
  delete archive;
}


/////////////////////////////////////////////////////////////////////
// Get the grid for the current timestep, so we don't have to query
// it over and over.  We return a pointer to the GridP since the 
// visit plugin doesn't actually know about Grid's (or GridP's), and
// so the handle doesn't get destructed.
extern "C"
GridP*
getGrid(DataArchive *archive, int timeStepNo) {
  GridP *grid = new GridP(archive->queryGrid(timeStepNo));
  return grid;
}


/////////////////////////////////////////////////////////////////////
// Destruct the GridP, which will decrement the reference count.
extern "C"
void
releaseGrid(GridP *grid) {
  delete grid;
}


/////////////////////////////////////////////////////////////////////
// Get the time for each cycle.
extern "C"
vector<double>
getCycleTimes(DataArchive *archive) {

  // Get the times and indices.
  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  archive->queryTimesteps(index, times);

  return times;
} 

  
/////////////////////////////////////////////////////////////////////
// Get all the information that may be needed for the current timestep,
// including variable/material info, and level/patch info
extern "C"
TimeStepInfo*
getTimeStepInfo(DataArchive *archive, GridP *grid, int timestep, bool useExtraCells) {
  int numLevels = (*grid)->numLevels();
  TimeStepInfo *stepInfo = new TimeStepInfo;
  stepInfo->levelInfo.resize(numLevels);

  // get variable information
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  archive->queryVariables(vars, types);
  stepInfo->varInfo.resize(vars.size());

  for (unsigned int i=0; i<vars.size(); i++) {
    VariableInfo &varInfo = stepInfo->varInfo[i];

    varInfo.name = vars[i];
    varInfo.type = types[i]->getName();

    // query each level for material info until we find something
    for (int l=0; l<numLevels; l++) {
      LevelP level = (*grid)->getLevel(l);
      const Patch* patch = *(level->patchesBegin());
      ConsecutiveRangeSet matls = archive->queryMaterials(vars[i], patch, timestep);
      if (matls.size() > 0) {

        // copy the list of materials
        for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
             matlIter != matls.end(); matlIter++)
          varInfo.materials.push_back(*matlIter);

        // don't query on any more levels
        break;
      }
    }
  }


  // get level information
  for (int l=0; l<numLevels; l++) {
    LevelInfo &levelInfo = stepInfo->levelInfo[l];
    LevelP level = (*grid)->getLevel(l);

    copyIntVector(levelInfo.refinementRatio, level->getRefinementRatio());
    copyIntVector(levelInfo.extraCells, level->getExtraCells());
    copyVector(levelInfo.spacing, level->dCell());
    copyVector(levelInfo.anchor, level->getAnchor());

    // patch info
    int numPatches = level->numPatches();
    levelInfo.patchInfo.resize(numPatches);

    for (int p=0; p<numPatches; p++) {
      const Patch* patch = level->getPatch(p);
      PatchInfo &patchInfo = levelInfo.patchInfo[p];

      // If the user wants to see the extra cells, just include them and let VisIt believe they are part
      // of the original data. This is accomplished by simply setting cc_low and cc_high to the extra
      // cell boundaries so that VisIt is none the wiser.
      if (useExtraCells)
      {
        copyIntVector(patchInfo.cc_low,  patch->getExtraCellLowIndex());  
        copyIntVector(patchInfo.cc_high, patch->getExtraCellHighIndex());
      }
      else
      {
        copyIntVector(patchInfo.cc_low, patch->getCellLowIndex());
        copyIntVector(patchInfo.cc_high, patch->getCellHighIndex());
      }
      copyIntVector(patchInfo.cc_extra_low,  patch->getExtraCellLowIndex());
      copyIntVector(patchInfo.cc_extra_high, patch->getExtraCellHighIndex());

      // nc indices
      if (useExtraCells)
      {
        copyIntVector(patchInfo.nc_low,  patch->getExtraNodeLowIndex());  
        copyIntVector(patchInfo.nc_high, patch->getExtraNodeHighIndex());
      }
      else 
      {
        copyIntVector(patchInfo.nc_low, patch->getNodeLowIndex());
        copyIntVector(patchInfo.nc_high, patch->getNodeHighIndex());
      }
      copyIntVector(patchInfo.nc_extra_low,  patch->getExtraNodeLowIndex());
      copyIntVector(patchInfo.nc_extra_high, patch->getExtraNodeHighIndex());

      patchInfo.proc_id = archive->queryPatchwiseProcessor(patch, timestep);
    }
  }


  return stepInfo;
}


/////////////////////////////////////////////////////////////////////
// Read the grid data for the given index range
template<template <typename> class VAR, typename T>
static GridDataRaw* readGridData(DataArchive *archive,
                                 const Patch *patch,
                                 const LevelP level,
                                 string variable_name,
                                 int material,
                                 int timestep,
                                 int low[3],
                                 int high[3]) {

  IntVector ilow(low[0], low[1], low[2]);
  IntVector ihigh(high[0], high[1], high[2]);

  // this queries the entire patch, including extra cells and boundary cells
  VAR<T> var;
  archive->queryRegion(var, variable_name, material, level.get_rep(), timestep, ilow, ihigh);

  //  IntVector low = var.getLowIndex();
  //  IntVector high = var.getHighIndex();

  GridDataRaw *gd = new GridDataRaw;
  gd->components = numComponents<T>();
  for (int i=0; i<3; i++) {
    gd->low[i] = low[i];
    gd->high[i] = high[i];
  }

  int n = (high[0]-low[0])*(high[1]-low[1])*(high[2]-low[2]);
  gd->data = new double[n*gd->components];

  T *p=var.getPointer();
  for (int i=0; i<n; i++)
    copyComponents<T>(&gd->data[i*gd->components], p[i]);
  
  return gd;
}


template<template<typename> class VAR>
GridDataRaw* getGridDataMainType(DataArchive *archive,
                                 const Patch *patch,
                                 const LevelP level,
                                 string variable_name,
                                 int material,
                                 int timestep,
                                 int low[3],
                                 int high[3],
                                 const Uintah::TypeDescription *subtype) {

  switch (subtype->getType()) {
  case Uintah::TypeDescription::double_type:
    return readGridData<VAR, double>(archive, patch, level, variable_name, material, timestep, low, high);
  case Uintah::TypeDescription::float_type:
    return readGridData<VAR, float>(archive, patch, level, variable_name, material, timestep, low, high);
  case Uintah::TypeDescription::int_type:
    return readGridData<VAR, int>(archive, patch, level, variable_name, material, timestep, low, high);
  case Uintah::TypeDescription::Vector:
    return readGridData<VAR, Vector>(archive, patch, level, variable_name, material, timestep, low, high);
  case Uintah::TypeDescription::Matrix3:
    return readGridData<VAR, Matrix3>(archive, patch, level, variable_name, material, timestep, low, high);
  case Uintah::TypeDescription::bool_type:
  case Uintah::TypeDescription::short_int_type:
  case Uintah::TypeDescription::long_type:
  case Uintah::TypeDescription::long64_type:
    cerr << "Subtype " << subtype->getName() << " is not implemented...\n";
    return NULL;
  default:
    cerr << "Unknown subtype\n";
    return NULL;
  }
}


extern "C"
GridDataRaw*
getGridData(DataArchive *archive,
            GridP *grid,
            int level_i,
            int patch_i,
            string variable_name,
            int material,
            int timestep,
            int low[3],
            int high[3]) {

  LevelP level = (*grid)->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // figure out what the type of the variable we're querying is
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  archive->queryVariables(vars, types);

  const Uintah::TypeDescription* maintype = NULL;
  const Uintah::TypeDescription* subtype = NULL;

  for (unsigned int i=0; i<vars.size(); i++) {
    if (vars[i] == variable_name) {
      maintype = types[i];
      subtype = maintype->getSubType();
    }
  }

  if (!maintype || !subtype) {
    cerr<<"couldn't find variable " << variable_name<<endl;
    return NULL;
  }


  switch(maintype->getType()) {
  case Uintah::TypeDescription::CCVariable:
    return getGridDataMainType<CCVariable>(archive, patch, level, variable_name, material, timestep, low, high, subtype);
  case Uintah::TypeDescription::NCVariable:
    return getGridDataMainType<NCVariable>(archive, patch, level, variable_name, material, timestep, low, high, subtype);
  case Uintah::TypeDescription::SFCXVariable:
    return getGridDataMainType<SFCXVariable>(archive, patch, level, variable_name, material, timestep, low, high, subtype);
  case Uintah::TypeDescription::SFCYVariable:
    return getGridDataMainType<SFCYVariable>(archive, patch, level, variable_name, material, timestep, low, high, subtype);
  case Uintah::TypeDescription::SFCZVariable:
    return getGridDataMainType<SFCZVariable>(archive, patch, level, variable_name, material, timestep, low, high, subtype);
  default:
    cerr << "Type is unknown.\n";
    return NULL;
  }
}




/////////////////////////////////////////////////////////////////////
// Read all the particle data for a given patch.
template<typename T>
ParticleDataRaw* readParticleData(DataArchive *archive,
                                  const Patch *patch,
                                  string variable_name,
                                  int material,
                                  int timestep) {

  ParticleDataRaw *pd = new ParticleDataRaw;
  pd->components = numComponents<T>();
  pd->num = 0;

  // figure out which material we're interested in
  ConsecutiveRangeSet allMatls = archive->queryMaterials(variable_name, patch, timestep);

  ConsecutiveRangeSet matlsForVar;
  if (material<0) {
    matlsForVar = allMatls;
  }
  else {
    // make sure the patch has the variable - use empty material set if it doesn't
    if (allMatls.size()>0 && allMatls.find(material) != allMatls.end())
      matlsForVar.addInOrder(material);
  }

  // first get all the particle subsets so that we know how many total particles we'll have
  vector<ParticleVariable<T>*> particle_vars;
  for( ConsecutiveRangeSet::iterator matlIter = matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ ) {
    int matl = *matlIter;

    ParticleVariable<T> *var = new ParticleVariable<T>;
    archive->query(*var, variable_name, matl, patch, timestep);

    particle_vars.push_back(var);
    pd->num += var->getParticleSubset()->numParticles();
  }


  // copy all the data
  int pi=0;
  pd->data = new double[pd->components * pd->num];
  for (unsigned int i=0; i<particle_vars.size(); i++) {
    for (ParticleSubset::iterator p = particle_vars[i]->getParticleSubset()->begin();
         p != particle_vars[i]->getParticleSubset()->end(); ++p) {

      copyComponents<T>(&pd->data[pi*pd->components],
                        (*particle_vars[i])[*p]);
      pi++;
    }
  }

  // cleanup
  for (unsigned int i=0; i<particle_vars.size(); i++)
    delete particle_vars[i];

  return pd;
}



extern "C"
ParticleDataRaw*
getParticleData(DataArchive *archive,
                GridP *grid,
                int level_i,
                int patch_i,
                string variable_name,
                int material,
                int timestep) {

  LevelP level = (*grid)->getLevel(level_i);
  const Patch *patch = level->getPatch(patch_i);

  // figure out what the type of the variable we're querying is
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  archive->queryVariables(vars, types);

  const Uintah::TypeDescription* maintype = NULL;
  const Uintah::TypeDescription* subtype = NULL;

  for (unsigned int i=0; i<vars.size(); i++) {
    if (vars[i] == variable_name) {
      maintype = types[i];
      subtype = maintype->getSubType();
    }
  }

  if (!maintype || !subtype) {
    cerr<<"couldn't find variable " << variable_name<<endl;
    return NULL;
  }


  switch (subtype->getType()) {
  case Uintah::TypeDescription::double_type:
    return readParticleData<double>(archive, patch, variable_name, material, timestep);
  case Uintah::TypeDescription::float_type:
    return readParticleData<float>(archive, patch, variable_name, material, timestep);
  case Uintah::TypeDescription::int_type:
    return readParticleData<int>(archive, patch, variable_name, material, timestep);
  case Uintah::TypeDescription::long64_type:
    return readParticleData<long64>(archive, patch, variable_name, material, timestep);
  case Uintah::TypeDescription::Point:
    return readParticleData<Point>(archive, patch, variable_name, material, timestep);
  case Uintah::TypeDescription::Vector:
    return readParticleData<Vector>(archive, patch, variable_name, material, timestep);
  case Uintah::TypeDescription::Matrix3:
    return readParticleData<Matrix3>(archive, patch, variable_name, material, timestep);
  default:
    cerr << "Unknown subtype for particle data: " << subtype->getName() << "\n";
    return NULL;
  }
}

