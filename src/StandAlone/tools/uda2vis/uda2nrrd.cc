/*

  The MIT License

  Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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
 *  uda2nrrd.cc: Converts a Uintah Data Archive (UDA) to a nrrd.
 *
 *  Written by:
 *   Many people...?
 *   Department of Computer Science
 *   University of Utah
 *   April 2003-2007
 *
 *  Copyright (C) 2003-2007 U of U
 */

#include <StandAlone/tools/uda2vis/particleData.h>

#include <StandAlone/tools/uda2vis/wrap_nrrd.h>

#include <StandAlone/tools/uda2vis/Args.h>
#include <StandAlone/tools/uda2vis/bc.h>
#include <StandAlone/tools/uda2vis/handleVariable.h>
#include <StandAlone/tools/uda2vis/particles.h>
#include <StandAlone/tools/uda2vis/QueryInfo.h>

#include <Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Math/MinMax.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>

#include <Core/OS/Dir.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Persistent/Pstreams.h>


#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/DataArchive/DataArchive.h>

#include <sci_hash_map.h>
#include <teem/nrrd.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;


void
  usage( const string& badarg, const string& progname )
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << "\n";
  cerr << "Usage: " << progname << " [options] "
       << "-uda <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h,--help  Prints this message out\n";

  cerr << "\nField Specifier Options\n";
  cerr << "  -v,--variable <variable name> - may not be used with -p\n";
  cerr << "  -p,--particledata - Pull out all the particle data into a single NRRD.  May not be used with -v\n";
  cerr << "  -m,--material <material number> [defaults to first material found]\n";
  cerr << "  -l,--level <level index> [defaults to 0]\n";
  cerr << "  -a,--all - Use all levels.  Overrides -l.  Uses the resolution\n";
  cerr << "             of the finest level. Fills the entire domain by \n";
  cerr << "             interpolating data from lower resolution levels\n";
  cerr << "             when necessary.  May not be used with -p.\n";
  cerr << "  -mo <operator> type of operator to apply to matricies.\n";
  cerr << "                 Options are none, det, norm, and trace\n";
  cerr << "                 [defaults to none]\n";
  cerr << "  -nbc,--noboundarycells - remove boundary cells from output\n";

  cerr << "\nOutput Options\n";
  cerr << "  -o,--out <outputfilename> [defaults to data]\n";
  cerr << "  -oi <index> [default to 0] - Output index to use in naming file.\n";
  cerr << "  -dh,--detatched-header - writes the data with detached headers.  The default is to not do this.\n";
  //    cerr << "  -binary (prints out the data in binary)\n";

  cerr << "\nTimestep Specifier Optoins\n";
  cerr << "  -tlow,--timesteplow [int] (only outputs timestep from int) [defaults to 0]\n";
  cerr << "  -thigh,--timestephigh [int] (only outputs timesteps up to int) [defaults to last timestep]\n";
  cerr << "  -tinc [int] (output every n timesteps) [defaults to 1]\n";
  cerr << "  -tstep,--timestep [int] (only outputs timestep int)\n";

  cerr << "\nChatty Options\n";
  cerr << "  -vv,--verbose (prints status of output)\n";
  cerr << "  -q,--quiet (very little output)\n";
  exit(1);
}


/////////////////////////////////////////////////////////////////////
extern "C"
DataArchive*
openDataArchive(const string& input_uda_name) {

  DataArchive *archive = scinew DataArchive(input_uda_name);

  return archive;
}

/////////////////////////////////////////////////////////////////////
extern "C"
void
closeDataArchive(DataArchive *archive) {
  delete archive;
}

/////////////////////////////////////////////////////////////////////
extern "C"
GridP*
getGrid(DataArchive *archive, int timeStepNo) {
  GridP *grid = new GridP(archive->queryGrid(timeStepNo));
  return grid;
}

/////////////////////////////////////////////////////////////////////
extern "C"
void
releaseGrid(GridP *grid) {
  delete grid;
}

/////////////////////////////////////////////////////////////////////
extern "C"
int*
getPeriodicBoundaries(DataArchive *archive, GridP *grid, int levelNo) {

  int* boundaryExists = new int[3];

  int numLevels = (*grid)->numLevels();

  LevelP level;

  if (levelNo < numLevels) {
    level = (*grid)->getLevel(levelNo);
    IntVector a = level->getPeriodicBoundaries();

    boundaryExists[0] = a.x();   
    boundaryExists[1] = a.y();   
    boundaryExists[2] = a.z();   
  }

  return boundaryExists;
}

/////////////////////////////////////////////////////////////////////
extern "C"
int*
getExtraCells(DataArchive *archive, GridP *grid, int levelNo) {

  int* extraCells = new int[3];

  int numLevels = (*grid)->numLevels();

  LevelP level;

  if (levelNo < numLevels) {
    level = (*grid)->getLevel(levelNo);
    IntVector a = level->getExtraCells();

    extraCells[0] = a.x();   
    extraCells[1] = a.y();   
    extraCells[2] = a.z();   
  }

  return extraCells;
}

/////////////////////////////////////////////////////////////////////
extern "C"
levelPatchVec*
getTotalNumPatches(DataArchive *archive, GridP *grid) {

  levelPatchVec* levelPatchVecPtr = new levelPatchVec();

  int numLevels = (*grid)->numLevels();

  LevelP level;
  for (int i = 0; i < numLevels; i++) {
    level = (*grid)->getLevel(i);
    IntVector rr = level->getRefinementRatio();
    // cout << "Refinement ratio, Level " << i << ": " << level->getRefinementRatio() << endl;
    levelPatch levelPatchObj(i, level->numPatches(), rr.x(), rr.y(), rr.z());
    levelPatchVecPtr->push_back(levelPatchObj);
    // *numPatches += level->numPatches();
  }	

  return levelPatchVecPtr;
}

/////////////////////////////////////////////////////////////////////
extern "C"
varMatls*
getMaterials(DataArchive *archive, GridP *grid, int timeStepNo, const string& variable_name) {

  varMatls* varMatlList = new varMatls();

  int numLevels = (*grid)->numLevels();

  LevelP level;
  for (int i = 0; i < numLevels; i++) {
    level = (*grid)->getLevel(i);
    const Patch* patch = *(level->patchesBegin());
    ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, timeStepNo);
    if (matls.size() > 0) { // Found particles, volume data should also be present there
      for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
           matlIter != matls.end(); matlIter++) {
        varMatlList->push_back(*matlIter);
      }
      break;
    }
  }

  return varMatlList;
}

/////////////////////////////////////////////////////////////////////
extern "C"
double*
getBBox(DataArchive *archive, GridP* grid, int levelNo) {

  LevelP level;
  BBox box;

  level = (*grid)->getLevel(levelNo);
  level->getSpatialRange(box);

  Point min = box.min();
  Point max = box.max();

  double *minMaxArr = new double[6];

  minMaxArr[0] = min.x(); minMaxArr[1] = min.y(); minMaxArr[2] = min.z();
  minMaxArr[3] = max.x(); minMaxArr[4] = max.y(); minMaxArr[5] = max.z();

  return minMaxArr;
} 


/////////////////////////////////////////////////////////////////////
extern "C"
patchInfoVec*
getPatchInfo(DataArchive *archive, GridP *grid, const string& varType, 
             bool remove_boundary) {

  int numLevels = (*grid)->numLevels();

  patchInfoVec* patchInfoVecPtr = new patchInfoVec();

  IntVector patch_lo, patch_hi, low, hi;
  Point min, max;
  int *hiLoArr = new int[6];
  int *indexArr = new int[6];
  double *minMaxArr = new double[6];

  LevelP level;
  for (int i = 0; i < numLevels; i++) {
    level = (*grid)->getLevel(i);

    if(remove_boundary) {
      level->findInteriorIndexRange(low, hi);
    } 
    else {
      level->findIndexRange(low, hi);
    }

    int numPatches = level->numPatches();
    for (int j = 0; j < numPatches; j++) {
      const Patch* patch = level->getPatch(j);

      if (remove_boundary) { // this needs to be kept outside the loop, same check again and again
        if(varType.find("CC") != string::npos) {
          patch_lo = patch->getCellLowIndex();
          patch_hi = patch->getCellHighIndex();
        } 
        else {
          patch_lo = patch->getNodeLowIndex();
          if(varType.find("SFCX") != string::npos) {
            patch_hi = patch_lo + (patch->getCellHighIndex() - patch->getCellLowIndex());
            if (patch_hi.x() == (hi.x() - 1)) {
              patch_hi = IntVector(patch_hi.x() + 1, patch_hi.y(), patch_hi.z());
            }
          }		
          else if(varType.find("SFCY") != string::npos)
            patch_hi = patch->getHighIndex(Patch::YFaceBased);
          else if(varType.find("SFCZ") != string::npos)
            patch_hi = patch->getHighIndex(Patch::ZFaceBased);
          else if(varType.find("NC") != string::npos) {
            patch_hi = patch_lo + (patch->getCellHighIndex() - patch->getCellLowIndex()) + IntVector(1, 1, 1);
          }
        }
      }
      else { // don't remove the boundary
        if(varType.find("CC") != string::npos) {
          patch_lo = patch->getExtraCellLowIndex();
          patch_hi = patch->getExtraCellHighIndex();
        } 
        else {
          patch_lo = patch->getExtraNodeLowIndex();
          if(varType.find("SFCX") != string::npos)
            patch_hi = patch->getSFCXHighIndex();
          else if(varType.find("SFCY") != string::npos)
            patch_hi = patch->getSFCYHighIndex();
          else if(varType.find("SFCZ") != string::npos)
            patch_hi = patch->getSFCZHighIndex();
          else if(varType.find("NC") != string::npos)
            patch_hi = patch->getExtraNodeHighIndex();
        }
      }

      indexArr[0] = patch_lo.x(); 
      indexArr[1] = patch_lo.y(); 
      indexArr[2] = patch_lo.z();
      indexArr[3] = patch_hi.x(); 
      indexArr[4] = patch_hi.y(); 
      indexArr[5] = patch_hi.z();  

      hiLoArr[0] = low.x();
      hiLoArr[1] = low.y();
      hiLoArr[2] = low.z();
      hiLoArr[3] = hi.x();
      hiLoArr[4] = hi.y();
      hiLoArr[5] = hi.z();

      if(remove_boundary) {
        min = patch->getBox().lower();
        max = patch->getBox().upper();
      }
      else {
        min = patch->getExtraBox().lower();
        max = patch->getExtraBox().upper();
      }

      minMaxArr[0] = min.x(); minMaxArr[1] = min.y(); minMaxArr[2] = min.z();
      minMaxArr[3] = max.x(); minMaxArr[4] = max.y(); minMaxArr[5] = max.z();

      int nCells = (patch->getCellHighIndex() - patch->getCellLowIndex()).y();

      patchInfo patchInfoObj(indexArr, minMaxArr, hiLoArr, nCells);
      patchInfoVecPtr->push_back(patchInfoObj);
    }
  }		

  return patchInfoVecPtr;
}


/////////////////////////////////////////////////////////////////////
extern "C"
typeDouble*
getTimeSteps(DataArchive *archive) {

  typeDouble* timeStepInfo = new typeDouble();

  // Get the times and indices.
  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  archive->queryTimesteps(index, times);

  int noIndex = index.size();
  timeStepInfo->reserve(noIndex);

  // (vector<double>)(*timeStepInfo) = times;  
  for (int i = 0; i < noIndex; i++) {
    timeStepInfo->push_back(times[i]);
  }

  return timeStepInfo;
} 


/////////////////////////////////////////////////////////////////////
extern "C"
udaVars*
getVarList(DataArchive *archive) {
  udaVars* udaVarList = new udaVars();

  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;

  archive->queryVariables(vars, types);

  for (unsigned int i = 0; i < vars.size(); i++) {
    string nameType = vars[i] + "/" + types[i]->getName(); 
    udaVarList->push_back(nameType);
  }

  return udaVarList;
}
  
  
/////////////////////////////////////////////////////////////////////
extern "C"
int*
getPVarLevelAndPatches(DataArchive *archive,
                       GridP *grid,
                       int timeStepNo,
                       const string& varName) {

  int* levelAndPatches = new int(2);

  bool found_particle_level = false;
  for( int lev = 0; lev < (*grid)->numLevels(); lev++ ) {
    LevelP particleLevel = (*grid)->getLevel( lev );
    const Patch* patch = *(particleLevel->patchesBegin());
    ConsecutiveRangeSet matls = archive->queryMaterials(varName, patch, timeStepNo);
    if( matls.size() > 0 ) {
      if( found_particle_level ) {
        // Ut oh... found particles on more than one level... don't know how 
        // to handle this yet...
        cout << "\n";
        cout << "Error: uda2nrrd currently can only handle particles on only a single level.  Goodbye.\n";
        cout << "\n";
        exit(1);
      }
      // The particles are on this level...
      found_particle_level = true;
      levelAndPatches[0] = lev;
      levelAndPatches[1] = particleLevel->numPatches();
      // cout << "Found the PARTICLES on level " << lev << ".\n";
    }
  }
      
  return levelAndPatches;
} 
  
 
/////////////////////////////////////////////////////////////////////
extern "C"
timeStep*
processData(DataArchive *archive, GridP *grid,
            int timeStepNo, 
            int level_index,
            int patchNo,
            string variable_name,
            int material,
            bool do_particles,
            bool dataReq)
{
  /*
   * Default values
   */

  Args args;
  args.quiet=true;

  try {

    ////////////////////////////////////////////////////////
    // Get the times and indices.

    vector<int> index;
    vector<double> times;

    // query time info from dataarchive
    archive->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    if( !args.quiet ) cout << "There are " << index.size() << " timesteps:\n";


    //////////////////////////////////////////////////////////
    // Get the variables and types
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    if( args.verbose ) cout << "There are " << vars.size() << " variables:\n";
    bool var_found = false;
    int var_index=-1;

    if( do_particles ) {
      unsigned int vi = 0;
      for( ; vi < vars.size(); vi++ ) {
        // if( vars[vi][0] == 'p' && vars[vi][1] == '.' ) { // starts with "p."
        if ( types[vi]->getType() == Uintah::TypeDescription::ParticleVariable ) { 
          // It is a particle variable
          if (vars[vi]==variable_name) {
            var_index = vi;
            break;
          }
        }
      }
      if( var_index < 0 ) {
        cout << "\n";
        cout << "Error: No particle variables found (\"p.something\")...\n";
        cout << "\n";
        cout << "Variables known are:\n";
        vi = 0;
        for( ; vi < vars.size(); vi++) {
          cout << "vars[" << vi << "] = " << vars[vi] << "\n";
        }
        cout << "\nGoodbye!!\n\n";
        exit(-1);
      }
    } 
    else { // Not particles...
      unsigned int vi = 0;
      for( ; vi < vars.size(); vi++ ) {
        if( variable_name == vars[vi] ) {
          var_found = true;
          break;
        }
      }
      if (!var_found) {
        cerr << "Variable \"" << variable_name << "\" was not found.\n";
        cerr << "If a variable name was not specified try -v [name].\n";
        cerr << "Possible variable names are:\n";
        vi = 0;
        for( ; vi < vars.size(); vi++) {
          cout << "vars[" << vi << "] = " << vars[vi] << "\n";
        }
        cerr << "\nExiting!!\n\n";
        exit(-1);
      }
      var_index = vi;
    }


    /////////////////////////////
    // Figure out the filename

    char filename_num[200];
    sprintf( filename_num, "_t%06d", index[timeStepNo] );


    // Check the level index
    if (level_index >= (*grid)->numLevels() || level_index < 0) {
      cerr << "level index is bad ("<<level_index<<").  Should be between 0 and "<<(*grid)->numLevels()<<".\n";
      exit(1); 
    }

    // Create a timeStep object, corresponding to every time step.
    timeStep* timeStepObjPtr = new timeStep();

    // Storing the time step name/ no.
    timeStepObjPtr->name.assign(filename_num + 1);
    timeStepObjPtr->no = index[timeStepNo];
  
    variable_name = vars[var_index];

    //////////////////////////////////////////////////
    // Set the level pointer

    LevelP level;

    // the data will be on the same level(s) for all timesteps.
    if( do_particles ) { // Determine which level the particles are on...
      bool found_particle_level = false;
      for( int lev = 0; lev < (*grid)->numLevels(); lev++ ) {
        LevelP particleLevel = (*grid)->getLevel( lev );
        const Patch* patch = *(particleLevel->patchesBegin());
        ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, timeStepNo);
        if( matls.size() > 0 ) {
          if( found_particle_level ) {
            // Ut oh... found particles on more than one level... don't know how 
            // to handle this yet...
            cout << "\n";
            cout << "Error: uda2nrrd currently can only handle particles on only a single level.  Goodbye.\n";
            cout << "\n";
            exit(1);
          }
          // The particles are on this level...
          found_particle_level = true;
          level = particleLevel;
          // cout << "Found the PARTICLES on level " << lev << ".\n";
        }
      }
    }
    else {
      if( args.use_all_levels ){ // set to level zero
        level = (*grid)->getLevel( 0 );
        if( (*grid)->numLevels() == 1 ){ // only one level to use
          args.use_all_levels = false;
        }
      } else {  // set to requested level
        level = (*grid)->getLevel(level_index);
      }
    }

    ///////////////////////////////////////////////////
    // Check the material number.

    const Patch* patch = *(level->patchesBegin());
      
    ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, timeStepNo);

    ConsecutiveRangeSet  materialsOfInterest;

    if( do_particles ) {
      materialsOfInterest = matls;
    } else {
      if (material == -1) {
        materialsOfInterest.addInOrder( *(matls.begin()) ); // Default: only interested in first material.
      } else {
        unsigned int mat_index = 0;

        ConsecutiveRangeSet::iterator matlIter = matls.begin();

        for( ; matlIter != matls.end(); matlIter++ ){
          int matl = *matlIter;
          if (matl == material) {
            materialsOfInterest.addInOrder( matl );
            break;
          }
          mat_index++;
        }
        if( mat_index == matls.size() ) { // We didn't find the right material...
          cerr << "Didn't find material " << material << " in the data.\n";
          cerr << "Trying next timestep.\n";
          exit(-1);
        }
      }
    }

    // get type and subtype of data
    const Uintah::TypeDescription* td = types[var_index];
    const Uintah::TypeDescription* subtype = td->getSubType();

    QueryInfo qinfo( archive, (*grid), level, variable_name, materialsOfInterest,
                     timeStepNo, args.use_all_levels, td );

    IntVector hi, low, range;
    BBox box;

    // Remove the edges if no boundary cells
    if( args.remove_boundary ){
      level->findInteriorIndexRange(low, hi);
      level->getInteriorSpatialRange(box);
    } else {

      const Patch* patch = level->getPatch(patchNo);

      Point min = patch->getExtraBox().lower();
      Point max = patch->getExtraBox().upper();

      IntVector extraCells = patch->getExtraCells();
  
      // necessary check - useful with periodic boundaries
      for (int i = 0; i < 3; i++) {
        if (extraCells(i) == 0) {
          extraCells(i) = 1;
        }
      }

      IntVector noCells = patch->getCellHighIndex() - patch->getCellLowIndex();

      box = BBox(min, max);

      low = patch->getNodeLowIndex() - extraCells;
      hi = patch->getNodeLowIndex() + noCells + extraCells + IntVector(1, 1, 1);
    }

    // this is a hack to make things work, substantiated in build_multi_level_field()
    range = hi - low;

    // Adjust the range for using all levels
    if( args.use_all_levels && (*grid)->numLevels() > 0 ){
      double exponent = (*grid)->numLevels() - 1;
      range.x( range.x() * int(pow(2.0, exponent)));
      range.y( range.y() * int(pow(2.0, exponent)));
      range.z( range.z() * int(pow(2.0, exponent)));
      low.x( low.x() * int(pow(2.0, exponent)));
      low.y( low.y() * int(pow(2.0, exponent)));
      low.z( low.z() * int(pow(2.0, exponent)));
      hi.x( hi.x() * int(pow(2.0, exponent)));
      hi.y( hi.y() * int(pow(2.0, exponent)));
      hi.z( hi.z() * int(pow(2.0, exponent)));

      if( args.verbose ){
        cout<< "The entire domain for all levels will have an index range of "
            << low <<" to "<< hi
            << " and a spatial range from "<< box.min()<< " to "
            << box.max()<<".\n";
      }
    }


    ///////////////////
    // Get the data...
    if( td->getType() == Uintah::TypeDescription::ParticleVariable ) {  // Handle Particles

      //ParticleDataContainer data;

      switch (subtype->getType()) {
      case Uintah::TypeDescription::double_type:
        handleParticleData<double>( qinfo, material, timeStepObjPtr->partVar, variable_name, patchNo );
        break;
      case Uintah::TypeDescription::float_type:
        handleParticleData<float>( qinfo, material, timeStepObjPtr->partVar, variable_name, patchNo );
        break;
      case Uintah::TypeDescription::int_type:
        handleParticleData<int>( qinfo, material, timeStepObjPtr->partVar, variable_name, patchNo );
        break;
      case Uintah::TypeDescription::long64_type:
        handleParticleData<long64>( qinfo, material, timeStepObjPtr->partVar, variable_name, patchNo );
        break;
      case Uintah::TypeDescription::Point:
        handleParticleData<Point>( qinfo, material, timeStepObjPtr->partVar, variable_name, patchNo );
        break;
      case Uintah::TypeDescription::Vector:
        handleParticleData<Vector>( qinfo, material, timeStepObjPtr->partVar, variable_name, patchNo );
        break;
      case Uintah::TypeDescription::Matrix3:
        handleParticleData<Matrix3>( qinfo, material, timeStepObjPtr->partVar, variable_name, patchNo );
        break;
      default:
        cerr << "Unknown subtype for particle data: " << subtype->getName() << "\n";
        exit(1);
      } // end switch( subtype )

        //particleDataArray.push_back( data );

    } else { // Handle Grid Variables

      switch (subtype->getType()) {
      case Uintah::TypeDescription::double_type:
        timeStepObjPtr->cellValColln = new cellVals();
        handleVariable<double>( qinfo, low, hi, range, box, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
        break;
      case Uintah::TypeDescription::float_type:
        timeStepObjPtr->cellValColln = new cellVals();
        handleVariable<float>( qinfo, low, hi, range, box, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
        break;
      case Uintah::TypeDescription::int_type:
        timeStepObjPtr->cellValColln = new cellVals();
        handleVariable<int>( qinfo, low, hi, range, box, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
        break;
      case Uintah::TypeDescription::Vector:
        timeStepObjPtr->cellValColln = new cellVals();
        handleVariable<Vector>( qinfo, low, hi, range, box, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
        break;
      case Uintah::TypeDescription::Matrix3:
        timeStepObjPtr->cellValColln = new cellVals();
        handleVariable<Matrix3>( qinfo, low, hi, range, box, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
        break;
      case Uintah::TypeDescription::bool_type:
      case Uintah::TypeDescription::short_int_type:
      case Uintah::TypeDescription::long_type:
      case Uintah::TypeDescription::long64_type:
        cerr << "Subtype " << subtype->getName() << " is not implemented...\n";
        exit(1);
        break;
      default:
        cerr << "Unknown subtype\n";
        exit(1);
      }
    }


    // Passing the 'variables', a vector member variable to the function. This is where all the particles, along with the variables, 
    // get stored. 

    if( do_particles ) {
      //saveParticleData( particleDataArray, *(timeStepObjPtr->varColln) );
    }

    return timeStepObjPtr;


  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << "\n";
    exit(1);
  } catch(...){
    cerr << "Caught unknown exception\n";
    exit(1);
  }
}

