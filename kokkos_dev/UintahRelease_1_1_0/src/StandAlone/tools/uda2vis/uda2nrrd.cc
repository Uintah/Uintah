/*

   The MIT License

   Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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

Args args;

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
int*
getPeriodicBoundaries(const string& input_uda_name, int timeStepNo, int levelNo) {
  DataArchive* archive = scinew DataArchive(input_uda_name);

  vector<int> index;
  vector<double> times;

  int* boundaryExists = new int[3];

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);
  int numLevels = grid->numLevels();

  LevelP level;

  if (levelNo < numLevels) {
    level = grid->getLevel(levelNo);
    IntVector a = level->getPeriodicBoundaries();

    boundaryExists[0] = a.x();   
    boundaryExists[1] = a.y();   
    boundaryExists[2] = a.z();   
  }

  delete archive;
  return boundaryExists;
}

/////////////////////////////////////////////////////////////////////
extern "C"
int*
getExtraCells(const string& input_uda_name, int timeStepNo, int levelNo) {
  DataArchive* archive = scinew DataArchive(input_uda_name);

  vector<int> index;
  vector<double> times;

  int* extraCells = new int[3];

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);
  int numLevels = grid->numLevels();

  LevelP level;

  if (levelNo < numLevels) {
    level = grid->getLevel(levelNo);
    IntVector a = level->getExtraCells();

    extraCells[0] = a.x();   
    extraCells[1] = a.y();   
    extraCells[2] = a.z();   
  }

  delete archive;
  return extraCells;
}

/////////////////////////////////////////////////////////////////////
extern "C"
levelPatchVec*
getTotalNumPatches(const string& input_uda_name, int timeStepNo) {
  DataArchive* archive = scinew DataArchive(input_uda_name);

  vector<int> index;
  vector<double> times;

  // int* numPatches = new int();
  // *numPatches = 0;

  levelPatchVec* levelPatchVecPtr = new levelPatchVec();

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);
  int numLevels = grid->numLevels();

  LevelP level;
  for (int i = 0; i < numLevels; i++) {
    level = grid->getLevel(i);
    IntVector rr = level->getRefinementRatio();
    // cout << "Refinement ratio, Level " << i << ": " << level->getRefinementRatio() << endl;
    levelPatch levelPatchObj(i, level->numPatches(), rr.x(), rr.y(), rr.z());
    levelPatchVecPtr->push_back(levelPatchObj);
    // *numPatches += level->numPatches();
  }	

  // return numPatches;

  delete archive;
  return levelPatchVecPtr;
}

/////////////////////////////////////////////////////////////////////
extern "C"
int*
getNumPatches(const string& input_uda_name, int timeStepNo, int levelNo) {
  DataArchive* archive = scinew DataArchive(input_uda_name);

  vector<int> index;
  vector<double> times;

  int* numPatches = new int();
  // *numPatches = 0;

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);

  LevelP level;
  level = grid->getLevel(levelNo);
  *numPatches = level->numPatches();

  delete archive;
  return numPatches;
}

/////////////////////////////////////////////////////////////////////
extern "C"
int*
getNumLevels(const string& input_uda_name, int timeStepNo) {
  DataArchive* archive = scinew DataArchive(input_uda_name);

  vector<int> index;
  vector<double> times;

  int* numLevels = new int();
  // *numLevels = 0;

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);
  *numLevels = grid->numLevels();

  delete archive;
  return numLevels;
}

/////////////////////////////////////////////////////////////////////
extern "C"
varMatls*
getMaterials(const string& input_uda_name, const string& variable_name, int timeStepNo) {
  DataArchive* archive = scinew DataArchive(input_uda_name);
  varMatls* varMatlList = new varMatls();

  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);
  int numLevels = grid->numLevels();

  LevelP level;
  for (int i = 0; i < numLevels; i++) {
    level = grid->getLevel(i);
    const Patch* patch = *(level->patchesBegin());
    ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, timeStepNo);
    if (matls.size() > 0) { // Found particles, volume data should also be present there
      for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
	  matlIter != matls.end(); matlIter++) {
	varMatlList->push_back(*matlIter);
      }
      break;
    }
    // break;
  }		

  // LevelP level;
  // level = grid->getLevel(0); // this may have to be modified as for particle data + AMR, its only on one level

  // const Patch* patch = *(level->patchesBegin());
  // ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, timeStepNo);

  /*for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
    matlIter != matls.end(); matlIter++) {
    varMatlList->push_back(*matlIter);
    }*/	

  delete archive;
  return varMatlList;
}

/////////////////////////////////////////////////////////////////////
extern "C"
double*
getBBox(const string& input_uda_name, int timeStepNo, int levelNo) {
  DataArchive* archive = scinew DataArchive(input_uda_name);

  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);

  LevelP level;
  BBox box;

  level = grid->getLevel(levelNo);
  level->getSpatialRange(box);

  Point min = box.min();
  Point max = box.max();

  double *minMaxArr = new double[6];

  minMaxArr[0] = min.x(); minMaxArr[1] = min.y(); minMaxArr[2] = min.z();
  minMaxArr[3] = max.x(); minMaxArr[4] = max.y(); minMaxArr[5] = max.z();

  delete archive;
  return minMaxArr;
} 

/////////////////////////////////////////////////////////////////////
extern "C"
int*
getPatchIndex(const string& input_uda_name, int timeStepNo, int levelNo, int patchNo, 
    const string& varType) { 
  // Modify this to include basis order, as implemented in handlePatchData
  // NC - 0, CC, SFCX, SFCY, SFCZ - 1
  DataArchive* archive = scinew DataArchive(input_uda_name);

  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  // This is needed here (it sets a member variable), without this queryGrid won't work
  archive->queryTimesteps(index, times);

  GridP grid = archive->queryGrid(timeStepNo);

  LevelP level;

  level = grid->getLevel(levelNo);
  const Patch* patch = level->getPatch(patchNo);

  IntVector patch_lo, patch_hi, low, hi;

  if(varType.find("CC") != string::npos) {
    patch_lo = patch->getExtraCellLowIndex__New();
    patch_hi = patch->getExtraCellHighIndex__New();
  } 
  else {
    patch_lo = patch->getExtraNodeLowIndex__New();
    // switch (qinfo.type->getType()) {
    // case Uintah::TypeDescription::SFCXVariable:
    if(varType.find("SFCX") != string::npos)
      patch_hi = patch->getSFCXHighIndex__New();
    // break;
    // case Uintah::TypeDescription::SFCYVariable:
    else if(varType.find("SFCY") != string::npos)
      patch_hi = patch->getSFCYHighIndex__New();
    // break;
    // case Uintah::TypeDescription::SFCZVariable:
    else if(varType.find("SFCZ") != string::npos)
      patch_hi = patch->getSFCZHighIndex__New();
    // break;
    // case Uintah::TypeDescription::NCVariable:
    else if(varType.find("NC") != string::npos)
      patch_hi = patch->getNodeHighIndex__New();
    // break;
    // default:
    // cerr << "build_field::unknown variable.\n";
    // exit(1);
  }

  // IntVector range = patch_hi - patch_lo;
  // cout << range << endl;

  level->findIndexRange(low, hi);

  // patch_lo = patch_lo - low;
  // patch_hi = patch_hi - low;

  // cout << patch_lo << " " << patch_hi << endl;

  patch_lo = patch->getExtraCellLowIndex__New() - low;
  patch_hi = patch->getExtraCellHighIndex__New() - low;

  int *indexArr = new int[6];

  indexArr[0] = patch_lo.x(); 
  indexArr[1] = patch_lo.y(); 
  indexArr[2] = patch_lo.z();
  indexArr[3] = patch_hi.x(); 
  indexArr[4] = patch_hi.y(); 
  indexArr[5] = patch_hi.z();

  cout << indexArr[3] - indexArr[0] << " " << indexArr[4] - indexArr[1] << " " << indexArr[5] - indexArr[2] << "\n";

  delete archive;
  return indexArr;
  }

  /////////////////////////////////////////////////////////////////////
  extern "C"
    patchInfoVec*
    getPatchInfo(const string& input_uda_name, int timeStepNo, const string& varType, 
	bool remove_boundary) {
      DataArchive* archive = scinew DataArchive(input_uda_name);

      vector<int> index;
      vector<double> times;

      // query time info from dataarchive
      // This is needed here (it sets a member variable), without this queryGrid won't work
      archive->queryTimesteps(index, times);

      GridP grid = archive->queryGrid(timeStepNo);
      int numLevels = grid->numLevels();

      patchInfoVec* patchInfoVecPtr = new patchInfoVec();

      IntVector patch_lo, patch_hi, low, hi;
      Point min, max;
      int *hiLoArr = new int[6];
      int *indexArr = new int[6];
      double *minMaxArr = new double[6];

      LevelP level;
      for (int i = 0; i < numLevels; i++) {
	level = grid->getLevel(i);

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
	      patch_lo = patch->getCellLowIndex__New();
	      patch_hi = patch->getCellHighIndex__New();
	    } 
	    else {
	      patch_lo = patch->getNodeLowIndex__New();
	      if(varType.find("SFCX") != string::npos) {
		// patch_hi = patch->getHighIndex(Patch::XFaceBased);
		/*if (patch_hi.x() < hi.x()) {
		  patch_hi = IntVector(patch_hi.x() + 1, patch_hi.y(), patch_hi.z());
		}*/

		patch_hi = patch_lo + (patch->getCellHighIndex__New() - patch->getCellLowIndex__New());
		if (patch_hi.x() == (hi.x() - 1)) {
		  patch_hi = IntVector(patch_hi.x() + 1, patch_hi.y(), patch_hi.z());
		}
              }		
	      else if(varType.find("SFCY") != string::npos)
		patch_hi = patch->getHighIndex(Patch::YFaceBased);
	      else if(varType.find("SFCZ") != string::npos)
		patch_hi = patch->getHighIndex(Patch::ZFaceBased);
	      else if(varType.find("NC") != string::npos) {
		patch_hi = patch_lo + (patch->getCellHighIndex__New() - patch->getCellLowIndex__New()) + IntVector(1, 1, 1);
		/*if (!(patch_hi.x() == hi.x())) {
		  patch_hi = IntVector(patch_hi.x() - 1, patch_hi.y(), patch_hi.z());
		}*/  
		/*if (!(patch_hi.y() == hi.y())) {
		  patch_hi = IntVector(patch_hi.x(), patch_hi.y() - 1, patch_hi.z());
		}*/  
		/*if (!(patch_hi.z() == hi.z())) {
		  patch_hi = IntVector(patch_hi.x(), patch_hi.y(), patch_hi.z() - 1);
		}*/  
		// patch_hi = patch->getNodeHighIndex__New();
	      }
	    }
	  }
	  else { // don't remove the boundary
	    if(varType.find("CC") != string::npos) {
	      patch_lo = patch->getExtraCellLowIndex__New();
	      patch_hi = patch->getExtraCellHighIndex__New();
	    } 
	    else {
	      patch_lo = patch->getExtraNodeLowIndex__New();
	      if(varType.find("SFCX") != string::npos)
		patch_hi = patch->getSFCXHighIndex__New();
	      else if(varType.find("SFCY") != string::npos)
		patch_hi = patch->getSFCYHighIndex__New();
	      else if(varType.find("SFCZ") != string::npos)
		patch_hi = patch->getSFCZHighIndex__New();
	      else if(varType.find("NC") != string::npos)
		patch_hi = patch->getExtraNodeHighIndex__New();
	    }
	  }

	  // cout << patch->getID() << " " << patch_lo << " " << patch_hi << endl;

	  /*if(remove_boundary) {
	    level->findInteriorIndexRange(low, hi);
	    } 
	    else {
	    level->findIndexRange(low, hi);
	    }*/	

	  // Moved above
	  // level->findIndexRange(low, hi);

	  indexArr[0] = patch_lo.x(); 
	  indexArr[1] = patch_lo.y(); 
	  indexArr[2] = patch_lo.z();
	  indexArr[3] = patch_hi.x(); 
	  indexArr[4] = patch_hi.y(); 
	  indexArr[5] = patch_hi.z();  

	  /*if ((j == 0) || (j == 1)) { 
	    cout << indexArr[0] << " " << indexArr[1] << " " << indexArr[2] << endl;
	    cout << indexArr[3] << " " << indexArr[4] << " " << indexArr[5] << endl;
	    }*/

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

	  int nCells = (patch->getCellHighIndex__New() - patch->getCellLowIndex__New()).y();

	  patchInfo patchInfoObj(indexArr, minMaxArr, hiLoArr, nCells);
	  patchInfoVecPtr->push_back(patchInfoObj);
	}
      }		

      delete archive;
      return patchInfoVecPtr;
    }

  /////////////////////////////////////////////////////////////////////
  extern "C"
    double*
    getPatchBBox(const string& input_uda_name, int timeStepNo, int levelNo, int patchNo) {
      DataArchive* archive = scinew DataArchive(input_uda_name);

      vector<int> index;
      vector<double> times;

      // query time info from dataarchive
      // This is needed here (it sets a member variable), without this queryGrid won't work
      archive->queryTimesteps(index, times);

      GridP grid = archive->queryGrid(timeStepNo);

      LevelP level;

      level = grid->getLevel(levelNo);
      const Patch* patch = level->getPatch(patchNo);

      // BBox bbox;
      // level->getSpatialRange(bbox);

      // Point min = bbox.min();
      // Point max = bbox.max();

      // Vector length = max - min;

      // IntVector hi, lo, patch_hi, patch_lo, range;

      // patch_lo = patch->getCellLowIndex();
      // patch_hi = patch->getCellHighIndex();

      // level->findIndexRange(lo, hi);

      // range = hi - lo;

      // cout << "patch_lo: " << patch_lo << endl;
      // cout << "patch_hi: " << patch_hi << endl;
      // cout << "lo: " << lo << endl;
      // cout << "hi: " << hi << endl;

      // double lowerx((double)(patch_lo.x() - lo.x()) / range.x());
      // double upperx((double)(patch_hi.x() - lo.x()) / range.x());

      // double lowery((double)(patch_lo.y() - lo.y()) / range.y());
      // double uppery((double)(patch_hi.y() - lo.y()) / range.y());

      // double lowerz((double)(patch_lo.z() - lo.z()) / range.z());
      // double upperz((double)(patch_hi.z() - lo.z()) / range.z());

      Point min = patch->getExtraBox().lower();
      Point max = patch->getExtraBox().upper();

      // cout << "Min: " << lower << " Max: " << upper << endl; 

      double *minMaxArr = new double[6];

      minMaxArr[0] = min.x(); minMaxArr[1] = min.y(); minMaxArr[2] = min.z();
      minMaxArr[3] = max.x(); minMaxArr[4] = max.y(); minMaxArr[5] = max.z();

      // minMaxArr[0] = min.x() + lowerx * length.x(); 
      // minMaxArr[1] = min.y() + lowery * length.y(); 
      // minMaxArr[2] = min.z() + lowerz * length.z();

      // minMaxArr[3] = min.x() + upperx * length.x(); 
      // minMaxArr[4] = min.y() + uppery * length.y(); 
      // minMaxArr[5] = min.z() + upperz * length.z();

      delete archive;
      return minMaxArr;
    }

  /////////////////////////////////////////////////////////////////////
  /*extern "C"
    int*
    getTimeSteps(const string& input_uda_name) {
    DataArchive* archive = scinew DataArchive(input_uda_name);

  // Get the times and indices.
  vector<int> index;
  vector<double> times;

  // query time info from dataarchive
  archive->queryTimesteps(index, times);

  int* noTimeSteps = new int(index.size());
  return noTimeSteps;
  }*/ 

  /////////////////////////////////////////////////////////////////////
  extern "C"
    // int*
    typeDouble*
    getTimeSteps(const string& input_uda_name) {
      DataArchive* archive = scinew DataArchive(input_uda_name);
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

      delete archive;
      return timeStepInfo;

      // int* noTimeSteps = new int(index.size());
      // return noTimeSteps;
    } 


  /////////////////////////////////////////////////////////////////////
  extern "C"
    udaVars*
    getVarList(const string& input_uda_name) {
      DataArchive* archive = scinew DataArchive(input_uda_name);
      udaVars* udaVarList = new udaVars();

      vector<string> vars;
      vector<const Uintah::TypeDescription*> types;

      archive->queryVariables(vars, types);

      for (unsigned int i = 0; i < vars.size(); i++) {
	string nameType = vars[i] + "/" + types[i]->getName(); 
	udaVarList->push_back(nameType);
	// cout << vars[i] << " " << types[i]->getName() << endl;
      }

      delete archive;
      return udaVarList;
    } 
  
  
  /////////////////////////////////////////////////////////////////////
  extern "C"
    int*
    getPVarLevelAndPatches(const string& input_uda_name,
                           const string& varName,
			   int timeStepNo) {
      DataArchive* archive = scinew DataArchive(input_uda_name);
            
      vector<int> index;
      vector<double> times;

      // query time info from dataarchive
      // This is needed here (it sets a member variable), without this queryGrid won't work
      archive->queryTimesteps(index, times);
      GridP grid = archive->queryGrid(timeStepNo);
      
      int* levelAndPatches = new int(2);
      	      
      bool found_particle_level = false;
      for( int lev = 0; lev < grid->numLevels(); lev++ ) {
        LevelP particleLevel = grid->getLevel( lev );
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
      
      delete archive;
      return levelAndPatches;
    } 
  
 
  /////////////////////////////////////////////////////////////////////
  extern "C"
    timeStep*
    processData(int argc, char argv[][128], 
	int timeStepNo, 
	bool dataReq, 
	int matlNo, 
	bool matlClassfication, 
	const string& varSelected,
        int patchNo) 
    {
      /*
       * Default values
       */
      bool do_binary=false;

      unsigned long time_step_lower = 0;
      // default to be last timestep, but can be set to 0
      unsigned long time_step_upper = (unsigned long)-1;
      unsigned long tinc = 1;

      string input_uda_name;
      string output_file_name("");
      int    output_file_index = 0; // Beginning index for modifying output file name.
      bool use_default_file_name = true;
      IntVector var_id(0,0,0);
      string variable_name("");
      // It will use the first material found unless other indicated.
      int material = -1;
      int level_index = 0;

      bool do_particles = false;

      // cout << argc << " " << *(argv[5]) << endl;

      /*
       * Parse arguments
       */
      for(int i=1;i<argc;i++){
	string s=argv[i];
	if(s == "-v" || s == "--variable") {
	  if( do_particles ) {
	    cout << "\n";
	    cout << "Error: you may only use -v or -p, not both!\n";
	    cout << "\n";
	    usage( "", argv[0] );
	  }
	  variable_name = string(argv[++i]);
	} else if (s == "-p" || s == "--particledata") {
	  if( args.use_all_levels ) {
	    cout << "\n";
	    cout << "Error: you may not use -a and -p at the same time!\n";
	    cout << "\n";
	    usage( "", argv[0] );
	  }
	  if( variable_name != "" ) {
	    cout << "\n";
	    cout << "Error: you may only use -v or -p, not both!\n";
	    cout << "\n";
	    usage( "", argv[0] );
	  }
	  do_particles = true;
	} else if (s == "-m" || s == "--material") {
	  material = atoi(argv[++i]);
	} else if (s == "-l" || s == "--level") {
	  level_index = atoi(argv[++i]);
	} else if (s == "-a" || s == "--all"){
	  args.use_all_levels = true;
	  if( do_particles ) {
	    cout << "\n";
	    cout << "Error: you may not use -a and -p at the same time!\n";
	    cout << "\n";
	    usage( "", argv[0] );
	  }
	} else if (s == "-vv" || s == "--verbose") {
	  args.verbose = true;
	} else if (s == "-q" || s == "--quiet") {
	  args.quiet = true;
	} else if (s == "-tlow" || s == "--timesteplow") {
	  time_step_lower = strtoul(argv[++i],(char**)NULL,10);
	} else if (s == "-thigh" || s == "--timestephigh") {
	  time_step_upper = strtoul(argv[++i],(char**)NULL,10);
	} else if (s == "-tstep" || s == "--timestep") {
	  time_step_lower = strtoul(argv[++i],(char**)NULL,10);
	  time_step_upper = time_step_lower;
	} else if (s == "-tinc") {
	  tinc = strtoul(argv[++i],(char**)NULL,10);
	} else if (s == "-i" || s == "--index") {
	  int x = atoi(argv[++i]);
	  int y = atoi(argv[++i]);
	  int z = atoi(argv[++i]);
	  var_id = IntVector(x,y,z);
	} else if( s ==  "-dh" || s == "--detatched-header") {
	  args.attached_header = false;
	} else if( (s == "-h") || (s == "--help") ) {
	  usage( "", argv[0] );
	} else if (s == "-uda") {
	  input_uda_name = string(argv[++i]);
	} else if (s == "-oi" ) {
	  output_file_index = atoi(argv[++i]);
	} else if (s == "-o" || s == "--out") {
	  output_file_name = string(argv[++i]);
	  use_default_file_name = false;
	} else if(s == "-mo") {
	  s = argv[++i];
	  if (s == "det")
	    args.matrix_op = Det;
	  else if (s == "norm")
	    args.matrix_op = Norm;
	  else if (s == "trace")
	    args.matrix_op = Trace;
	  else if (s == "none")
	    args.matrix_op = None;
	  else
	    usage(s, argv[0]);
	} else if(s == "-binary") {
	  do_binary=true;
	} else if(s == "-nbc" || s == "--noboundarycells") {
	  args.remove_boundary = true;
	} else {
	  usage(s, argv[0]);
	}
      }

      if(input_uda_name == ""){
	cerr << "No archive file specified\n";
	usage("", argv[0]);
      }

      try {
	DataArchive* archive = scinew DataArchive(input_uda_name);

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
	vector<unsigned int> var_indices;

	if( do_particles ) {
	  unsigned int vi = 0;
	  for( ; vi < vars.size(); vi++ ) {
	    // if( vars[vi][0] == 'p' && vars[vi][1] == '.' ) { // starts with "p."
	    if ( types[vi]->getType() == Uintah::TypeDescription::ParticleVariable ) { 
	      // It is a particle variable
	      var_indices.push_back( vi );
	    }
	  }
	  if( var_indices.size() == 0 ) {
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
	  var_indices.push_back( vi );
	}

	if( use_default_file_name ) { // Then use the variable name for the output name

	  if( do_particles ) {
	    output_file_name = "particles";
	  } else {
	    output_file_name = variable_name;
	  }
	  if( !args.quiet ) {
	    cout << "Using variable name (" << output_file_name
	      << ") as output file base name.\n";
	  }
	}

	/////////////////////////////////////////////////////
	// figure out the lower and upper bounds on the timesteps
	if (time_step_lower >= times.size()) {
	  cerr << "timesteplow must be between 0 and " << times.size()-1 << "\n";
	  exit(1);
	}

	// set default max time value
	if (time_step_upper == (unsigned long)-1) {
	  if( args.verbose )
	    cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
	  time_step_upper = times.size() - 1;
	}

	if (time_step_upper >= times.size() || time_step_upper < time_step_lower) {
	  cerr << "timestephigh("<<time_step_lower<<") must be greater than " << time_step_lower 
	    << " and less than " << times.size()-1 << "\n";
	  exit(1);
	}

	if( !args.quiet ) { 
	  if( time_step_lower != time_step_upper ) {
	    cout << "Extracting data from timesteps " << time_step_lower << " to " << time_step_upper << ".  "
	      << "Times: " << times[time_step_lower] << " to " << times[time_step_upper]
	      << "\n";
	  } else {
	    cout << "Extracting data from timestep " << time_step_lower << " (time: " << times[time_step_lower] << ").\n";
	  }
	}

	// Create a vector of class timeStep. This is where we store all time step data.
	// udaData *dataBank = new udaData(); // No longer needed

	////////////////////////////////////////////////////////
	// Loop over each timestep
	// for( unsigned long time = time_step_lower; time <= time_step_upper; time += tinc ) {

	unsigned long time = time_step_lower + timeStepNo * tinc;

	/////////////////////////////
	// Figure out the filename

	char filename_num[200];
	sprintf( filename_num, "_t%06d", index[time] );

	string filename( output_file_name + filename_num );

	// Check the level index
	double current_time = times[time];
	GridP grid = archive->queryGrid(time);
	if (level_index >= grid->numLevels() || level_index < 0) {
	  cerr << "level index is bad ("<<level_index<<").  Should be between 0 and "<<grid->numLevels()<<".\n";
	  cerr << "Trying next timestep.\n";
	  exit(1); 
	  // continue;
	}

	vector<ParticleDataContainer> particleDataArray;

	// Create a timeStep object, corresponding to every time step.
	timeStep* timeStepObjPtr = new timeStep();

	// Storing the time step name/ no.
	timeStepObjPtr->name.assign(filename_num + 1);
	timeStepObjPtr->no = index[time];

	LevelP level;

	// Loop over the specified variable(s)...
	//
	// ... Currently you can only specify one grid var, or all particles vars.
	// ... This loop is used to run over the one grid var, or over all the particle vars...
	// ... However, it should be easy to allow the user to create multiple grid var
	// ... NRRDs at the same time using this loop...

	// p.x should always be at the top 
	for (unsigned int varCount = 0; varCount < var_indices.size(); varCount++) {
	  if (vars[var_indices[varCount]].compare("p.x") == 0) {
	    unsigned int tmpIndex = var_indices[0];
	    var_indices[0] = var_indices[varCount];
	    var_indices[varCount] = tmpIndex;
	    break;
	  }
	}

	for( unsigned int cnt = 0; cnt < var_indices.size(); cnt++ ) {

	  unsigned int var_index = var_indices[cnt];
	  variable_name = vars[var_index];

	  if( !args.quiet ) {
	    // cout << "Extracting data for " << vars[var_index] << ": " << types[var_index]->getName() << "\n";
	  }

	  //////////////////////////////////////////////////
	  // Set the level pointer

	  if( level.get_rep() == NULL ) {  // Only need to get the level for the first timestep... as
	    // the data will be on the same level(s) for all timesteps.
	    if( do_particles ) { // Determine which level the particles are on...
	      bool found_particle_level = false;
	      for( int lev = 0; lev < grid->numLevels(); lev++ ) {
		LevelP particleLevel = grid->getLevel( lev );
		const Patch* patch = *(particleLevel->patchesBegin());
		ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, time);
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
		level = grid->getLevel( 0 );
		if( grid->numLevels() == 1 ){ // only one level to use
		  args.use_all_levels = false;
		}
	      } else {  // set to requested level
		level = grid->getLevel(level_index);
	      }
	    }
	  }

	  ///////////////////////////////////////////////////
	  // Check the material number.

	  const Patch* patch = *(level->patchesBegin());
	  ConsecutiveRangeSet matls = archive->queryMaterials(variable_name, patch, time);

	  if( args.verbose ) {
	    // Print out all the material indicies valid for this timestep
	    cout << "Valid materials for " << variable_name << " at time[" << time << "](" << current_time << ") are:  ";
	    for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
		matlIter != matls.end(); matlIter++) {
	      cout << *matlIter << ", ";
	    }
	    cout << "\n";
	  }

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
		continue;
	      }
	    }
	  }

	  // get type and subtype of data
	  const Uintah::TypeDescription* td = types[var_index];
	  const Uintah::TypeDescription* subtype = td->getSubType();

	  QueryInfo qinfo( archive, grid, level, variable_name, materialsOfInterest,
	      time, args.use_all_levels, td );

	  IntVector hi, low, range;
	  BBox box;

	  // Remove the edges if no boundary cells
	  if( args.remove_boundary ){
	    level->findInteriorIndexRange(low, hi);
	    level->getInteriorSpatialRange(box);
	  } else {
	    // level->findIndexRange(low, hi);
	    // level->getSpatialRange(box);

	    const Patch* patch = level->getPatch(patchNo);

	    Point min = patch->getExtraBox().lower();
	    Point max = patch->getExtraBox().upper();

	    IntVector extraCells = patch->getExtraCells();
	    IntVector noCells = patch->getCellHighIndex__New() - patch->getCellLowIndex__New();

	    // cout << noCells << endl;

	    box = BBox(min, max);

	    // low = patch->getCellLowIndex__New() - extraCells;
	    // hi = patch->getCellHighIndex__New() + extraCells;

	    low = patch->getNodeLowIndex__New() - extraCells;
	    hi = patch->getNodeLowIndex__New() + noCells + extraCells + IntVector(1, 1, 1);
	    
	    // Point min = level->getBox(low, hi).lower(); 
	    // Point max = level->getBox(low, hi).upper(); 
	    
	    // box = BBox(min, max);
	  
	    // cout << "Min/Max: " << min << " --- " << max << endl;
	  }

	  // this is a hack to make things work, substantiated in build_multi_level_field()
	  range = hi - low /*+ IntVector(1, 1, 1)*/;

	  // cout << "Low/hi/range: " << low << " " << hi << " " << range << endl;

	  /*if (qinfo.type->getType() == Uintah::TypeDescription::CCVariable) {
	    IntVector cellLo, cellHi;
	    if( args.remove_boundary ) {
	      level->findInteriorCellIndexRange(cellLo, cellHi);
	    } else {
	      level->findCellIndexRange(cellLo, cellHi);
	    }
	    if (is_periodic_bcs(cellHi, hi)) {
	      // cout << "is_periodic_bcs test\n";
	      IntVector newrange(0,0,0);
	      get_periodic_bcs_range(cellHi, hi, range, newrange);
	      range = newrange;
	    }
	  }*/

	  // Adjust the range for using all levels
	  if( args.use_all_levels && grid->numLevels() > 0 ){
	    double exponent = grid->numLevels() - 1;
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

	    ParticleDataContainer data;

	    switch (subtype->getType()) {
	      case Uintah::TypeDescription::double_type:
		/*data =*/ handleParticleData<double>( qinfo, matlNo, matlClassfication, data, varSelected, patchNo );
		break;
	      case Uintah::TypeDescription::float_type:
		/*data =*/ handleParticleData<float>( qinfo, matlNo, matlClassfication, data, varSelected, patchNo );
		break;
	      case Uintah::TypeDescription::int_type:
		/*data =*/ handleParticleData<int>( qinfo, matlNo, matlClassfication, data, varSelected, patchNo );
		break;
	      case Uintah::TypeDescription::long64_type:
		/*data =*/ handleParticleData<long64>( qinfo, matlNo, matlClassfication, data, varSelected, patchNo );
		break;
	      case Uintah::TypeDescription::Point:
		/*data =*/ handleParticleData<Point>( qinfo, matlNo, matlClassfication, data, varSelected, patchNo );
		break;
	      case Uintah::TypeDescription::Vector:
		/*data =*/ handleParticleData<Vector>( qinfo, matlNo, matlClassfication, data, varSelected, patchNo );
		break;
	      case Uintah::TypeDescription::Matrix3:
		/*data =*/ handleParticleData<Matrix3>( qinfo, matlNo, matlClassfication, data, varSelected, patchNo );
		break;
	      default:
		cerr << "Unknown subtype for particle data: " << subtype->getName() << "\n";
		exit(1);
	    } // end switch( subtype )

	    particleDataArray.push_back( data );

	  } else { // Handle Grid Variables

	    switch (subtype->getType()) {
	      case Uintah::TypeDescription::double_type:
		timeStepObjPtr->cellValColln = new cellVals();
		handleVariable<double>( qinfo, low, hi, range, box, filename, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
		break;
	      case Uintah::TypeDescription::float_type:
		timeStepObjPtr->cellValColln = new cellVals();
		handleVariable<float>( qinfo, low, hi, range, box, filename, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
		break;
	      case Uintah::TypeDescription::int_type:
		timeStepObjPtr->cellValColln = new cellVals();
		handleVariable<int>( qinfo, low, hi, range, box, filename, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
		break;
	      case Uintah::TypeDescription::Vector:
		timeStepObjPtr->cellValColln = new cellVals();
		handleVariable<Vector>( qinfo, low, hi, range, box, filename, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
		break;
	      case Uintah::TypeDescription::Matrix3:
		timeStepObjPtr->cellValColln = new cellVals();
		handleVariable<Matrix3>( qinfo, low, hi, range, box, filename, args, *(timeStepObjPtr->cellValColln), dataReq, patchNo );
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
	} // end variables loop

	// Passing the 'variables', a vector member variable to the function. This is where all the particles, along with the variables, 
	// get stored. 

	if( do_particles ) {
	  timeStepObjPtr->varColln = new variables();
	  saveParticleData( particleDataArray, filename, *(timeStepObjPtr->varColln) );
	}

	// Adding time step object to the data bank 
	// dataBank->push_back(timeStepObj);

	// particleDataArray.clear();

	delete archive;
	return timeStepObjPtr;

	// } // end time step loop

      } catch (Exception& e) {
	cerr << "Caught exception: " << e.message() << "\n";
	exit(1);
      } catch(...){
	cerr << "Caught unknown exception\n";
	exit(1);
      }
    }

