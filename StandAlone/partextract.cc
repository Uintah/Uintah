/*
 *  partextract.cc: Print out a uintah data archive for particle data
 *
 *  Written by:
 *   Biswajit Banerjee
 *   July 2004
 *
 */

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>
#include <Core/Containers/Array3.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

Mutex cerrLock( "cerrLock" );

typedef struct{
  vector<ShareAssignParticleVariable<double> > pv_double_list;
  vector<ShareAssignParticleVariable<float> > pv_float_list;
  vector<ShareAssignParticleVariable<Point> > pv_point_list;
  vector<ShareAssignParticleVariable<Vector> > pv_vector_list;
  vector<ShareAssignParticleVariable<Matrix3> > pv_matrix3_list;
  ShareAssignParticleVariable<Point> p_x;
} MaterialData;

// takes a string and replaces all occurances of old with newch
string replaceChar(string s, char old, char newch) {
  string result;
  for (int i = 0; i<(int)s.size(); i++)
    if (s[i] == old)
      result += newch;
    else
      result += s[i];
  return result;
}

void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "") cerr << "Error parsing argument: " << badarg << endl;

  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h[elp]\n";
  cerr << "  -partvar <variable name>\n";
  cerr << "  -partid <particleid>\n";
  cerr << "  -part_stress [avg or equiv or all]\n";
  cerr << "  -part_strain [avg / true / all / lagrange / euler]\n";
  cerr << "  -timesteplow [int] (only outputs timestep from int)\n";
  cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
  cerr << "USAGE IS NOT FINISHED\n\n";
  exit(1);
}

////////////////////////////////////////////////////////////////////////////
//
// Get particle strains (avg, true or all)
//
////////////////////////////////////////////////////////////////////////////
void 
getParticleStrains(DataArchive* da, string flag) {

  // Parse the flag and check which option is needed
  bool doAverage = false;
  bool doTrue = false;
  bool doLagrange = false;
  bool doEuler = false;
  //   bool doAll = false;
  if (flag == "avg") doAverage = true;
  else if (flag == "true") doTrue = true;
  else if (flag == "lagrange") doLagrange = true;
  else if (flag == "euler") doEuler = true;
  //   else doAll = true;

  // Check if all the required variables are there .. for all cases
  // we need p.deformationMeasure and for the volume average we need p.volume
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());

  bool gotVolume = false;
  bool gotDeform = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == "p.volume") gotVolume = true;
    if (var == "p.deformationMeasure") gotDeform = true;
  }
  if (!gotDeform) {
    cerr << "\n **Error** getParticleStrains : DataArchiver does not "
         << "contain p.deformationMeasure\n";
    exit(1);
  }
  if (doAverage && !gotVolume) {
    cerr << "\n **Error** getParticleStrains : DataArchiver does not "
         << "contain p.volume\n";
    exit(1);
  }

  // Now that the variables have been found, get the data for all available 
  // time steps  from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  //cout << "There are " << index.size() << " timesteps:\n";
      
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = times.size() - 1 ;
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(time);

    vector<double> volumeVector;
    vector<Matrix3> deformVector;
    double totVol = 0.0;

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patchIterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
	const Patch* patch = *iter;
        ++patchIndex; 

	// Loop thru all the variables 
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];

	  // Check if the variable is a ParticleVariable
	  if(td->getType() == Uintah::TypeDescription::ParticleVariable) { 

	    // loop thru all the materials
	    ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	    ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
	    for(; matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;

	      // Find the name of the variable
	      if (doAverage) {
		if (var == "p.volume") {
		  switch(td->getSubType()->getType())
		  {
		  case Uintah::TypeDescription::double_type:
		    {
		      ParticleVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      if(pset->numParticles() > 0){
			ParticleSubset::iterator iter = pset->begin();
			for(;iter != pset->end(); iter++){
			  volumeVector.push_back(value[*iter]);
			  totVol += value[*iter];
			}
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      ParticleVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      if(pset->numParticles() > 0){
			ParticleSubset::iterator iter = pset->begin();
			for(;iter != pset->end(); iter++){
			  volumeVector.push_back((double)(value[*iter]));
			  totVol += value[*iter];
			}
		      }
		    }
		  break;
		  default:
		    cerr << "Particle Variable of unknown type: " 
			 << td->getSubType()->getType() << endl;
		    break;
		  }
		}
	      }
	      if (var == "p.deformationMeasure") {
	        //cout << "Material: " << matl << endl;
		ParticleVariable<Matrix3> value;
		da->query(value, var, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		if(pset->numParticles() > 0){
		  ParticleSubset::iterator iter = pset->begin();
		  for(;iter != pset->end(); iter++){
                    particleIndex idx = *iter;

                    // Find the right stretch tensor
                    Matrix3 deformGrad = value[*iter];
                    Matrix3 stretch(0.0), rotation(0.0);
                    deformGrad.polarDecomposition(stretch, rotation, 
                                                  1.0e-10, true);

                    if (doAverage) deformVector.push_back(stretch);
                    else {

                      if (doTrue) {

		        // Find the eigenvalues of the stretch
                        double lambda[3];
                        int numEigen = stretch.getEigenValues(lambda[0], 
                                                              lambda[1], 
                                                              lambda[2]);

			//cout << "True Strain = ";
                        cout << time << " " << patchIndex << " " << matl ;
                        for (int ii = 0; ii < numEigen; ++ii) {
                          double trueStrain = 0.0;
			  if ((1.0+lambda[ii]) > 0.0) 
                            trueStrain = log(1.0+lambda[ii]);
                          cout << " " << trueStrain ;
                        }
                        cout << endl;
                      } else if (doLagrange) {
                        Matrix3 F = deformGrad;
                        Matrix3 FT = F.Transpose();
                        Matrix3 I; I.Identity();
                        Matrix3 E = (FT*F - I)*0.5;
                        cout.setf(ios::scientific,ios::floatfield);
                        cout.precision(4);
                        cout << time << " " << patchIndex << " " << matl
                             << " " << idx ;
                        for (int ii = 0; ii < 3; ++ii) cout << " " << E(ii,ii);
                        cout << " " << E(1,2); cout << " " << E(2,0);
                        cout << " " << E(0,1);
                        cout << endl;
                      } else if (doEuler) {
                        // Calculate the Almansi-Hamel strain tensor
                        Matrix3 Finv = deformGrad.Inverse();
                        Matrix3 FTinv = Finv.Transpose();
                        Matrix3 I; I.Identity();
                        Matrix3 e = (I - FTinv*Finv)*0.5;
                        cout.setf(ios::scientific,ios::floatfield);
                        cout.precision(4);
                        cout << time << " " << patchIndex << " " << matl
                             << " " << idx ;
                        for (int ii = 0; ii < 3; ++ii) cout << " " << e(ii,ii);
                        cout << " " << e(1,2); cout << " " << e(2,0);
                        cout << " " << e(0,1);
                        cout << endl;
                      } else {
                        //cout << "Deformation Gradient = \n" ;
                        //cout << "Right Stretch = \n";
                        //cout << "Rotation = \n";
                        cout << time << " " << patchIndex << " " << matl ;
                        for (int ii = 0; ii < 3; ++ii) {
                          for (int jj = 0; jj < 3; ++jj) {
			    cout << " " << deformGrad(ii,jj);
                          }
			}
                        cout << endl;
                        cout << time << " " << patchIndex << " " << matl ;
                        for (int ii = 0; ii < 3; ++ii) {
                          for (int jj = 0; jj < 3; ++jj) {
			    cout << " " << stretch(ii,jj);
                          }
			}
                        cout << endl;
                        cout << time << " " << patchIndex << " " << matl ;
                        for (int ii = 0; ii < 3; ++ii) {
                          for (int jj = 0; jj < 3; ++jj) {
			    cout << " " << rotation(ii,jj);
                          }
			}
                        cout << endl;
		      }
		    }
		  }
		}
	      } // end of var compare if
	    } // end of material loop
	  } // end of ParticleVariable if
	} // end of variable loop
      } // end of patch loop
    } // end of level loop

    if (doAverage) {
      // Now that the volume vector and variable vector are available just
      // do a weighted average
      ASSERTEQ(volumeVector.size(), deformVector.size());
      Matrix3 avVar;
      for (unsigned int ii = 0; ii < volumeVector.size() ; ++ii) {
	avVar += ((deformVector[ii]*volumeVector[ii])/totVol);
      }
      for (int ii = 0; ii < 3; ++ii) {
	for (int jj = 0; jj < 3; ++jj) {
	  cout << avVar(ii,jj) << "  " ;
	}
	cout << endl;
      }
    }
  } // end of time step loop
}

////////////////////////////////////////////////////////////////////////////
//
// Get particle stresses (avg, equiv or all)
//
////////////////////////////////////////////////////////////////////////////
void 
getParticleStresses(DataArchive* da, string flag) {

  // Parse the flag and check which option is needed
  bool doAverage = false;
  bool doEquiv = false;
  //   bool doAll = false;
  if (flag == "avg") doAverage = true;
  else if (flag == "equiv") doEquiv = true;
  //   else doAll = true;

  // Check if all the required variables are there .. for all cases
  // we need p.stress and for the volume average we need p.volume
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());

  bool gotVolume = false;
  bool gotStress = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == "p.volume") gotVolume = true;
    if (var == "p.stress") gotStress = true;
  }
  if (!gotStress) {
    cerr << "\n **Error** getParticleStresses : DataArchiver does not "
         << "contain p.stress\n";
    exit(1);
  }
  if (doAverage && !gotVolume) {
    cerr << "\n **Error** getParticleStresses : DataArchiver does not "
         << "contain p.volume\n";
    exit(1);
  }

  // Now that the variables have been found, get the data for all available 
  // time steps // from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  //cout << "There are " << index.size() << " timesteps:\n";
      
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = times.size() - 1 ;
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(time);

    vector<double> volumeVector;
    vector<Matrix3> stressVector;
    double totVol = 0.0;

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patchIterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
	const Patch* patch = *iter;
        ++patchIndex; 

	// Loop thru all the variables 
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];

	  // Check if the variable is a ParticleVariable
	  if(td->getType() == Uintah::TypeDescription::ParticleVariable) { 

	    // loop thru all the materials
	    ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	    ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
	    for(;matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;

	      // Find the name of the variable
	      if (doAverage) {
		if (var == "p.volume") {
		  switch(td->getSubType()->getType())
		  {
		  case Uintah::TypeDescription::double_type:
		    {
		      ParticleVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      if(pset->numParticles() > 0){
			ParticleSubset::iterator iter = pset->begin();
			for(;iter != pset->end(); iter++){
			  volumeVector.push_back(value[*iter]);
			  totVol += value[*iter];
			}
		      }
		    }
		  break;
		  case Uintah::TypeDescription::float_type:
		    {
		      ParticleVariable<float> value;
		      da->query(value, var, matl, patch, time);
		      ParticleSubset* pset = value.getParticleSubset();
		      if(pset->numParticles() > 0){
			ParticleSubset::iterator iter = pset->begin();
			for(;iter != pset->end(); iter++){
			  volumeVector.push_back((double)(value[*iter]));
			  totVol += value[*iter];
			}
		      }
		    }
		  break;
		  default:
		    cerr << "Particle Variable of unknown type: " 
			 << td->getSubType()->getType() << endl;
		    break;
		  }
		}
	      } 
	      if (var == "p.stress") {
		//cout << "Material: " << matl << endl;
		ParticleVariable<Matrix3> value;
		da->query(value, var, matl, patch, time);
		ParticleSubset* pset = value.getParticleSubset();
		if(pset->numParticles() > 0){
		  ParticleSubset::iterator iter = pset->begin();
		  for(;iter != pset->end(); iter++){
		    if (doAverage) {
		      stressVector.push_back(value[*iter]);
		    } else {
		      Matrix3 stress = value[*iter];
		      double sig11 = stress(0,0);
		      double sig12 = stress(0,1);
		      double sig13 = stress(0,2);
		      double sig22 = stress(1,1);
		      double sig23 = stress(1,2);
		      double sig33 = stress(2,2);
		      if (doEquiv) {
			double s12 = sig11 - sig22;
			double s23 = sig22 - sig33;
			double s31 = sig33 - sig11;
			double sigeff = sqrt(0.5*(s12*s12 + s23*s23 + s31*s31 + 
                                             6.0*(sig12*sig12 + sig23*sig23 + 
                                                  sig13*sig13)));
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << sigeff << endl; 
		      } else {
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << sig11 << " " << sig22 << " " << sig33 
			     << " " << sig23 << " " << sig13 << " " << sig12 
                             << endl;
		      }
		    }
		  }
		}
	      } // end of var compare if
	    } // end of material loop
	  } // end of ParticleVariable if
	} // end of variable loop
      } // end of patch loop
    } // end of level loop

    if (doAverage) {
      // Now that the volume vector and variable vector are available just
      // do a weighted average
      ASSERTEQ(volumeVector.size(), stressVector.size());
      Matrix3 avVar;
      for (unsigned int ii = 0; ii < volumeVector.size() ; ++ii) {
	avVar += ((stressVector[ii]*volumeVector[ii])/totVol);
      }
      for (int ii = 0; ii < 3; ++ii) {
	for (int jj = 0; jj < 3; ++jj) {
	  cout << avVar(ii,jj) << "  " ;
	}
	cout << endl;
      }
    }
  } // end of time step loop
}

////////////////////////////////////////////////////////////////////////////
//
// Print a particle variable
//
////////////////////////////////////////////////////////////////////////////
void printParticleVariable(DataArchive* da, 
			   string particleVariable,
                           long64 particleID,
			   unsigned long time_step_lower,
			   unsigned long time_step_upper){

  // Check if the particle variable is available
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  bool variableFound = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if (var == particleVariable) variableFound = true;
  }
  if (!variableFound) {
    cerr << "Variable " << particleVariable << " not found\n"; 
    exit(1);
  }

  // Now that the variable has been found, get the data for all 
  // available time steps // from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  //cout << "There are " << index.size() << " timesteps:\n";
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(time);

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patchIterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
	const Patch* patch = *iter;
        ++patchIndex; 

	// Loop thru all the variables 
	for(int v=0;v<(int)vars.size();v++){
	  std::string var = vars[v];
	  const Uintah::TypeDescription* td = types[v];
	  const Uintah::TypeDescription* subtype = td->getSubType();

	  // Check if the variable is a ParticleVariable
	  if(td->getType() == Uintah::TypeDescription::ParticleVariable) { 

	    // loop thru all the materials
	    ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	    ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
	    for(; matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;

	      // Find the name of the variable
	      if (var == particleVariable) {
		//cout << "Material: " << matl << endl;
		switch(subtype->getType()){
		case Uintah::TypeDescription::double_type:
		  {
		    ParticleVariable<double> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
                      if (particleID == 0) {
			for(;iter != pset->end(); iter++){
			  cout << time << " " << patchIndex << " " << matl; 
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter] << endl;
			}
                      } else {
			for(;iter != pset->end(); iter++){
			  if (particleID != pid[*iter]) continue;
			  cout << time << " " << patchIndex << " " << matl; 
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter] << endl;
			}
                      }
		    }
		  }
		break;
		case Uintah::TypeDescription::float_type:
		  {
		    ParticleVariable<float> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
                      if (particleID == 0) {
			for(;iter != pset->end(); iter++){
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter] << endl;
			}
                      } else {
			for(;iter != pset->end(); iter++){
			  if (particleID != pid[*iter]) continue;
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter] << endl;
			}
                      }
		    }
		  }
		break;
		case Uintah::TypeDescription::int_type:
		  {
		    ParticleVariable<int> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
                      if (particleID == 0) {
			for(;iter != pset->end(); iter++){
			  cout << time << " " << patchIndex << " " << matl;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter] << endl;
			}
                      } else {
			for(;iter != pset->end(); iter++){
			  if (particleID != pid[*iter]) continue;
			  cout << time << " " << patchIndex << " " << matl;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter] << endl;
			}
                      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Point:
		  {
		    ParticleVariable<Point> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
                      if (particleID == 0) {
			for(;iter != pset->end(); iter++){
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter](0) 
			       << " " << value[*iter](1)
			       << " " << value[*iter](2) << endl;
			}
                      } else {
			for(;iter != pset->end(); iter++){
			  if (particleID != pid[*iter]) continue;
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter](0) 
			       << " " << value[*iter](1)
			       << " " << value[*iter](2) << endl;
			}
		      }
                    }
		  }
		break;
		case Uintah::TypeDescription::Vector:
		  {
		    ParticleVariable<Vector> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
                      if (particleID == 0) {
			for(;iter != pset->end(); iter++){
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter][0] 
			       << " " << value[*iter][1]
			       << " " << value[*iter][2] << endl;
			}
                      } else {
			for(;iter != pset->end(); iter++){
			  if (particleID != pid[*iter]) continue;
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  cout << " " << value[*iter][0] 
			       << " " << value[*iter][1]
			       << " " << value[*iter][2] << endl;
			}
                      }
		    }
		  }
		break;
		case Uintah::TypeDescription::Matrix3:
		  {
		    ParticleVariable<Matrix3> value;
		    da->query(value, var, matl, patch, time);
		    ParticleVariable<long64> pid;
		    da->query(pid, "p.particleID", matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
                      if (particleID == 0) {
			for(;iter != pset->end(); iter++){
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  for (int ii = 0; ii < 3; ++ii) {
			    for (int jj = 0; jj < 3; ++jj) {
			      cout << " " << value[*iter](ii,jj) ;
			    }
			  }
			  cout << endl;
			}
                      } else {
			for(;iter != pset->end(); iter++){
			  if (particleID != pid[*iter]) continue;
			  cout << time << " " << patchIndex << " " << matl ;
			  cout << " " << pid[*iter];
			  for (int ii = 0; ii < 3; ++ii) {
			    for (int jj = 0; jj < 3; ++jj) {
			      cout << " " << value[*iter](ii,jj) ;
			    }
			  }
			  cout << endl;
			}
		      }
		    }
		  }
		break;
		case Uintah::TypeDescription::long64_type:
		  {
		    ParticleVariable<long64> value;
		    da->query(value, var, matl, patch, time);
		    ParticleSubset* pset = value.getParticleSubset();
		    if(pset->numParticles() > 0){
		      ParticleSubset::iterator iter = pset->begin();
		      for(;iter != pset->end(); iter++){
                        cout << time << " " << patchIndex << " " << matl ;
			cout << " " << value[*iter] << endl;
		      }
		    }
		  }
		break;
		default:
		  cerr << "Particle Variable of unknown type: " 
		       << subtype->getType() << endl;
		  break;
		}
	      } // end of var compare if
	    } // end of material loop
	  } // end of ParticleVariable if
	} // end of variable loop
      } // end of patch loop
    } // end of level loop
  } // end of time step loop
}

int main(int argc, char** argv)
{
  /*
   * Default values
   */
  bool do_partvar=false;
  bool do_partid=false;
  bool do_part_stress = false;
  bool do_part_strain = false;
  bool do_av_part_stress = false;
  bool do_av_part_strain = false;
  bool do_equiv_part_stress = false;
  bool do_true_part_strain = false;
  bool do_lagrange_part_strain = false;
  bool do_euler_part_strain = false;
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = 1;
  unsigned long time_step_inc = 1;
  bool tslow_set = false;
  bool tsup_set = false;
  bool tsinc_set = false;
  string filebase;
  string particleVariable;
  long64 particleID = 0;

  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-partvar"){
      do_partvar=true;
      particleVariable = argv[++i]; 
      if (particleVariable[0] == '-') 
        usage("-partvar <particle variable name>", argv[0]);
    } else if(s == "-partid"){
      do_partid=true;
      string id = argv[++i];
      if (id[0] == '-') 
        usage("-partid <particle id>", argv[0]);
      particleID = strtoll(argv[i],(char**)NULL,10); 
    } else if(s == "-part_stress"){
      do_part_stress = true;
      if (++i < argc) {
	s = argv[i];
	if (s == "avg") do_av_part_stress=true;
	else if (s == "equiv") do_equiv_part_stress=true;
	else if (s == "all") do_part_stress=true;
        else
	  usage("-part_stress [avg or equiv or all]", argv[0]);
      }
    } else if(s == "-part_strain"){
      do_part_strain = true;
      if (++i < argc) {
	s = argv[i];
	if (s == "avg") do_av_part_strain=true;
	else if (s == "true") do_true_part_strain=true;
	else if (s == "all") do_part_strain=true;
	else if (s == "lagrangian") do_lagrange_part_strain=true;
	else if (s == "eulerian") do_euler_part_strain=true;
        else
	  usage("-part_strain [avg / true / all / lagrangian / eulerian]", 
                argv[0]);
      }
    } else if (s == "-timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
      tslow_set = true;
    } else if (s == "-timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
      tsup_set = true;
    } else if (s == "-timestepinc") {
      time_step_inc = strtoul(argv[++i],(char**)NULL,10);
      tsinc_set = true;
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
    } 
  }
  filebase = argv[argc-1];

  if(filebase == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    DataArchive* da = scinew DataArchive(filebase);
    
    // Get the particle stresses
    if (do_part_stress) {
      if (do_av_part_stress) {
	cout << "\t Volume average stress = " << endl;
	getParticleStresses(da, "avg");
      } 
      else if (do_equiv_part_stress) getParticleStresses(da, "equiv");
      else getParticleStresses(da, "all");
    } 

    // Get the particle strains
    if (do_part_strain) {
      if (do_av_part_strain) {
	cout << "\t Volume average strain = " << endl;
	getParticleStrains(da, "avg");
      } 
      else if (do_true_part_strain) getParticleStrains(da, "true");
      else if (do_lagrange_part_strain) getParticleStrains(da, "lagrange");
      else if (do_euler_part_strain) getParticleStrains(da, "euler");
      else getParticleStrains(da, "all");
    } 

    // Print a particular particle variable
    if (do_partvar) {
      vector<int> index;
      vector<double> times;
      da->queryTimesteps(index, times);
      ASSERTEQ(index.size(), times.size());
      if (!tslow_set)
	time_step_lower =0;
      else if (time_step_lower >= times.size()) {
	cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
	abort();
      }
      if (!tsup_set)
	time_step_upper = times.size()-1;
      else if (time_step_upper >= times.size()) {
	cerr << "timestephigh must be between 0 and " << times.size()-1 << endl;
	abort();
      }
      printParticleVariable(da, particleVariable, particleID, time_step_lower, 
                            time_step_upper);
    }
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}
