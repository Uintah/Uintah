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
 *  partextract.cc: Print out a uintah data archive for particle data
 *
 *  Written by:
 *   Biswajit Banerjee
 *   July 2004
 *   Rewritten extensively by Jim Guilkey 
 *   October 2018
 *   Now, by specifying multiple -partvar arguments, user can extract multiple
 *   particle variables into a single line
 *   e.g.; > partextract -mat 0 -partvar p.damage -partvar p.velocity -timestep 2 -include_position_output MyData.uda.000
 *   output now includes a header line descibing what variables have been output
 *
 */

#if defined( __PGI )
   // pgCC version 7.1-2 does not define strtoll (in stdlib.h or
   // anywhere)... However, this seems to fake the compiler into
   // not complaining.
#  define _ISOC99_SOURCE
#endif


#include <Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
//#include <Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/SymmMatrix3.h>
#include <Core/OS/Dir.h>
#include <Core/Parallel/Parallel.h>
#include <Core/DataArchive/DataArchive.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h> // for strtoll

using namespace std;
using namespace Uintah;

// declarations
void usage(const std::string& badarg, const std::string& progname);
void getParticleStrains(DataArchive* da, int mat, long64 particleID, 
		        string flag, unsigned long time_step_lower,
                                     unsigned long time_step_upper);
void getParticleStresses(DataArchive* da, int mat, long64 particleID, 
		         string flag, unsigned long time_step_lower,
                                      unsigned long time_step_upper,
                                      bool include_position_output);
void printParticleVariables(DataArchive* da, int mat, 
                            vector<string> particleVariable,
                            long64 particleID, unsigned long time_step_lower,
                            unsigned long time_step_upper,
                            unsigned long time_step_inc,
                            bool include_position_output);
void computeEquivStress(const Matrix3& sig, double& sigeqv);
void computePressure(const Matrix3& sig, double& press);
void computeEquivStrain(const Matrix3& F, double& epseqv);
void computeTrueStrain(const Matrix3& F, Vector& strain);
void computeGreenLagrangeStrain(const Matrix3& F, Matrix3& E);
void computeGreenAlmansiStrain(const Matrix3& F, Matrix3& e);
void computeStretchRotation(const Matrix3& F, Matrix3& R, Matrix3& U);
void printCauchyStress(const Matrix3& stress);

int
main( int argc, char** argv )
{
  Uintah::Parallel::initializeManager(argc, argv);

  /*
   * Default values
   */
  int mat = -1;
  bool do_partvar=false;
//  bool do_partid=false;
  bool do_part_stress = false;
  bool do_part_pressure = false;
  bool do_part_strain = false;
  bool do_av_part_stress = false;
  bool do_av_part_strain = false;
  bool do_equiv_part_stress = false;
  bool do_equiv_part_strain = false;
  bool do_true_part_strain = false;
  bool do_lagrange_part_strain = false;
  bool do_euler_part_strain = false;
  bool include_position_output = false;
  unsigned long time_step_lower = 0;
  unsigned long time_step_upper = 1;
  unsigned long time_step_inc = 1;
  bool tslow_set = false;
  bool tsup_set = false;
//  bool tsinc_set = false;
  string filebase;
  vector<string> particleVariable;
  long64 particleID = 0;

  // set defaults for cout
  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(8);
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-mat"){
      mat = atoi(argv[++i]);
    } else if(s == "-partvar"){
      do_partvar=true;
//      particleVariable = argv[++i]; 
      particleVariable.push_back(argv[++i]);
      if (particleVariable[particleVariable.size()-1][0] == '-') 
        usage("-partvar <particle variable name>", argv[0]);
    } else if(s == "-partid"){
//      do_partid=true;
      string id = argv[++i];
      if (id[0] == '-') 
        usage("-partid <particle id>", argv[0]);
      particleID = strtoll(argv[i],(char**)nullptr,10); 
    } else if(s == "-part_stress"){
      do_part_stress = true;
      if (++i < argc) {
        s = argv[i];
        if (s == "avg") do_av_part_stress=true;
        else if (s == "equiv") do_equiv_part_stress=true;
        else if (s == "all") do_part_stress=true;
        else if (s == "pressure") do_part_pressure=true;
        else
          usage("-part_stress [avg or equiv or pressure or all]", argv[0]);
      }
    } else if(s == "-part_strain"){
      do_part_strain = true;
      if (++i < argc) {
        s = argv[i];
        if (s == "avg") do_av_part_strain=true;
        else if (s == "true") do_true_part_strain=true;
        else if (s == "equiv") do_equiv_part_strain=true;
        else if (s == "all") do_part_strain=true;
        else if (s == "lagrangian") do_lagrange_part_strain=true;
        else if (s == "eulerian") do_euler_part_strain=true;
        else
          usage("-part_strain [avg / true / all / lagrangian / eulerian]", 
                argv[0]);
      }
    } else if (s == "-timestep") {
      time_step_lower = strtoul(argv[++i],(char**)nullptr,10);
      time_step_upper = time_step_lower;
      tslow_set = true;
      tsup_set = true;
    } else if (s == "-timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)nullptr,10);
      tslow_set = true;
    } else if (s == "-timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)nullptr,10);
      tsup_set = true;
    } else if (s == "-timestepinc") {
      time_step_inc = strtoul(argv[++i],(char**)nullptr,10);
    } else if (s == "-include_position_output") {
      include_position_output = true;
    } 
  }
  filebase = argv[argc-1];

  if(filebase == "" || filebase == argv[0]){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    DataArchive* da = scinew DataArchive(filebase);
 
    // Recover time data
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
      cerr << "timestephigh must be between 0 and " << times.size()-1 <<endl;
      abort();
    }

    // Get the particle stresses
    if (do_part_stress) {
      if (do_av_part_stress) {
        cout << "\t Volume average stress = " << endl;
        getParticleStresses(da, mat, particleID,"avg",  time_step_lower, time_step_upper,
                            include_position_output);
      } else if (do_equiv_part_stress) {
        getParticleStresses(da, mat, particleID,"equiv",time_step_lower, time_step_upper,
                            include_position_output);
      } else if (do_part_pressure) {
        getParticleStresses(da, mat, particleID,"pressure",time_step_lower, time_step_upper,
                            include_position_output);
      } else {
        getParticleStresses(da, mat, particleID,"all",  time_step_lower, time_step_upper,
                            include_position_output);
      }
    } 

    // Get the particle strains
    if (do_part_strain) {
      if (do_av_part_strain) {
        cout << "\t Volume average strain = " << endl;
        getParticleStrains(da,mat, particleID,"avg",     time_step_lower,time_step_upper);
      } else if (do_true_part_strain) {
        getParticleStrains(da,mat, particleID,"true",    time_step_lower,time_step_upper);
      } else if (do_equiv_part_strain) {
        getParticleStrains(da,mat, particleID,"equiv",   time_step_lower,time_step_upper);
      } else if (do_lagrange_part_strain) {
        getParticleStrains(da,mat, particleID,"lagrange",time_step_lower,time_step_upper);
      } else if (do_euler_part_strain) {
        getParticleStrains(da,mat, particleID,"euler",   time_step_lower,time_step_upper);
      } else {
        getParticleStrains(da,mat, particleID,"all",     time_step_lower,time_step_upper);
      }
    } 

    // Print a particular particle variable
    if (do_partvar) {
      printParticleVariables(da, mat, particleVariable, particleID, 
                             time_step_lower, time_step_upper, time_step_inc,
                             include_position_output);
    }
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }
}

void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "") cerr << "Error parsing argument: " << badarg << endl;

  cerr << "Usage: " << progname << " [options] <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -mat <material id>\n";
  cerr << "  -partvar <variable name>\n";
  cerr << "  -partid <particleid>\n";
  cerr << "  -part_stress [avg or equiv or pressure or all]\n";
  cerr << "  -part_strain [avg/true/equiv/all/lagrangian/eulerian]\n", 
  cerr << "  -timestep [int] (only outputs data for timestep int)\n";
  cerr << "  -timesteplow [int] (only outputs timestep from int)\n";
  cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
  cerr << "  -include_position_output (add particle position before other data output)\n";
  cerr << "USAGE IS NOT FINISHED\n\n";
  exit(1);
}

////////////////////////////////////////////////////////////////////////////
//
// Get particle strains (avg, true or all)
//
////////////////////////////////////////////////////////////////////////////
void 
getParticleStrains(DataArchive* da, int mat, long64 particleID, string flag,
                   unsigned long time_step_lower, unsigned long time_step_upper) {

  // Parse the flag and check which option is needed
  bool doAverage = false;
  bool doTrue = false;
  bool doLagrange = false;
  bool doEuler = false;
  bool doEquiv = false;
  //   bool doAll = false;
  if (flag == "avg") doAverage = true;
  else if (flag == "equiv") doEquiv = true;
  else if (flag == "true") doTrue = true;
  else if (flag == "lagrange") doLagrange = true;
  else if (flag == "euler") doEuler = true;
  //   else doAll = true;

  // Check if all the required variables are there .. for all cases
  // we need p.deformationMeasure and for the volume average we need p.volume
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables( vars, num_matls, types );
  ASSERTEQ( vars.size(), types.size() );

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
//  cerr << "There are " << index.size() << " timesteps:\n";
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
    double time = times[t];
    //cerr << "Time = " << time << endl;
    GridP grid = da->queryGrid(t);

    vector<double> volumeVector;
    vector<Matrix3> deformVector;
    double totVol = 0.0;

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patch_iterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        ++patchIndex; 

        // Loop thru all the variables 
        for(int v=0;v<(int)vars.size();v++){
          std::string var = vars[v];

          if (var != "p.volume" && var != "p.deformationMeasure") continue;
          const Uintah::TypeDescription* td = types[v];

          // Check if the variable is a ParticleVariable
          if(td->getType() == Uintah::TypeDescription::ParticleVariable) { 

            // loop thru all the materials
            ConsecutiveRangeSet matls = da->queryMaterials(var, patch, t);
            ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
            for(; matlIter != matls.end(); matlIter++){
              int matl = *matlIter;

              if (mat != -1 && matl != mat) continue;

              // Find the name of the variable
              if (doAverage) {
                cerr << "Finding volume average strains ... " << endl;
                if (var == "p.volume") {
                  switch(td->getSubType()->getType()) {
		  case Uintah::TypeDescription::double_type:
		    {
		      ParticleVariable<double> value;
		      da->query(value, var, matl, patch, t);
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
		      da->query(value, var, matl, patch, t);
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
                } else if (var == "p.deformationMeasure") {
                  //cerr << "Material: " << matl << endl;
                  ParticleVariable<Matrix3> value;
                  da->query(value, var, matl, patch, t);
                  ParticleSubset* pset = value.getParticleSubset();
                  if(pset->numParticles() > 0){
                    ParticleSubset::iterator iter = pset->begin();
                    for(;iter != pset->end(); iter++){
                      particleIndex idx = *iter;

                      // Find the right stretch tensor
                      Matrix3 deformGrad = value[idx];
                      Matrix3 stretch(0.0), rotation(0.0);
                      deformGrad.polarDecomposition(stretch, rotation, 
                                                    1.0e-10, true);

                      cout.setf(ios::scientific,ios::floatfield);
                      cout.precision(6);

                      deformVector.push_back(stretch);
                    } 
                  } // end of pset > 0
                } // end of var if
                continue;
              } // end of doAverage

              // If not an average calculation
              if (var == "p.deformationMeasure") {
                //cerr << "Material: " << matl << endl;
                ParticleVariable<Matrix3> value;
                da->query(value, var, matl, patch, t);
                ParticleVariable<long64> pid;
                da->query(pid, "p.particleID", matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();
                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
                  if (particleID == 0) {
                    for(;iter != pset->end(); iter++){
                      particleIndex idx = *iter;

                      // Get the deformation gradient
                      Matrix3 F = value[idx];
                      cout.setf(ios::scientific,ios::floatfield);
                      cout.precision(6);

                      cout << time << " " << pid[idx] << " " << patchIndex 
                           << " " << matl ;
                      if (doTrue) {
                        Vector trueStrain(0.0);
                        computeTrueStrain(F, trueStrain);
                      } else if (doEquiv) {
                        double equiv_strain = 0.0;
                        computeEquivStrain(F, equiv_strain);
                      } else if (doLagrange) {
                        Matrix3 E(0.0);
                        computeGreenLagrangeStrain(F,E);
                      } else if (doEuler) {
                        Matrix3 e(0.0);
                        computeGreenAlmansiStrain(F,e);
                      } else {
                        Matrix3 U(0.0), R(0.0);
                        computeStretchRotation(F,R,U);
                      }
                    }
                  } else {
                    for(;iter != pset->end(); iter++){
                      particleIndex idx = *iter;

                      if (particleID != pid[*iter]) continue;

                      // Get the deformation gradient
                      Matrix3 F = value[idx];
                      cout.setf(ios::scientific,ios::floatfield);
                      cout.precision(6);

                      cout << time << " " << pid[idx] << " " << patchIndex 
                           << " " << matl ;
                      if (doTrue) {
                        Vector trueStrain(0.0);
                        computeTrueStrain(F, trueStrain);
                      } else if (doEquiv) {
                        double equiv_strain = 0.0;
                        computeEquivStrain(F, equiv_strain);
                      } else if (doLagrange) {
                        Matrix3 E(0.0);
                        computeGreenLagrangeStrain(F,E);
                      } else if (doEuler) {
                        Matrix3 e(0.0);
                        computeGreenAlmansiStrain(F,e);
                      } else {
                        Matrix3 U(0.0), R(0.0);
                        computeStretchRotation(F,R,U);
                      }
                      break;
                    } // end of pset iter loop
                  } // end of particleId = 0 if
                } // end of numparticle > 0 if
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
getParticleStresses(DataArchive* da, int mat, long64 particleID, string flag,
                    unsigned long time_step_lower, unsigned long time_step_upper,
                    bool include_position_output) {

  // Parse the flag and check which option is needed
  bool doAverage = false;
  bool doEquiv = false;
  bool doPress = false;
  //   bool doAll = false;
  if (flag == "avg"){
    doAverage = true;
  } else if (flag == "equiv"){
    doEquiv = true;
  } else if (flag == "pressure"){
    doPress = true;
  }
  //   else doAll = true;

  // Check if all the required variables are there .. for all cases
  // we need p.stress and for the volume average we need p.volume
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables( vars, num_matls, types );
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
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t++){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(t);

    vector<double> volumeVector;
    vector<Matrix3> stressVector;
    vector<Point> posVector;
    double totVol = 0.0;

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patch_iterator iter = level->patchesBegin(); 
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
            ConsecutiveRangeSet matls = da->queryMaterials(var, patch, t);
            ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
            for(;matlIter != matls.end(); matlIter++){
              int matl = *matlIter;

              if (mat != -1 && matl != mat) continue;

              // Find the name of the variable
              if (doAverage) {
                if (var == "p.volume") {
                  switch(td->getSubType()->getType()) {
                    case Uintah::TypeDescription::double_type:
                      {
                        ParticleVariable<double> value;
                        da->query(value, var, matl, patch, t);
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
                        da->query(value, var, matl, patch, t);
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
                } else if (var == "p.stress") {
                  ParticleVariable<Matrix3> value;
                  da->query(value, var, matl, patch, t);
                  ParticleSubset* pset = value.getParticleSubset();
                  if(pset->numParticles() > 0){
                    ParticleSubset::iterator iter = pset->begin();
                    for(;iter != pset->end(); iter++){
                      stressVector.push_back(value[*iter]);
                    } 
                  }
                }
                continue;
              } 

              // If not a volume averaged quantity
              if (var == "p.stress") {
                ParticleVariable<Matrix3> value;
                da->query(value, var, matl, patch, t);
                ParticleVariable<long64> pid;
                da->query(pid, "p.particleID", matl, patch, t);
                ParticleSubset* pset = value.getParticleSubset();


                ParticleVariable<Point> value_pos;
                da->query(value_pos, "p.x", matl, patch, t);

                if(pset->numParticles() > 0){
                  ParticleSubset::iterator iter = pset->begin();
                  for(;iter != pset->end(); iter++){
                    if (particleID == 0) {
                      Matrix3 stress = value[*iter];

                      cout.setf(ios::scientific,ios::floatfield);
                      cout.precision(6);

                      cout << time << " " << pid[*iter] << " " << patchIndex 
                           << " " << matl ;
                      if(include_position_output){
                        cout << " " << value_pos[*iter].x()
                             << " " << value_pos[*iter].y()
                             << " " << value_pos[*iter].z() ;
                      }

                      if (doEquiv) {
                        double sigeff = 0.0;
                        computeEquivStress(stress, sigeff);
                      } else if(doPress) {
                        double press = 0.0;
                        computePressure(stress, press);
                      } else {
                        printCauchyStress(stress);
                      }
                    } else {
                      Matrix3 stress = value[*iter];

                      if (particleID != pid[*iter]) continue;

                      cout.setf(ios::scientific,ios::floatfield);
                      cout.precision(6);

                      cout << time << " " << pid[*iter] << " " << patchIndex 
                           << " " << matl ;
                      if(include_position_output){
                        cout << " " << value_pos[*iter].x()
                             << " " << value_pos[*iter].y()
                             << " " << value_pos[*iter].z() ;
                      }

                      if (doEquiv) {
                        double sigeff = 0.0;
                        computeEquivStress(stress, sigeff);
                      } else if(doPress) {
                        double press = 0.0;
                        computePressure(stress, press);
                      } else {
                        printCauchyStress(stress);
                      }
                      break;
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
void printParticleVariables(DataArchive* da, 
                            int mat,
                            vector<string> particleVariable,
                            long64 particleID,
                            unsigned long time_step_lower,
                            unsigned long time_step_upper,
                            unsigned long time_step_inc,
                            bool include_position_output)
{
  // Check if the particle variable is available
  vector<string> vars;
  vector<int> num_matls;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables( vars, num_matls, types );
  ASSERTEQ(vars.size(), types.size());

  // Check to see if p.particleID is available
  bool have_partIDs = false;
  for(unsigned int v=0;v<vars.size();v++){
    std::string var = vars[v];
    if(var == "p.particleID"){
      have_partIDs = true;
    }
  }

  // Make sure other variables requested are in the saved data
  for(unsigned int pv=0;pv<particleVariable.size();pv++){
    bool variableFound = false;
    for(unsigned int v=0;v<vars.size();v++){
      std::string var = vars[v];
      const Uintah::TypeDescription* td = types[v];
      if(var == particleVariable[pv]){
        variableFound = true;
        if(td->getType() != Uintah::TypeDescription::ParticleVariable) {
           cerr << "partextract is only for particle variables" << endl;
           exit(1);
        } // end of ParticleVariable if
      }
    } // loop over list of all variables
    if (!variableFound) {
      cerr << "Variable " << particleVariable[pv] << " not found\n"; 
      exit(1);
    }
  } // loop over particleVariables

  // Print out a header line
  if(have_partIDs){
    cout << "% Time patchIndex material particleID";
  } else {
    cout << "% Time patchIndex material";
  }
  if(include_position_output){
    cout << " position";
  }
  for(unsigned int pv=0;pv<particleVariable.size();pv++){
    cout << " " << particleVariable[pv];
  }
  cout << endl;

  // Now that the variables have been found, get the data for all 
  // available time steps from the data archive
  vector<int> index;
  vector<double> times;
  da->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  //cout << "There are " << index.size() << " timesteps:\n";

  int matl = mat;
      
  // Loop thru all time steps and store the volume and variable (stress/strain)
  for(unsigned long t=time_step_lower;t<=time_step_upper;t+=time_step_inc){
    double time = times[t];
    //cout << "Time = " << time << endl;
    GridP grid = da->queryGrid(t);

    // Loop thru all the levels
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);

      // Loop thru all the patches
      Level::const_patch_iterator iter = level->patchesBegin(); 
      int patchIndex = 0;
      for(; iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        ++patchIndex; 

        // Find the name of the variable
        ParticleVariable<Point> pos;
        da->query(pos, "p.x", matl, patch, t);
        ParticleVariable<long64> pid;
        if(have_partIDs){
          da->query(pid, "p.particleID", matl, patch, t);
        }
        ParticleSubset* pset = pos.getParticleSubset();
        if(pset->numParticles() > 0){
         ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            if (particleID == 0 || particleID == pid[*iter]) {
              cout << time << " " << patchIndex << " " << matl; 
              if(have_partIDs){
                cout << " " << pid[*iter];
              }
              if(include_position_output){
                cout << " " << pos[*iter].x()
                     << " " << pos[*iter].y()
                     << " " << pos[*iter].z() ;
              }
              // Loop over all the requested particle variables
              for(unsigned int pv=0;pv<particleVariable.size();pv++){
                // Loop thru all the variables 
                for(int v=0;v<(int)vars.size();v++){
                  std::string var = vars[v];
                  const Uintah::TypeDescription* td = types[v];
                  const Uintah::TypeDescription* subtype = td->getSubType();
 
                  if (var == particleVariable[pv]) {
                    switch(subtype->getType()){
                      case Uintah::TypeDescription::double_type:
                        {
                          ParticleVariable<double> value;
                          da->query(value, var, matl, patch, t);
                          cout << " " << value[*iter]; 
                        }
                      break;
                      case Uintah::TypeDescription::float_type:
                        {
                          ParticleVariable<float> value;
                          da->query(value, var, matl, patch, t);
                          cout << " " << value[*iter]; 
                        }
                      break;
                      case Uintah::TypeDescription::int_type:
                        {
                          ParticleVariable<int> value;
                          da->query(value, var, matl, patch, t);
                          cout << " " << value[*iter]; 
                        }
                      break;
                      case Uintah::TypeDescription::Point:
                        {
                          ParticleVariable<Point> value;
                          ParticleSubset::iterator iter = pset->begin();
                              cout << " " << value[*iter](0) 
                                   << " " << value[*iter](1)
                                   << " " << value[*iter](2) << " ";
                        }
                      break;
                      case Uintah::TypeDescription::Vector:
                       {
                         ParticleVariable<Vector> value;
                         da->query(value, var, matl, patch, t);
                         cout << " " << value[*iter][0] 
                              << " " << value[*iter][1]
                              << " " << value[*iter][2] << " ";
                       }
                      break;
                      case Uintah::TypeDescription::Matrix3:
                       {
                         ParticleVariable<Matrix3> value;
                         da->query(value, var, matl, patch, t);
                         for (int ii = 0; ii < 3; ++ii) {
                           for (int jj = 0; jj < 3; ++jj) {
                             cout << " " << value[*iter](ii,jj) ;
                           }
                         }
                         cout << " ";
                       }
                      break;
                      case Uintah::TypeDescription::long64_type:
                       {
                         ParticleVariable<long64> value;
                         da->query(value, var, matl, patch, t);
                         cout << " " << value[*iter] << " ";
                       }
                      break;
                      default:
                        cerr << "Particle Variable of unknown type: " 
                             << subtype->getType() << endl;
                      break;
                    } // switch
                  } // end of var compare if
                } // end of variable loop
              } // end of loop over particleVariables
            } // if all particleIDs or this particular particleID
            cout << endl;
          } // end of loop over particles
        } // if pset not empty
      } // end of patch loop
    } // end of level loop
  } // end of time step loop
}

void computeEquivStress(const Matrix3& stress, double& sigeff)
{
  Matrix3 I; I.Identity();
  Matrix3 s = stress - I*stress.Trace()/3.0;
  sigeff = sqrt(1.5*s.NormSquared());

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  cout << " " << sigeff << endl;
}

void computePressure(const Matrix3& stress, double& press)
{
  press = (-1./3.)*stress.Trace();

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  cout << " " << press << endl;
}

void computeEquivStrain(const Matrix3& F, double& equiv_strain)
{
  Matrix3 I; I.Identity();
  Matrix3 E = (F.Transpose()*F-I)*0.5;
  equiv_strain = sqrt(E.NormSquared()/1.5);

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  cout << " " << equiv_strain ;
  cout << endl;
}

void computeTrueStrain(const Matrix3& F, Vector& strain)
{
  // Compute the left Cauchy-Green tensor
  Matrix3 C = F.Transpose()*F;

  // Find the eigenvalues of C
  Vector Lambda; Matrix3 direction;
  C.eigen(Lambda, direction);

  // Find the stretches
  Vector lambda;
  for (int ii =0; ii < 3; ++ii) {
    lambda[ii] = sqrt(Lambda[ii]);
    strain[ii] = log(lambda[ii]);
  }

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  for (int ii = 0; ii < 3; ++ii) {
    cout << " " << strain[ii] ;
  }
  cout << endl;
}

void computeGreenLagrangeStrain(const Matrix3& F, Matrix3& E)
{
  Matrix3 I; I.Identity();
  Matrix3 C = F.Transpose()*F;
  E = (C - I)*0.5;

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  for (int ii = 0; ii < 3; ++ii) cout << " " << E(ii,ii);
  cout << " " << E(1,2); cout << " " << E(2,0);
  cout << " " << E(0,1);
  cout << endl;
}

// Calculate the Almansi-Hamel strain tensor
void computeGreenAlmansiStrain(const Matrix3& F, Matrix3& e)
{
  Matrix3 I; I.Identity();
  Matrix3 b = F*F.Transpose();
  Matrix3 binv = b.Inverse();
  e = (I - binv)*0.5;

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  for (int ii = 0; ii < 3; ++ii) cout << " " << e(ii,ii);
  cout << " " << e(1,2); cout << " " << e(2,0);
  cout << " " << e(0,1);
  cout << endl;
}

void computeStretchRotation(const Matrix3& F, Matrix3& R, Matrix3& U)
{
  F.polarDecomposition(U, R, 1.0e-10, true);

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      cout << " " << F(ii,jj);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      cout << " " << U(ii,jj);
    }
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      cout << " " << R(ii,jj);
    }
  }
  cout << endl;
}

void printCauchyStress(const Matrix3& stress)
{
  double sig11 = stress(0,0);
  double sig12 = stress(0,1);
  double sig13 = stress(0,2);
  double sig22 = stress(1,1);
  double sig23 = stress(1,2);
  double sig33 = stress(2,2);

  cout.setf(ios::scientific,ios::floatfield);
  cout.precision(6);

  cout << " " << sig11 << " " << sig22 << " " << sig33 
       << " " << sig23 << " " << sig13 << " " << sig12 
       << endl;
}
