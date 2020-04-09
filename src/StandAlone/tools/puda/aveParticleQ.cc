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

#include <StandAlone/tools/puda/aveParticleQ.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
//  Compute the mass weight averages for each timestep

void
Uintah::aveParticleQuanties( DataArchive      * da,
                             CommandLineFlags & clf )
{

  vector<const Uintah::TypeDescription*> types;
  vector<string> vars;
  da->queryVariables( vars, types );
  ASSERTEQ( vars.size(), types.size() );

  cout << "There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++){
    cout << vars[i] << ": " << types[i]->getName() << endl;
  }  

  //__________________________________
  //
  vector<int>     index;
  vector<double>  times;
  da->queryTimesteps( index, times );
  cout << "There are " << index.size() << " timesteps:\n";

 
  //__________________________________
  //  Bulletproofing  Do all the variables exist
  int n = 0;
  n += std::count( vars.begin(), vars.end(), "p.x" );
  n += std::count( vars.begin(), vars.end(), "p.mass" );
  n += std::count( vars.begin(), vars.end(), "p.velocity" );
  n += std::count( vars.begin(), vars.end(), "p.temperature" );
  n += std::count( vars.begin(), vars.end(), "p.stress" );
  n += std::count( vars.begin(), vars.end(), "p.localizedMPM" );

  if( n != 6 ){
    ostringstream err;
    cout << " n: " << n << endl;
    err<< "\n  ERROR: One of variables (p.x, p.mass, p.velocity, p.temperature, p.stress, p.localizedMPM) was not found in the uda\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }

  //__________________________________
  //  Open files and write out the headers
  // Two files per MPM material
  
  unsigned long timestep0 = 0;
  GridP grid    = da->queryGrid( timestep0 );
  LevelP level  = grid->getLevel( grid->numLevels()-1 );
  const Patch* patch0 = level->getPatch(0);
  ConsecutiveRangeSet matls = da->queryMaterials( "p.x", patch0, timestep0 );

  // maps for each matl
  std::map<int, std::ofstream> fileMap;    // non-failed particles
  std::map<int, std::ofstream> f_fileMap;  // failed particles

  for( auto m_iter = matls.begin(); m_iter != matls.end(); m_iter++ ){
    int matl = *m_iter;

    ofstream& strm = fileMap[matl];
    ostringstream fname0;
    fname0 << "aveParticleQ_" << matl << ".dat";
    strm.open( fname0.str() );
    
    ofstream& f_strm = f_fileMap[matl];
    ostringstream fname1;
    fname1 << "FailedAveParticleQ_" << matl << ".dat";
    f_strm.open( fname1.str() );

    strm.setf(ios::scientific,ios::floatfield);
    strm.precision(15);
    f_strm.setf(ios::scientific,ios::floatfield);
    f_strm.precision(15);
    
    strm   << "# uda: " << da->name() << "\n";
    f_strm << "# uda: " << da->name() << "\n";
    
    strm   << "# Material: " << matl << "\n";
    f_strm << "# Material: " << matl << "\n";    
    
    strm   << "# Stress: signed von Mises (equivalent stress)";
    f_strm << "# Stress: signed von Mises (equivalent stress)";
    
    strm   << "# This file contains averaged quantities of particles that have NOT failed.\n";
    f_strm << "# This file contains averaged quantities of particles that have failed.\n";

    strm   << "# time                 meanVel.x            meanVel.y             meanVel.z             meanMagVel            "
           << "  totalMass            avgTemperature       stress                KE                    nParticles" << endl;
           
    f_strm << "# time                 meanVel.x            meanVel.y             meanVel.z             meanMagVel            "
           << "  totalMass            avgTemperature       stress                KE                    nParticles" << endl;
  }

  
  //__________________________________
  //  Loop over timesteps
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);

  for(unsigned long t=clf.time_step_lower; t<=clf.time_step_upper; t+=clf.time_step_inc){
    double time = times[t];
    cout << "Working on timestep : " << t << " time: " << time << endl;

    GridP grid = da->queryGrid(t);
    LevelP level = grid->getLevel(grid->numLevels()-1);

    //__________________________________
    // loop over materials
    ConsecutiveRangeSet matls = da->queryMaterials( "p.x", patch0, t );

    for( auto m_iter = matls.begin(); m_iter != matls.end(); m_iter++ ){
      int matl = *m_iter;


      //__________________________________
      //  Initialized summed quantities
      Vector f_total_mom(0.,0.,0.);       // failed particles
      double f_total_mass   = 0.;
      double f_total_intE   = 0.;
      double f_total_stress =0.;
      double f_KE           = 0.;
      long int f_pCount     = 0.;


      Vector total_mom(0.,0.,0.);         // not failed
      double total_mass   = 0.;
      double total_intE   = 0.;
      double total_stress = 0.;
      double KE           = 0.;
      long int pCount     = 0.;


      //__________________________________
      //  Loop over patches
      for( auto iter = level->patchesBegin(); iter != level->patchesEnd(); iter++ ){
        const Patch* patch = *iter;

        //__________________________________
        //  retrieve variables
        ParticleVariable<Point>   pPos;
        ParticleVariable<Vector>  pVel;
        ParticleVariable<double>  pMass;
        ParticleVariable<double>  pTemp;
        ParticleVariable<Matrix3> pStress;
        ParticleVariable<int>     pLocalized;

        da->query( pPos,       "p.x",            matl, patch, t );
        da->query( pMass,      "p.mass",         matl, patch, t );
        da->query( pVel,       "p.velocity",     matl, patch, t );
        da->query( pStress,    "p.stress",       matl, patch, t );
        da->query( pTemp,      "p.temperature",  matl, patch, t );
        da->query( pLocalized, "p.localizedMPM", matl, patch, t );

        ParticleSubset* pset = pPos.getParticleSubset();

        if(pset->numParticles() > 0){

          //__________________________________
          //  Compute sums
          ParticleSubset::iterator piter = pset->begin();

          for(;piter != pset->end(); piter++){
            particleIndex idx = *piter;

            double mass       = pMass[idx];
            double vel_mag_sq = pVel[idx].length2();
            
            // signed von Mises Stress
            Matrix3 I;
            I.Identity();

            double sigMean      = pStress[idx].Trace()/3.0;
            double plusMinusOne = sigMean/abs( sigMean );

            Matrix3 sig_deviatoric   = pStress[idx] - I*sigMean;
            double sigEquivalent     = sqrt( (sig_deviatoric.NormSquared())*1.5 );
            double signedEquivStress = plusMinusOne * sigEquivalent;
            
            if (isnan( signedEquivStress ) ) {
              signedEquivStress = 0.;
            }

            
            //__________________________________
            //  
            if( pLocalized[idx] ){        // particle has failed.
              f_pCount        += 1;
              f_total_mass    += mass;
              f_total_mom     += mass * pVel[idx];
              f_total_intE    += mass * pTemp[idx];              //  cp is a constant so don't need to add it
              f_total_stress  += mass * signedEquivStress;
              f_KE            += 0.5 * mass * vel_mag_sq;
            }
            else{
              pCount         += 1;
              total_mass     += mass;
              total_mom      += mass * pVel[idx];
              total_intE     += mass * pTemp[idx];              //  cp is a constant so don't need to add it
              total_stress   += mass * signedEquivStress;
              KE             += 0.5 * mass * vel_mag_sq;
            }
          } // particle loop
        }  // if pset >0
      }  // patches


      //__________________________________
      //  Compute means
      Vector mean_vel   = total_mom/total_mass;
      double avg_temp   = total_intE/total_mass;
      double mag_vel    = mean_vel.length();
      double avg_stress = total_stress/total_mass;
      
      Vector f_mean_vel   = f_total_mom/f_total_mass;
      double f_avg_temp   = f_total_intE/f_total_mass;
      double f_mag_vel    = f_mean_vel.length();
      double f_avg_stress = f_total_stress/f_total_mass;

      
      if( isnan(f_mean_vel.length()) || f_total_mass == 0){
        f_mean_vel   = Vector(0,0,0);
        f_mag_vel    = 0;
        f_avg_temp   = 0;
        f_avg_stress = 0;
      }
      
      
      fileMap[matl]   << time << " " <<   mean_vel.x() << " " <<   mean_vel.y() << " " <<   mean_vel.z() << " " <<   mag_vel << " " <<   total_mass   << " " <<   avg_temp   << " " << avg_stress   << " " <<   KE   << " " << pCount   << endl;
      f_fileMap[matl] << time << " " << f_mean_vel.x() << " " << f_mean_vel.y() << " " << f_mean_vel.z() << " " << f_mag_vel << " " <<   f_total_mass << " " <<   f_avg_temp << " " << f_avg_stress << " " <<   f_KE << " " << f_pCount << endl;
    }  //  matls
  }  // time
  
  //__________________________________
  //  flush and close files
  for( auto m_iter = matls.begin(); m_iter != matls.end(); m_iter++ ){
    int matl = *m_iter;
    fileMap[matl].flush();
    fileMap[matl].close();
  
    f_fileMap[matl].flush();
    f_fileMap[matl].close();  
  }
  
} // end

