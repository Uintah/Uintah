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


#include <StandAlone/tools/puda/pStressHistogram.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;
#define nBins  100

////////////////////////////////////////////////////////////////////////
//  Compute a historgram of particle stress

// If this particle has failed return true.  If pLocalized doesn't exist
// then always return true 
bool pNotFailed( bool  pLocalExists,
                const ParticleVariable<int> &  pLocalized,
                const particleIndex         & idx )
{
  if( pLocalExists  ){
    if( pLocalized[idx] == 0 ){
      return true;
    }
    return false;
  }
  // this uda doesn't contain pLocalized
  return true;
}
//______________________________________________________________________
//
void
Uintah::pStressHistogram( DataArchive      * da, 
                          CommandLineFlags & clf )
{
  vector<string> vars;
  vector<int> notUsed;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, notUsed, types);
  
  cout << "There are " << vars.size() << " variables:\n";
  for(int i=0;i<(int)vars.size();i++) {
    cout << vars[i] << ": " << types[i]->getName() << endl;
  }
  
  //__________________________________
  //  Bulletproofing  Do all the variables exist
  int n = 0;
  n += std::count( vars.begin(), vars.end(), "p.x" );
  n += std::count( vars.begin(), vars.end(), "p.stress" );
  
  if( n != 2 ){
    ostringstream err;
    err<< "\n  ERROR: One of variables (p.x, p.stress) was not found in the uda\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }
  
  // is p.localized in the uda
  bool pLocalExists = ( std::count( vars.begin(), vars.end(), "p.localizedMPM" ) );

  // output when a timestep was saved.
  vector<int>    index;
  vector<double> times;
  da->queryTimesteps(index, times);
  
  cout << "There are " << index.size() << " timesteps:\n";
  
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << i << "  timestep: " << index[i] << ": " << times[i] << endl;
  }
          
  //__________________________________
  //  Timestep loop    
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
  
  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
  
    bool noErrors = true;
    
    double time = times[t];
    cout << "Working on time = " << time << endl;
    
    // create file stream and write out the header
    int matl = clf.matl;
    ostringstream filename;
    filename << "histogram." << setw(4) << setfill('0') << t/clf.time_step_inc << ".dat";
    ofstream out_strm( filename.str() );
    
    out_strm   << "# uda: " << da->name() << endl;
    out_strm   << "# Material: " << matl << endl;
    out_strm   << "# Timestep: " << t << " time: " << time << endl;
    out_strm.flush();
    
    // initialize historgram
    double stress_bins[nBins];
    int num_in_bin[nBins];
    
    for(int i=0;i<nBins;i++){
      num_in_bin[i]=0;
    }

    //__________________________________
    //  Level loop
    GridP grid = da->queryGrid(t);
  
    for(int l=0;l<grid->numLevels();l++){
    
      LevelP level = grid->getLevel(l);
      double min_eq_stress =  DBL_MAX;
      double max_eq_stress = -DBL_MAX;
      
      //__________________________________
      //  Patch loop
      for( auto iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){
          
        const Patch* patch = *iter;
        
        //__________________________________
        //   find min & max equivalent stresses
        ParticleVariable<Point>   pX;
        ParticleVariable<Matrix3> pStress;
        ParticleVariable<int>     pLocalized;
        
        da->query( pX,      "p.x",      matl, patch, t );
        da->query( pStress, "p.stress", matl, patch, t );
        
        if( pLocalExists ) {
          da->query( pLocalized, "p.localizedMPM", matl, patch, t );
        }
        
        ParticleSubset* pset = pX.getParticleSubset();
        
        if(pset->numParticles() > 0){
        
          for(auto iter = pset->begin() ;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            
            if( pNotFailed( pLocalExists, pLocalized, idx) ){
              Matrix3 one; 
              one.Identity();
            
              Matrix3 Mdev     = pStress[idx] - one*(pStress[idx].Trace()/3.0);
              double eq_stress = sqrt(Mdev.NormSquared()*1.5);
            
              min_eq_stress = min( min_eq_stress, eq_stress );
              max_eq_stress = max( max_eq_stress, eq_stress );
            }
          }
        }
      }

      //__________________________________
      //
      // inialize the bins
      
      if( abs(max_eq_stress - min_eq_stress) < DBL_EPSILON ){
        cout << "\n  ERROR: the max_eq_stress == min_eq_stress.   Ignoring this timestep \n\n";
        noErrors = false;
        continue;
      } 
      
      double stress_interval = (max_eq_stress - min_eq_stress)/(double)nBins + 1.0;  // the 1.0 prevents exceeding array bounds
      
      for(int i=0;i<nBins;i++){
        stress_bins[i] = min_eq_stress + ((double) i) * stress_interval;
      }

      //__________________________________
      //  Patch loop
      for(auto iter = level->patchesBegin(); iter != level->patchesEnd(); iter++){
          
        const Patch* patch = *iter;
        
        //__________________________________
        //   histogram bins
        ParticleVariable<Point>   pX;
        ParticleVariable<Matrix3> pStress;
        ParticleVariable<int>     pLocalized;
        
        da->query( pX,         "p.x",            matl, patch, t );
        da->query( pStress,    "p.stress",       matl, patch, t );
        
        if( pLocalExists ) {
          da->query( pLocalized, "p.localizedMPM", matl, patch, t );
        }
        
        
        ParticleSubset* pset = pX.getParticleSubset();
        
        if(pset->numParticles() > 0){
          
          for(auto iter = pset->begin() ;iter != pset->end(); iter++){
            particleIndex idx = *iter;
           
            if( pNotFailed( pLocalExists, pLocalized, idx) ){
            
              Matrix3 one; 
              one.Identity();

              Matrix3 Mdev     = pStress[idx] - one*(pStress[idx].Trace()/3.0);
              double eq_stress = sqrt(Mdev.NormSquared()*1.5);

              int bin = (int)((eq_stress - min_eq_stress)/(stress_interval + DBL_EPSILON));

              // bulletproofing
              if( bin < 0 || bin >= nBins){
                cout << "\n  ERROR: this particle exceeds the histogram bin range (" << bin << ").  Now clamping it to either 0 or nBins.\n\n";
                bin = min(bin, nBins);
                bin = max(bin, 0);
              }

              num_in_bin[bin]++;
            }
          }
        }
      }  
    }   // for levels
    
    //__________________________________
    //  Write to a file
    if( noErrors && out_strm.is_open() ){
      for(int i=0;i<nBins;i++){
        out_strm << stress_bins[i] << " " << num_in_bin[i] << endl;
      }
    }
    out_strm.close();
  }
} // end jim3()
