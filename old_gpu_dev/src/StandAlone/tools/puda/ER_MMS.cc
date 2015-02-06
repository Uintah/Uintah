
#include <StandAlone/tools/puda/ER_MMS.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              ER_MMS   O P T I O N
// Compares the exact solution for the Expanding Ring MMS problem with
// the MPM solution.  Computes the L-infinity error on the displacement
// at each timestep and reports to the command line.

void
Uintah::ER_MMS( DataArchive * da, CommandLineFlags & clf )
{
  vector<string> vars;
  vector<const Uintah::TypeDescription*> types;
  da->queryVariables(vars, types);
  ASSERTEQ(vars.size(), types.size());
  
  cout << "There are " << vars.size() << " variables:\n";
  
  for(int i=0;i<(int)vars.size();i++)
    cout << vars[i] << ": " << types[i]->getName() << endl;
      
  vector<int> index;
  vector<double> times;
  
  da->queryTimesteps(index, times);
  
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << index[i] << ": " << times[i] << endl;
  }
      
  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, clf.time_step_lower, clf.time_step_upper);
      
  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
    double time = times[t];
    GridP grid = da->queryGrid(t);

    //__________________________________
    //  hard coded constants!!!!
//    double mu    = 5000000;
//    double bulk  = 3333333.33;
    double E     = 1e7;
    double rho0  = 1000.0;
    double c     = sqrt(E/rho0);
//    double A0     = 1.;                    // << This is normalized below
  	double ri = 0.4;
 	double ro = 0.6;
  	double c1 = (-6.*ri)/(ro*(ro - 3.*ri));
  	double c2 = (3.*(ro + ri))/(pow(ro,2)*(ro - 3.*ri));
  	double c3 = -2./(pow(ro,2)*(ro - 3.*ri));
    int    TotalNumParticles  = 0;   
    double max_errorAllLevels = 0.0;
    double TotalSumError      = 0.0;
    Point  worstPosAllLevels  = Point(-9,-9,-9);
    IntVector worstCellAllLevels = IntVector(-9,-9,-9);
    
    int numLevels = grid->numLevels();
    vector<double>    LinfLevel(numLevels);
    vector<double>    L2normLevel(numLevels);
    vector<Point>     worstPosLevel(numLevels);
    vector<IntVector> worstCellLevel(numLevels);
    vector<int>       numParticles(numLevels);
    
    //__________________________________
    //  Level loop
    for(int l=0;l<numLevels;l++){
      LevelP level = grid->getLevel(l);
    
      double sumError  = 0.0;
      double max_error = 0;
      numParticles[l]  = 0;
      Point worstPos   = Point(-9,-9,-9);
      IntVector worstCell(-9,-9,-9);
      
      Vector dx = level->dCell();             // you need to normalize the variable A by the 
      double A = 0.1;
      //__________________________________
      // Patch loop
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){

        const Patch* patch = *iter;
        
        int matl = clf.matl_jim;
        ParticleVariable<Point>  value_pos;
        ParticleVariable<Vector> value_disp;

        da->query(value_pos,  "p.x",           matl, patch, t);
        da->query(value_disp, "p.displacement",matl, patch, t);
          
        ParticleSubset* pset = value_pos.getParticleSubset(); 
        numParticles[l] += pset->numParticles();
        //__________________________________
        //  Compute the error.       
        if(pset->numParticles() > 0){  // are there particles on this patch
        

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){

            Point refx = value_pos[*iter]-value_disp[*iter];
  	    double R = sqrt(refx.x()*refx.x() + refx.y()*refx.y());
          
            Vector u_exact = A*Vector(sin(c*M_PI*time)*(c3*pow(R,2.) + c2*R +c1)*refx.x(),
                                      sin(c*M_PI*time)*(c3*pow(R,2.) + c2*R +c1)*refx.y(),
                                      0);
            
            double error = (u_exact - value_disp[*iter]).length();
            sumError += error*error;
                
            if (error>max_error){
              max_error = error;
              worstPos  = value_pos[*iter];
              worstCell = patch->getCellIndex(worstPos);
            }
          }  // particle Loop
            
        }  //if
      }  // for patches
      LinfLevel[l]      = max_error;
      worstPosLevel[l]  = worstPos;
      worstCellLevel[l] = worstCell;
      
      if(sumError != 0){
        L2normLevel[l]    = sqrt( sumError/(double)numParticles[l]);
      }else{
        L2normLevel[l]    = 0.0;
      }
      
      cout << "     Level: " << level->getIndex() << " L_inf Error: " << LinfLevel[l] << ", L2norm: " << L2normLevel[l] 
           << " numParticles: " << numParticles[l] << " , Worst particle: " << worstPos << ", " << worstCell << endl;
      
      TotalSumError     += sumError;
      TotalNumParticles += numParticles[l];
      
      if (max_error > max_errorAllLevels) {
        max_errorAllLevels = max_error;
        worstPosAllLevels  = worstPos;
        worstCellAllLevels = worstCell;
      }
    }   // for levels
    double L2norm = sqrt( TotalSumError /(double)TotalNumParticles );
    
    cout << "time: " << time << " , L_inf Error: " << max_errorAllLevels << " , L2norm Error: "<< L2norm << " , Worst particle: " << worstPosAllLevels << " " << worstCellAllLevels << endl;
    
    //__________________________________
    // write data to the files (L_norms & L_normsPerLevels)
    FILE *outFile;
    
    // output level information
    outFile = fopen("L_normsPerLevel","w");
    fprintf(outFile, "#Time,  Level,   L_inf,    L2norm,    NumParticles\n");
    for(int l=0;l<numLevels;l++){
      fprintf(outFile, "%16.16le, %i,  %16.16le,  %16.16le  %i\n", time, l, LinfLevel[l], L2normLevel[l],numParticles[l]);
    }
    fclose(outFile);
    
    // overall 
    outFile = fopen("L_norms","w");
    fprintf(outFile, "#Time,    L_inf,    L2norm\n");
    fprintf(outFile, "%16.16le, %16.16le, %16.16le\n", time, max_errorAllLevels, L2norm);
    fclose(outFile); 
  }
} // end ER_MMS()

