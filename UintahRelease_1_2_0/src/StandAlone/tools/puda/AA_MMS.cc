
#include <StandAlone/tools/puda/AA_MMS.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              AA_MMS   O P T I O N
// Compares the exact solution for the 3D axis aligned MMS problem with
// the MPM solution.  Computes the L-infinity error on the displacement
// at each timestep and reports to the command line.

void
Uintah::AA_MMS( DataArchive * da, CommandLineFlags & clf )
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

    double error, max_error=0.;
    double mu = 3846.;
    double bulk = 8333.;
    double E = 9.*bulk*mu/(3.*bulk+mu);
    double rho0=1.0;
    double c = sqrt(E/rho0);
    double A=.1;
    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        particleIndex worst_idx = 0;
        const Patch* patch = *iter;
        int matl = clf.matl_jim;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<Point> value_pos;
        ParticleVariable<Vector> value_disp;
        da->query(value_pos,  "p.x",           matl, patch, t);
        da->query(value_disp, "p.displacement",matl, patch, t);
        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            Point refx = value_pos[*iter]-value_disp[*iter];
            Vector u_exact=A*Vector(sin(M_PI*refx.x())*sin(c*M_PI*time),
                             sin(M_PI*refx.y())*sin(2.*M_PI/3.+c*M_PI*time),
                             sin(M_PI*refx.z())*sin(4.*M_PI/3.+c*M_PI*time));
            error = (u_exact-value_disp[*iter]).length();
            if (error>max_error){
                 max_error=error;
                 worst_idx=*iter;
            }
          } // for
        }  //if
    cout << "time = " << time << ", L_inf Error = " << max_error << ", Worst particle = " << value_pos[worst_idx] << endl;
      }  // for patches
    }   // for levels
  }
} // end AA_MMS()

