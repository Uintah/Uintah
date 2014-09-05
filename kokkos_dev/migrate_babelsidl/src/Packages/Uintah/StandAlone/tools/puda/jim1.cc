
#include <Packages/Uintah/StandAlone/tools/puda/jim1.h>

#include <Packages/Uintah/StandAlone/tools/puda/util.h>

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>

#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              J I M 1   O P T I O N
// This currently pulls out particle position, velocity and ID
// and prints that on one line for each particle.  This is useful
// for postprocessing of particle data for particles which move
// across patches.

void
Uintah::jim1( DataArchive * da, CommandLineFlags & clf )
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
    cout << "time = " << time << endl;
    GridP grid = da->queryGrid(time);
    ostringstream fnum;
    string filename;
    fnum << setw(4) << setfill('0') << t/clf.time_step_inc;
    string partroot("partout");
    filename = partroot+ fnum.str();
    ofstream partfile(filename.c_str());

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      cout << "Level: " <<  endl;
      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        int matl = clf.matl_jim1;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        ParticleVariable<long64> value_pID;
        ParticleVariable<Point> value_pos;
        ParticleVariable<Vector> value_vel;
        ParticleVariable<Matrix3> value_strs;
        da->query(value_pID, "p.particleID", matl, patch, time);
        da->query(value_pos, "p.x",          matl, patch, time);
        da->query(value_vel, "p.velocity",matl, patch, time);
        da->query(value_strs,"p.stress",     matl, patch, time);
        ParticleSubset* pset = value_pos.getParticleSubset();
        if(pset->numParticles() > 0){
          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            partfile << value_pos[*iter].x() << " "
                     << value_vel[*iter].x() << " "
                     << value_strs[*iter](0,0) << " ";
            partfile << value_pID[*iter] <<  endl;
          } // for
#if 0
          for(;iter != pset->end(); iter++){
            partfile << value_pos[*iter].x() << " " <<
              value_pos[*iter].y() << " " <<
              value_pos[*iter].z() << " "; 
            partfile << value_vel[*iter].x() << " " <<
              value_vel[*iter].y() << " " <<
              value_vel[*iter].z() << " "; 
            partfile << value_pID[*iter] <<  endl;
          } // for
#endif
        }  //if
      }  // for patches
    }   // for levels
  }
} // end jim1()

