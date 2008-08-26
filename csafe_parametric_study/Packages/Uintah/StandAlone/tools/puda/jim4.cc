#include <Packages/Uintah/StandAlone/tools/puda/jim4.h>
#include <Packages/Uintah/StandAlone/tools/puda/util.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <iomanip>
#include <fstream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////
//              J I M 4   O P T I O N
//
//  Computes the integral of P*V for the products of reaction for the
//  parameter study in order to compute the amount of stored energy in
//  the products of reaction.  Also computes thermal energy and KE.
//  KE is only available via the checkpoint files
void
Uintah::jim4( DataArchive * da, CommandLineFlags & clf )
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

  ostringstream fnum;
  string filename("PV_rhoCvT_KE.dat");
  ofstream outfile(filename.c_str());

  for(unsigned long t=clf.time_step_lower;t<=clf.time_step_upper;t+=clf.time_step_inc){
    double time = times[t];
    cout << "time = " << time << endl;

    outfile.precision(14);
    outfile << time << " "; 
    GridP grid = da->queryGrid(t);

    for(int l=0;l<grid->numLevels();l++){
      LevelP level = grid->getLevel(l);
      cout << "\t    Level: " << level->getIndex() << ", id " << level->getID() << endl;

      Vector dx = level->dCell();
      double cell_vol = dx.x()*dx.y()*dx.z();
      double PV=0;
      double rhoT=0;
      double KE=0;

      for(Level::const_patchIterator iter = level->patchesBegin();
          iter != level->patchesEnd(); iter++){
        const Patch* patch = *iter;
        int matl = clf.matl_jim;
        //__________________________________
        //   P A R T I C L E   V A R I A B L E
        CCVariable<double> val_press, val_vol_frac, val_rho, val_temp;
        CCVariable<Vector> val_vel;
        da->query(val_press,    "press_CC",       0,    patch, t);
        da->query(val_vol_frac, "vol_frac_CC",    matl, patch, t);
        da->query(val_rho,      "rho_CC",         matl, patch, t);
        da->query(val_temp,     "temp_CC",        matl, patch, t);
        da->query(val_vel,      "vel_CC",         matl, patch, t);

        for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
           IntVector c = *iter;
           PV   += val_press[c] * val_vol_frac[c]*cell_vol;
           rhoT += 716*val_rho[c] * val_temp[c]*cell_vol;
           KE   += .5*(val_vel[c].length() * val_vel[c].length()) * val_rho[c]*cell_vol;
        }
      }  // for patches
      outfile << PV << " " << rhoT << " " << KE << " ";
    }  // for levels
    outfile << endl;

  }
} // end jim3()
