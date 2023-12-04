
#include <StandAlone/tools/puda/makeRawFile.h>
#include <StandAlone/tools/puda/util.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <fstream>
#include <vector>

using namespace Uintah;
using namespace std;

/*______________________________________________________________________
                   makeRawFile
 

//______________________________________________________________________*/

unsigned int get1DIndex(int i, int j, int k,
                        int X, int Y)
{
  // given voxel indices in 3D space, return an index
  // into the 1D array that stores the voxel values
  return k*(X*Y) + j*X + i;
}

void
Uintah::makeRawFile( DataArchive * da, CommandLineFlags & clf )
{
  vector<int> index;
  vector<double> times;
  
  da->queryTimesteps(index, times);
  
  ASSERTEQ(index.size(), times.size());
  cout << "There are " << index.size() << " timesteps:\n";
  for( int i = 0; i < (int)index.size(); i++ ) {
    cout << index[i] << ": " << times[i] << endl;
  }

  findTimestep_loopLimits( clf.tslow_set, clf.tsup_set, times, 
                           clf.time_step_lower, clf.time_step_upper);
  
  GridP grid = da->queryGrid(0); // Query Grid at time 0

  LevelP level = grid->getLevel(0); // Only one level (0)

  IntVector lowIndex, highIndex;
  level->findInteriorCellIndexRange(lowIndex, highIndex);

  unsigned int totalVoxels = highIndex.x()*highIndex.y()*highIndex.z();
  char* voxel;
  voxel = new char[totalVoxels];

  ostringstream fnumx, fnumy, fnumz;
  fnumx << highIndex.x();
  fnumy << highIndex.y();
  fnumz << highIndex.z();
  string outRoot = "cells_";
  string outFile = outRoot + fnumx.str() + "_" + fnumy.str() + "_"
                           + fnumz.str() + ".raw";
  // write header to file
  FILE *output;
  output = fopen( outFile.c_str(),"w");       

  for(Level::const_patch_iterator iter = level->patchesBegin(); 
                                  iter != level->patchesEnd(); iter++){
    const Patch* patch = *iter;
    int matl = 0;
    CCVariable<int> NAPID;

    da->query( NAPID,  "cellNAPID",  matl, patch, 0);

    //__________________________________
    //  Sum contributions over patch        
    for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      unsigned int oneDIndex = get1DIndex(c.x(), c.y(), c.z(),
                                          highIndex.x(), highIndex.y());
      int val = NAPID[c];
      if(val>0){
        voxel[oneDIndex]=0;
      } else {
        voxel[oneDIndex]=1;
      }
    }
  } // for patches
  fwrite(voxel,1,totalVoxels,output);
  fclose(output);
} // end makeRawFile()
