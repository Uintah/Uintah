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


#include <CCA/Components/LoadBalancers/CostModelForecaster.h>
#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Util/DebugStream.h>
using namespace Uintah;
using namespace SCIRun;
   
namespace Uintah
{
static DebugStream stats("ProfileStats", false);
void
CostModelForecaster::addContribution( DetailedTask *task, double cost )
{
  const PatchSubset *patches=task->getPatches();
  
  if( patches == 0 ) {
    return;
  }
 
  //compute cost per cell so the measured time can be distributed poportionally by cells
  int num_cells=0;
  for(int p=0;p<patches->size();p++)
  {
    const Patch* patch=patches->get(p);
    num_cells+=patch->getNumExtraCells();
  }
  double cost_per_cell=cost/num_cells;

  //loop through patches
  for(int p=0;p<patches->size();p++)
  {
    const Patch* patch=patches->get(p);

    execTimes[patch->getID()]+=patch->getNumExtraCells()*cost_per_cell;
  }
}

void CostModelForecaster::collectPatchInfo(const GridP grid, vector<PatchInfo> &patch_info) 
{

  vector<vector<int> > num_particles;
  d_lb->collectParticles(grid.get_rep(),num_particles);

  vector<PatchInfo> patchList;
  vector<int> num_patches(d_myworld->size(),0);

  int total_patches=0;
  //for each level
  for(int l=0;l<grid->numLevels();l++) {
    //for each patch
    const LevelP& level = grid->getLevel(l);
    total_patches+=level->numPatches();
    for (int p=0;p<level->numPatches();p++) {
      const Patch *patch = level->getPatch(p);
      //compute number of patches on each processor
      int owner=d_lb->getPatchwiseProcessorAssignment(patch);
      num_patches[owner]++;
      //if I own patch
      if(owner==d_myworld->myrank())
      {
        // add to patch list
        PatchInfo pinfo(num_particles[l][p],patch->getNumExtraCells(),execTimes[patch->getID()]);
        patchList.push_back(pinfo);
      }
    }
  }

  vector<int> displs(d_myworld->size(),0), recvs(d_myworld->size(),0);

  //compute recvs and displs
  for(int i=0;i<d_myworld->size();i++)
    recvs[i]=num_patches[i]*sizeof(PatchInfo);
  for(int i=1;i<d_myworld->size();i++)
    displs[i]=displs[i-1]+recvs[i-1];

  patch_info.resize(total_patches);
  //allgather the patch info
  MPI_Allgatherv(&patchList[0], patchList.size()*sizeof(PatchInfo),  MPI_BYTE,
                    &patch_info[0], &recvs[0], &displs[0], MPI_BYTE,
                    d_myworld->getComm());

}
void
CostModelForecaster::finalizeContributions( const GridP currentGrid )
{

  //least squares to compute coefficients
#if 0 //parallel

#else //serial
  //collect the patch information needed to compute the coefficients
  vector<PatchInfo> patch_info;
  collectPatchInfo(currentGrid,patch_info);

  if(stats.active() && d_myworld->myrank()==0)
  {
    static int j=0;
    for(size_t i=0;i<patch_info.size();i++)
    {
      stats << j << " " << patch_info[i] << endl;
    }
  }
  //compute least squares
#endif

  //update coefficients

  execTimes.clear();
}

void
CostModelForecaster::getWeights(const Grid* grid, vector<vector<int> > num_particles, vector<vector<double> >&costs)
{
  CostModeler::getWeights(grid,num_particles,costs);
}
  
ostream& operator<<(ostream& out, const CostModelForecaster::PatchInfo &pi)
{
  out << pi.num_cells << " " << pi.num_particles << " " << pi.execTime ;
  return out;
}
}
