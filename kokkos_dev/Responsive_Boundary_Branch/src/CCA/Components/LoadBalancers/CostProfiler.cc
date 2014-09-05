/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/LoadBalancers/CostProfiler.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Util/DebugStream.h>
using namespace Uintah;
using namespace SCIRun;
   
void
CostProfiler::setMinPatchSize( const vector<IntVector> & min_patch_size )
{
  d_profiler.setMinPatchSize(min_patch_size);
}

void
CostProfiler::addContribution( DetailedTask *task, double cost )
{
  if( task->getPatches() == 0 ) {
    return;
  }
#if 0  
  const PatchSubset *patches=task->getPatches();
  for(int p=0;p<patches->size();p++)
  {
    const Patch* patch=patches->get(p);
    if(d_lb->getPatchwiseProcessorAssignment(patch)!=d_myworld->myrank())
      cout << d_myworld->myrank() << " error patch is owned by processor:" << d_lb->getPatchwiseProcessorAssignment(patch) << " for task:" << task->getName() << endl;
  }
#endif
  d_profiler.addContribution( task->getPatches(), cost );
}

void
CostProfiler::outputError( const GridP currentGrid )
{
  d_profiler.outputError( currentGrid );
}

void
CostProfiler::finalizeContributions( const GridP currentGrid )
{
  d_profiler.finalizeContributions( currentGrid );  
}

void
CostProfiler::getWeights(const Grid* grid, vector<vector<int> > num_particles, vector<vector<double> >&costs)
{
  costs.resize(grid->numLevels());
  //for each level
  for (int l=0; l<grid->numLevels();l++)
  {
    LevelP level=grid->getLevel(l);
    vector<Region> regions(level->numPatches());

    costs[l].resize(level->numPatches());
    for(int p=0; p<level->numPatches();p++)
    {
      const Patch *patch=level->getPatch(p);
      regions[p]=Region(patch->getCellLowIndex(),patch->getCellHighIndex());
    }
    d_profiler.getWeights(l,regions,costs[l]);
  }
}

void
CostProfiler::initializeWeights( const Grid * oldgrid, const Grid * newgrid )
{
  d_profiler.initializeWeights( oldgrid, newgrid );
}

void
CostProfiler::reset()
{
  d_profiler.reset();
}
