#include <Packages/Uintah/CCA/Components/LoadBalancers/CostProfiler.h>
#include <Core/Util/DebugStream.h>
using namespace Uintah;
using namespace SCIRun;
   
void CostProfiler::setMinPatchSize(const vector<IntVector> &min_patch_size)
{
  d_profiler.setMinPatchSize(min_patch_size);
}

void CostProfiler::addContribution(const PatchSubset *patches, double cost)
{
  d_profiler.addContribution(patches,cost);
}

void CostProfiler::outputError(const GridP currentGrid)
{
  d_profiler.outputError(currentGrid);
}
void CostProfiler::finalizeContributions(const GridP currentGrid)
{
  d_profiler.finalizeContributions(currentGrid);  
}

void CostProfiler::getWeights(int l, const vector<Region> &regions, vector<double> &weights)
{
  d_profiler.getWeights(l,regions,weights);
}

void CostProfiler::initializeWeights(const Grid* oldgrid, const Grid* newgrid)
{
  d_profiler.initializeWeights(oldgrid,newgrid);
}
void CostProfiler::reset()
{
  d_profiler.reset();
}
