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


#include <Packages/Uintah/CCA/Components/LoadBalancers/CostProfiler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Util/DebugStream.h>
using namespace Uintah;
using namespace SCIRun;
  
static DebugStream stats("ProfileStats",false);

void CostProfiler::setMinPatchSize(const vector<IntVector> &min_patch_size)
{
  d_minPatchSize=min_patch_size;
  d_profiler.setNumLevels(d_minPatchSize.size());
  d_fineProfiler.setNumLevels(d_minPatchSize.size());
}

void CostProfiler::addContribution(DetailedTask *task, double cost)
{
  if(task->getPatches()==0)
    return;

  vector<Region> regions;
  vector<int> levels;
  if(task->getTask()->isFineTask() )
  {
    const PatchSubset *patches=task->getPatches();

    for(int p=0;p<patches->size();p++)
    {
      const Patch* patch=patches->get(p);
      int l=patch->getLevel()->getIndex();
      Region r=Region(patch->getCellLowIndex__New(),patch->getCellHighIndex__New());
      
      Patch::selectType fp;
      patch->getFineLevelPatches(fp);

      //map region to the finer level 
      r.low()=r.low()*d_refinementRatios[l];
      r.high()=r.high()*d_refinementRatios[l];

      //for each finer level patch
      for(int f=0;f<fp.size();f++)
      {
        //intersect r and fine patch
        IntVector low=Max(r.getLow(),fp[f]->getCellLowIndex__New());
        IntVector high=Min(r.getHigh(),fp[f]->getCellHighIndex__New());
        
        //coarsen by fine level min patch size
        low=low/d_minPatchSize[l+1];
        high=high/d_minPatchSize[l+1];
        
        //add the intersection to regions
        regions.push_back(Region(low,high));
        levels.push_back(l+1);
      }
    }
    d_fineProfiler.addContribution(regions,levels,cost);
  }
  else  //this is a normal task to profile it in the normal profiler 
  {
    const PatchSubset *patches=task->getPatches();

    for(int p=0;p<patches->size();p++)
    {
      const Patch* patch=patches->get(p);
      int l=patch->getLevel()->getIndex();

      Region r=Region(patch->getCellLowIndex__New(),patch->getCellHighIndex__New());

      //coarsen by min patch size
      r.low()=r.low()/d_minPatchSize[l];
      r.high()=r.high()/d_minPatchSize[l];

      regions.push_back(r);
      levels.push_back(l);
    } 
    d_profiler.addContribution(regions,levels,cost);
  }
}

void CostProfiler::outputError(const GridP currentGrid)
{
  //for each level
  for (int l=0; l<currentGrid->numLevels();l++)
  {
    LevelP level=currentGrid->getLevel(l);
    vector<Region> regions(level->numPatches());
    vector<Region> fineRegions(level->numPatches());
    for(int p=0; p<level->numPatches();p++)
    {
      const Patch *patch=level->getPatch(p);
      regions[p]=Region(patch->getCellLowIndex__New()/d_minPatchSize[l],patch->getCellHighIndex__New()/d_minPatchSize[l]);
      if(l+1<currentGrid->numLevels())
      {
        fineRegions[p]=Region(patch->getCellLowIndex__New()*d_refinementRatios[l]/d_minPatchSize[l+1],patch->getCellHighIndex__New()*d_refinementRatios[l]/d_minPatchSize[l+1]);
      }
    }

    vector<double> measured, predicted, measured_fine, predicted_fine;
    //get measured and predicted weights
    d_profiler.getMeasuredAndPredictedWeights(l,regions,measured,predicted);

    double total_measured_error=0, total_volume=0, total_measured=0, total_predicted=0, total_measured_percent_error=0;
    if(d_myworld->myrank()==0)
    {
      //add measured and predicted
      for(unsigned int r=0;r<measured.size();r++)
      {
        total_measured+=measured[r];
        total_predicted+=predicted[r];
        total_measured_error+=fabs(measured[r]-predicted[r]);
        total_measured_percent_error+=fabs(measured[r]-predicted[r])/measured[r];
        total_volume+=regions[r].getVolume();
      }

      //cout << "Normal Profile Error: " << l << " " << total_measured_error/total_volume << " " << total_measured << " " << total_predicted << endl;
      cout << "Normal Profile Error: " << l << " " << total_measured_error/regions.size() << " " << total_measured_percent_error/regions.size() << " " << total_measured << " " << total_predicted << endl;

    }

    double total_measured_error_fine=0, total_volume_fine=0, total_measured_fine=0, total_predicted_fine=0, total_measured_percent_error_fine=0;
    if(l+1<currentGrid->numLevels())
    {
      d_fineProfiler.getMeasuredAndPredictedWeights(l+1,fineRegions,measured_fine,predicted_fine);
      if(d_myworld->myrank()==0)
      {
        for(unsigned int r=0;r<measured_fine.size();r++)
        {
          total_measured_fine+=measured_fine[r];
          total_predicted_fine+=predicted_fine[r];
          total_measured_error_fine+=fabs(measured_fine[r]-predicted_fine[r]);
          if(measured_fine[r]>0)
          {
            total_measured_percent_error_fine+=fabs(measured_fine[r]-predicted_fine[r])/measured_fine[r];
          }
          total_volume_fine+=fineRegions[r].getVolume();
        }

        //cout << "Fine Profile Error:" << l << " " << total_measured_error_fine/total_volume_fine 
        //  << " " << total_measured_fine << " " << total_predicted_fine  << endl;
        cout << "Fine Profile Error:" << l << " " << total_measured_error_fine/regions.size() 
          << " " << total_measured_percent_error_fine/regions.size() 
          << " " << total_measured_fine << " " << total_predicted_fine  << endl;

      }
    }
    if(d_myworld->myrank()==0) 
    {
      //cout << "Total Profile Error:" << l << " " << (total_measured_error+total_measured_error_fine)/(total_volume+total_volume_fine) << " "
      //  << total_measured+total_measured_fine << " " << total_predicted+total_predicted_fine << endl;
      cout << "Total Profile Error:" << l << " " << (total_measured_error+total_measured_error_fine)/regions.size() << " "
        <<  (total_measured_percent_error+total_measured_percent_error_fine)/regions.size() << " "
        << total_measured+total_measured_fine << " " << total_predicted+total_predicted_fine << endl;
    }
  }
}
void CostProfiler::finalizeContributions(const GridP currentGrid)
{
  if(stats.active())
    outputError(currentGrid);
  
  d_profiler.finalizeContributions();  
  d_fineProfiler.finalizeContributions();

}

void CostProfiler::getWeights(int l, const vector<Region> &regions, vector<double> &weights)
{

  vector<Region> normalRegions(regions);
  weights.resize(regions.size());

  //coarsen by minimum patch size
  for(unsigned int r=0;r<regions.size();r++)
  {
    normalRegions[r].low()=normalRegions[r].low()/d_minPatchSize[l];
    normalRegions[r].high()=normalRegions[r].high()/d_minPatchSize[l];
  }

  //get weights for the normal tasks
  d_profiler.getWeights(l,normalRegions,weights);
  //if a level below this task exists
  if(l+1<d_numLevels)
  {
    vector<Region> fineRegions(regions);
    vector<double> fineWeights(regions.size());

    //map regions to finer level and coarsen by minimum patch size
    for(unsigned int i=0;i<fineRegions.size();i++)
    {
     fineRegions[i].low()=fineRegions[i].low()*d_refinementRatios[l]/d_minPatchSize[l+1];
     fineRegions[i].high()=fineRegions[i].high()*d_refinementRatios[l]/d_minPatchSize[l+1];
    }
    
    //get weights for finer regions
    d_fineProfiler.getWeights(l+1,fineRegions,fineWeights); 
    
    //add fine level weights
    for(unsigned int i=0;i<fineRegions.size();i++)
      weights[i]+=fineWeights[i];
  }
}

void CostProfiler::initializeWeights(const Grid* oldgrid, const Grid* newgrid)
{

  //for each level
  for(int l=1;l<newgrid->numLevels();l++)
  {
    vector<Region> old_regions, new_regions;

    if(oldgrid && oldgrid->numLevels()>l)
    {
      //create old_level vector
      for(int p=0; p<oldgrid->getLevel(l)->numPatches(); p++) 
      {
        const Patch* patch = oldgrid->getLevel(l)->getPatch(p);
        old_regions.push_back(Region(patch->getCellLowIndex__New()/d_minPatchSize[l], patch->getCellHighIndex__New()/d_minPatchSize[l]));
      }
    }
    
    //compute regions in new level that are not in old
    deque<Region> new_regions_q, dnew, dold(old_regions.begin(),old_regions.end());
    
    //create dnew to contain a subset of the new patches
    for(int p=0; p<newgrid->getLevel(l)->numPatches(); p++) 
    {
      const Patch* patch = newgrid->getLevel(l)->getPatch(p);
      if(p%d_myworld->size()==d_myworld->myrank())
        dnew.push_back(Region(patch->getCellLowIndex__New()/d_minPatchSize[l], patch->getCellHighIndex__New()/d_minPatchSize[l]));
    }
    
    //compute difference
    new_regions_q=Region::difference(dnew, dold);

    new_regions.assign(new_regions_q.begin(),new_regions_q.end());

    d_profiler.initializeWeights(old_regions,new_regions,l);
    d_fineProfiler.initializeWeights(old_regions,new_regions,l);
  }

}
void CostProfiler::reset()
{
  d_profiler.reset();
  d_fineProfiler.reset();
}
