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


#include <Packages/Uintah/CCA/Components/LoadBalancers/ProfileDriver.h>
#include <Core/Util/DebugStream.h>
using namespace Uintah;
using namespace SCIRun;
   
static DebugStream stats("ProfileStats",false);

void ProfileDriver::setMinPatchSize(const vector<IntVector> &min_patch_size)
{
  d_minPatchSize=min_patch_size;
  costs.resize(d_minPatchSize.size());
  d_minPatchSizeVolume.resize(d_minPatchSize.size());
  for(int l=0;l<(int)d_minPatchSize.size();l++)
  {
    d_minPatchSizeVolume[l]=d_minPatchSize[l][0]*d_minPatchSize[l][1]*d_minPatchSize[l][2];
  }
}

void ProfileDriver::addContribution(const PatchSubset* patches, double cost)
{
  if(patches)
  {
    //compute number of datapoints
    int num_points=0;
    for(int p=0;p<patches->size();p++)
    {
      const Patch* patch=patches->get(p);
      int l=patch->getLevel()->getIndex();
      num_points+=patch->getNumCells()/d_minPatchSizeVolume[l];
    }
    double average_cost=cost/num_points;

    //loop through patches
    for(int p=0;p<patches->size();p++)
    {
      const Patch* patch=patches->get(p);
      int l=patch->getLevel()->getIndex();

      //coarsen region by minimum patch size
      IntVector low=patch->getCellLowIndex__New()/d_minPatchSize[l];
      IntVector high=patch->getCellHighIndex__New()/d_minPatchSize[l];

      //loop through datapoints
      for(CellIterator iter(low,high); !iter.done(); iter++)
      {
        //add cost to current contribution
        costs[l][*iter].current+=average_cost;
        //if(d_myworld->myrank()==0)
        //   cout << "level:" << l << " adding: " << average_cost << " to: " << *iter << " new total:" << costs[l][*iter].current << endl;
      }
    }
  }
  else
  {
      //if(d_myworld->myrank()==0)
      //   cout << "no patches\n";
  }
}

void ProfileDriver::outputError(const GridP currentGrid)
{
    
  vector<double> proc_costs(d_myworld->size(),0);
  //for each level
  for (int l=0; l<currentGrid->numLevels();l++)
  {
    LevelP level=currentGrid->getLevel(l);
    vector<Region> regions(level->numPatches());

    for(int p=0; p<level->numPatches();p++)
    {
      const Patch *patch=level->getPatch(p);
      regions[p]=Region(patch->getCellLowIndex__New(),patch->getCellHighIndex__New());
    }

    vector<double> predicted_sum(regions.size(),0), measured_sum(regions.size(),0);
    vector<double>  predicted(regions.size(),0), measured(regions.size(),0);

    for(int r=0;r<(int)regions.size();r++)
    {
      //coarsen region by minimum patch size
      IntVector low=regions[r].getLow()/d_minPatchSize[l];
      IntVector high=regions[r].getHigh()/d_minPatchSize[l];

      //loop through datapoints
      for(CellIterator iter(low,high); !iter.done(); iter++) 
      {
        //search for point in the map
        map<IntVector,Contribution>::iterator it=costs[l].find(*iter);

        //if in the map
        if(it!=costs[l].end())
        {
          //add predicted and measured costs to respective arrays
          predicted[r]+=it->second.weight;
          measured[r]+=it->second.current;
        } 
      }
    }

    //allreduce sum weights
    if(d_myworld->size()>1)
    {
      MPI_Reduce(&predicted[0],&predicted_sum[0],predicted.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
      MPI_Reduce(&measured[0],&measured_sum[0],measured.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    }
    
    for( int p=0;p<level->numPatches();p++)
    {
      const Patch *patch=level->getPatch(p);
     
      int proc=d_lb->getPatchwiseProcessorAssignment(patch);
      proc_costs[proc]+=measured_sum[p];
    }
    
    if(d_myworld->myrank()==0)
    {
      //calculate total cost for normalization
      double total_measured=0, total_predicted=0;
      double total_measured_error=0, total_measured_percent_error=0;
      double total_volume=0;
      for(int r=0;r<(int)regions.size();r++)
      {
        total_measured+=measured_sum[r];
        total_predicted+=predicted_sum[r];
        total_measured_error+=fabs(measured_sum[r]-predicted_sum[r]);
        total_measured_percent_error+=fabs(measured_sum[r]-predicted_sum[r])/measured_sum[r];
        total_volume+=regions[r].getVolume();
      }

      stats << "Profile Error: " << l << " " << total_measured_error/regions.size() << " " << total_measured_percent_error/regions.size() << " " << total_measured << " " << total_predicted << endl;
    }
  }

  if(d_myworld->myrank()==0)
  {
    double meanCost=0;
    double maxCost=proc_costs[0];
    int maxLoc=0;
    for(int p=0;p<d_myworld->size();p++)
    {
      meanCost+=proc_costs[p];
      if(maxCost<proc_costs[p])
      {
        maxCost=proc_costs[p];
        maxLoc=p;
      }
    }
    stats << "LoadBalance Measured:  Mean:" << meanCost/d_myworld->size() << " Max:" << maxCost << " on processor " << maxLoc << endl;
  }
}
void ProfileDriver::finalizeContributions(const GridP currentGrid)
{
 
  if(stats.active())
  {
    outputError(currentGrid);
  }
   
  timesteps++;
  //for each level
  for(int l=0;l<(int)costs.size();l++)
  {
    //for each datapoint
    for(map<IntVector,Contribution>::iterator iter=costs[l].begin();iter!=costs[l].end();)
    {
      //save and increment iterator
      map<IntVector,Contribution>::iterator it=iter++;
     
      //create a reference to the data
      Contribution &data=it->second;
      
      //update timestep
      if(data.current>0)
        data.timestep=0;
      else
        data.timestep++;

      if(timesteps<=2)
      {
        //first couple timesteps should be set to last timestep to initialize the system
          //the first couple timesteps are not representative of the actual cost as extra things
          //are occuring.
        data.weight=data.current;
      }
      else
      {

        //update exponential averagea
        data.weight=d_alpha*data.current+(1-d_alpha)*data.weight;
      }
      
      //reset current
      data.current=0;
      
      //if data is old 
      if (data.timestep>log(.001*d_alpha)/log(1-d_alpha))
      {
           //erase saved iterator in order to save space and time
           costs[l].erase(it);
      }
    }
  }
}

void ProfileDriver::getWeights(int l, const vector<Region> &regions, vector<double> &weights)
{
  if(regions.size()==0)
    return;

  weights.resize(regions.size());
  vector<double> partial_weights(regions.size(),0);      

  for(int r=0;r<(int)regions.size();r++)
  {
    //coarsen region by minimum patch size
    IntVector low=regions[r].getLow()/d_minPatchSize[l];
    IntVector high=regions[r].getHigh()/d_minPatchSize[l];
 
    //loop through datapoints
    for(CellIterator iter(low,high); !iter.done(); iter++)
    {
      //search for point in the map so we don't inadvertantly add it to the map.  If it doesn't exist on this processor
      //it should exist on another processor
      map<IntVector,Contribution>::iterator it=costs[l].find(*iter);
        
      //if in the map
      if(it!=costs[l].end())
      {
        //add cost to weight
        partial_weights[r]+=it->second.weight;
      }
    }
  }

  //allreduce sum weights
  if(d_myworld->size()>1)
    MPI_Allreduce(&partial_weights[0],&weights[0],weights.size(),MPI_DOUBLE,MPI_SUM,d_myworld->getComm());
 
}

void ProfileDriver::initializeWeights(const Grid* oldgrid, const Grid* newgrid)
{
  //for each level
  for(int l=1;l<newgrid->numLevels();l++)
  {
    vector<Region> old_level;

    if(oldgrid && oldgrid->numLevels()>l)
    {
      //create old_level vector
      for(int p=0; p<oldgrid->getLevel(l)->numPatches(); p++) 
      {
        const Patch* patch = oldgrid->getLevel(l)->getPatch(p);
        old_level.push_back(Region(patch->getCellLowIndex__New(), patch->getCellHighIndex__New()));
      }
    }
    
    //get weights on old_level
    vector<double> weights;
    getWeights(l,old_level,weights);
    
    double volume=0;
    double weight=0;
    //compute average cost per datapoint
    for(int r=0;r<(int)old_level.size();r++)
    {
      volume+=old_level[r].getVolume();
      weight+=weights[r];
    }
    double average_cost=weight/volume*d_minPatchSizeVolume[l];
   
    //if there is no cost data
    if(average_cost==0)
    {
      //set each datapoint to the same small value
      average_cost=1;
    }
    
    //compute regions in new level that are not in old
    
    vector<Region> new_regions, dnew, dold(old_level.begin(),old_level.end());
    
    //create dnew to contain a subset of the new patches
    for(int p=0; p<newgrid->getLevel(l)->numPatches(); p++) 
    {
      const Patch* patch = newgrid->getLevel(l)->getPatch(p);
      if(p%d_myworld->size()==d_myworld->myrank())
        dnew.push_back(Region(patch->getCellLowIndex__New(), patch->getCellHighIndex__New()));
    }
    
    //compute difference
    new_regions=Region::difference(dnew, dold);
   

    int i=0;
    //initialize weights 
    for(vector<Region>::iterator it=new_regions.begin();it!=new_regions.end();it++)
    {
      //add regions to my map
      IntVector low=it->getLow()/d_minPatchSize[l];
      IntVector high=it->getHigh()/d_minPatchSize[l];

      //loop through datapoints
      for(CellIterator iter(low,high); !iter.done(); iter++)
      {
        //add cost to current contribution
        costs[l][*iter].weight=average_cost;
      } //end cell iteration
      i++;
    } //end region iteration
  }// end levels iteration
}
void ProfileDriver::reset()
{
  for(int i=0;i<(int)costs.size();i++)
  {
    costs[i].clear();
  }
  timesteps=0;
}
