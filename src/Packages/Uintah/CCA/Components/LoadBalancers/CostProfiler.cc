#include <Packages/Uintah/CCA/Components/LoadBalancers/CostProfiler.h>
#include <Core/Util/DebugStream.h>
using namespace Uintah;
using namespace SCIRun;
   
static DebugStream stats("ProfileStats",false);

void CostProfiler::setMinPatchSize(const vector<IntVector> &min_patch_size)
{
  d_minPatchSize=min_patch_size;
  costs.resize(d_minPatchSize.size());
  d_minPatchSizeVolume.resize(d_minPatchSize.size());
  for(int l=0;l<d_minPatchSize.size();l++)
  {
    d_minPatchSizeVolume[l]=d_minPatchSize[l][0]*d_minPatchSize[l][1]*d_minPatchSize[l][2];
  }
}

void CostProfiler::addContribution(const PatchSubset* patches, double cost)
{
  if(patches)
  {
    //compute number of datapoints
    int num_points=0;
    for(int p=0;p<patches->size();p++)
    {
      const Patch* patch=patches->get(p);
      int l=patch->getLevel()->getIndex();
      num_points+=patch->getInteriorVolume()/d_minPatchSizeVolume[l];
    }
    double average_cost=cost/num_points;
  
    //loop through patches
    for(int p=0;p<patches->size();p++)
    {
      const Patch* patch=patches->get(p);
      int l=patch->getLevel()->getIndex();
    
      //coarsen region by minimum patch size
      IntVector low=patch->getInteriorCellLowIndex()/d_minPatchSize[l];
      IntVector high=patch->getInteriorCellHighIndex()/d_minPatchSize[l];
    
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

void CostProfiler::outputError(const GridP currentGrid)
{
    //for each level
    for (int l=0; l<currentGrid->numLevels();l++)
    {
      LevelP level=currentGrid->getLevel(l);
      vector<Region> regions(level->numPatches());

      for(int p=0; p<level->numPatches();p++)
      {
        const Patch *patch=level->getPatch(p);
        regions[p]=Region(patch->getInteriorCellLowIndex(),patch->getInteriorCellHighIndex());
      }

      vector<double> predictedzo_sum(regions.size(),0), predictedfo_sum(regions.size(),0), predictedso_sum(regions.size(),0), predictedto_sum(regions.size(),0), measured_sum(regions.size(),0);
      vector<double> predictedzo(regions.size(),0), predictedfo(regions.size(),0), predictedso(regions.size(),0), predictedto(regions.size(),0), measured(regions.size(),0);
 
      for(int r=0;r<regions.size();r++)
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
            //add predictedfo and measured costs to respective arrays
            predictedzo[r]+=it->second.zoweight;
            predictedfo[r]+=it->second.foweight;
            predictedso[r]+=it->second.soweight;
            predictedto[r]+=it->second.toweight;
            measured[r]+=it->second.current;
          } 
        }
      }

      //allreduce sum weights
      if(d_myworld->size()>1)
      {
        MPI_Reduce(&predictedzo[0],&predictedzo_sum[0],predictedzo.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
        MPI_Reduce(&predictedfo[0],&predictedfo_sum[0],predictedfo.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
        MPI_Reduce(&predictedso[0],&predictedso_sum[0],predictedso.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
        MPI_Reduce(&predictedto[0],&predictedto_sum[0],predictedto.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
        MPI_Reduce(&measured[0],&measured_sum[0],measured.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
      }
      if(d_myworld->myrank()==0)
      {
        double total_zoerror=0,mean_zoerror=0, total_percent_zoerror=0, mean_percent_zoerror=0;
        double total_foerror=0,mean_foerror=0, total_percent_foerror=0, mean_percent_foerror=0;
        double total_soerror=0,mean_soerror=0, total_percent_soerror=0, mean_percent_soerror=0;
        double total_toerror=0,mean_toerror=0, total_percent_toerror=0, mean_percent_toerror=0;
        double total_measured=0, total_zopredicted=0, total_fopredicted=0, total_sopredicted=0, total_topredicted=0;
        for(int r=0;r<predictedfo.size();r++)
        {
          total_measured+=measured_sum[r];
          total_zopredicted+=predictedzo_sum[r];
          total_fopredicted+=predictedfo_sum[r];
          total_sopredicted+=predictedso_sum[r];
          total_topredicted+=predictedto_sum[r];
          double zoerror=fabs(predictedzo_sum[r]-measured_sum[r]);
          double foerror=fabs(predictedfo_sum[r]-measured_sum[r]);
          double soerror=fabs(predictedso_sum[r]-measured_sum[r]);
          double toerror=fabs(predictedto_sum[r]-measured_sum[r]);
          total_zoerror+=zoerror;
          total_foerror+=foerror;
          total_soerror+=soerror;
          total_toerror+=toerror;
          ASSERT(measured_sum[r]!=0);
          total_percent_zoerror+=(zoerror/measured_sum[r]);
          total_percent_foerror+=(foerror/measured_sum[r]);
          total_percent_soerror+=(soerror/measured_sum[r]);
          total_percent_toerror+=(toerror/measured_sum[r]);
        }
        mean_zoerror=total_zoerror/predictedzo.size();
        mean_foerror=total_foerror/predictedfo.size();
        mean_soerror=total_soerror/predictedso.size();
        mean_toerror=total_toerror/predictedto.size();
        mean_percent_zoerror=total_percent_zoerror/predictedzo.size();
        mean_percent_foerror=total_percent_foerror/predictedfo.size();
        mean_percent_soerror=total_percent_soerror/predictedso.size();
        mean_percent_toerror=total_percent_toerror/predictedto.size();
       stats << timesteps << " " << l <<  " " << mean_percent_zoerror << " " << mean_percent_foerror << " " <<mean_percent_soerror << " " << mean_percent_toerror << " "
             << total_measured << " " << " " << total_zopredicted << " " << total_fopredicted << " " << total_sopredicted << " " << total_topredicted << endl;
       cout << timesteps << " " << l <<  " " << mean_percent_zoerror << " " << mean_percent_foerror << " " <<mean_percent_soerror << " " << mean_percent_toerror << " "
             << total_measured << " " << " " << total_zopredicted << " " << total_fopredicted << " " << total_sopredicted << " " << total_topredicted << endl;
      }
  }
}
void CostProfiler::finalizeContributions(const GridP currentGrid)
{
 
  if(stats.active())
  {
    outputError(currentGrid);
  }
   
  timesteps++;
  //for each level
  for(int l=0;l<costs.size();l++)
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

      //calculate alpha
      double alpha=2.0/(min(d_timestepWindow,timesteps)+1);

      //update exponential averagea
      data.zoweight=data.current;
      data.foweight=alpha*data.current+(1-alpha)*data.foweight;
      data.soweight=alpha*data.foweight+(1-alpha)*data.soweight;
      data.toweight=alpha*data.soweight+(1-alpha)*data.toweight;
      //reset current
      data.current=0;
      
      //if data is old 
      if (data.timestep>log(.001*alpha)/log(1-alpha))
      {
           //erase saved iterator in order to save space and time
           costs[l].erase(it);
      }
    }
  }
}

void CostProfiler::getWeights(int l, const vector<Region> &regions, vector<double> &weights)
{
  weights.resize(regions.size());
  vector<double> partial_weights(regions.size(),0);      

  for(int r=0;r<regions.size();r++)
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
        //add cost to weight
        partial_weights[r]+=it->second.zoweight;
      }
    }
  }

  //allreduce sum weights
  if(d_myworld->size()>1)
    MPI_Allreduce(&partial_weights[0],&weights[0],weights.size(),MPI_DOUBLE,MPI_SUM,d_myworld->getComm());
 
}

void CostProfiler::initializeWeights(const Grid* oldgrid, const Grid* newgrid)
{
  if(d_myworld->myrank()==0)
          stats << timesteps << " " << 9999 << " " 
                << 0 << " " << 0 << " " << 0 << " " << 0 << " " 
                << 0 << " " << 0 << " " << 0 << " " << 0 << " " 
                << 0 << " "  << endl;
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
        old_level.push_back(Region(patch->getInteriorCellLowIndex(), patch->getInteriorCellHighIndex()));
      }
    }
    
    //get weights on old_level
    vector<double> weights;
    getWeights(l,old_level,weights);
    
    double volume=0;
    double weight=0;
    //compute average cost per datapoint
    for(int r=0;r<old_level.size();r++)
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
    
    deque<Region> new_regions, dnew, dold(old_level.begin(),old_level.end());
    
    //create dnew to contain a subset of the new patches
    for(int p=0; p<newgrid->getLevel(l)->numPatches(); p++) 
    {
      if(p%d_myworld->size()==d_myworld->myrank())
      {
        const Patch* patch = newgrid->getLevel(l)->getPatch(p);
        dnew.push_back(Region(patch->getInteriorCellLowIndex(), patch->getInteriorCellHighIndex()));
      }
    }
    
    //compute difference
    new_regions=Region::difference(dnew, dold);
    
    //initialize weights 
    int i=0;
    for(deque<Region>::iterator it=new_regions.begin();it!=new_regions.end();it++)
    {
      IntVector low=it->getLow()/d_minPatchSize[l];
      IntVector high=it->getHigh()/d_minPatchSize[l];
      
      //loop through datapoints
      for(CellIterator iter(low,high); !iter.done(); iter++)
      {
        //add cost to current contribution
        costs[l][*iter].zoweight=average_cost;
        costs[l][*iter].foweight=average_cost;
        costs[l][*iter].soweight=average_cost;
        costs[l][*iter].toweight=average_cost;
      } //end cell iteration
    } //end region iteration
  }// end levels iteration
}
void CostProfiler::reset()
{
  for(int i=0;i<costs.size();i++)
  {
    costs[i].clear();
  }
  timesteps=0;
}
