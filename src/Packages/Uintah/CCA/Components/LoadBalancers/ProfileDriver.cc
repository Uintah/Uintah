#include <Packages/Uintah/CCA/Components/LoadBalancers/ProfileDriver.h>
using namespace Uintah;
using namespace SCIRun;
   
void ProfileDriver::addContribution(const vector<Region>& regions, const vector<int> &levels, double cost)
{
  int num_points=0;

  //compute number of data points
  for(unsigned int r=0;r<regions.size();r++)
  {
    const Region &region=regions[r];
    num_points+=region.getVolume();
  }
  //compute average cost per data point
  double average_cost=cost/num_points;

  for(unsigned int r=0;r<regions.size();r++)
  {
    Region region=regions[r];
    //for each datapoint
    for(CellIterator iter=CellIterator(region.getLow(),region.getHigh());!iter.done();iter++)
    {
      //add average contribution
      costs[levels[r]][*iter].current+=average_cost;
    }
  }
}

void ProfileDriver::finalizeContributions()
{
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

  //loop through each region
  for(unsigned int r=0;r<regions.size();r++)
  {
    //loop through datapoints
    for(CellIterator iter(regions[r].getLow(),regions[r].getHigh()); !iter.done(); iter++)
    {
      //search for point in the map
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

void ProfileDriver::initializeWeights(const vector<Region> &old_regions, const vector<Region> &new_regions, int l)
{
  //send in a list of regions and a level to initialize
    //compute average value of level
    //set value for those regions
  
  vector<double> weights;
  getWeights(l,old_regions,weights);
  
  double total_weight=0;
  double num_points=0;

  //compute total weight and number of points in the map
  for(unsigned int r=0;r<old_regions.size();r++)
  {
    total_weight+=weights[r];
    num_points+=old_regions[r].getVolume();
  }

  double average_weight=total_weight/num_points;

  //initialize the new regions to the average weight
  for(unsigned int r=0;r<new_regions.size();r++)
  {
    for(CellIterator iter(new_regions[r].getLow(),new_regions[r].getHigh());!iter.done();iter++)
    {
      costs[l][*iter].weight=average_weight;
    }
  }
}
void ProfileDriver::reset()
{
  for(int i=0;i<(int)costs.size();i++)
  {
    costs[i].clear();
  }
  timesteps=0;
}

void ProfileDriver::getMeasuredAndPredictedWeights(int l, const vector<Region> &regions, vector<double> &measured, vector<double> &predicted)
{
    vector<double> measured_local(regions.size(),0);
    vector<double> predicted_local(regions.size(),0);
    
    measured.assign(regions.size(),0);
    predicted.assign(regions.size(),0);
  
    //for each region
    for(unsigned int r=0;r<regions.size();r++)
    {
      //for each data point in region
      for(CellIterator iter(regions[r].getLow(),regions[r].getHigh());!iter.done();iter++)
      {
      
        map<IntVector,Contribution>::iterator it=costs[l].find(*iter);
        
        //if we have an entry in the map add it to our stats
        if(it!=costs[l].end())
        {
          measured_local[r]+=it->second.current;
          predicted_local[r]+=it->second.weight;
        }
      }
    }
    
    //reduce stats onto first processor
    if(d_myworld->size()>1)
    {
      MPI_Reduce(&measured_local[0],&measured[0],measured.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
      MPI_Reduce(&predicted_local[0],&predicted[0],predicted.size(),MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    }
 //
   
}
