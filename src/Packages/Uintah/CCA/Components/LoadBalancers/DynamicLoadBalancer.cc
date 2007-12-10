#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Time.h>
#include <Core/Exceptions/InternalError.h>

#include <iostream> // debug only
#include <stack>
#include <vector>
using namespace Uintah;
using namespace SCIRun;
using std::cerr;
static DebugStream doing("DynamicLoadBalancer_doing", false);
static DebugStream lb("DynamicLoadBalancer_lb", false);
static DebugStream dbg("DynamicLoadBalancer", false);
static DebugStream stats("LBStats",false);

DynamicLoadBalancer::DynamicLoadBalancer(const ProcessorGroup* myworld)
   : LoadBalancerCommon(myworld), sfc(myworld)
{
  d_lbInterval = 0.0;
  d_lastLbTime = 0.0;
  d_lbTimestepInterval = 0;
  d_lastLbTimestep = 0;
  d_checkAfterRestart = false;

  d_dynamicAlgorithm = patch_factor_lb;  
  d_do_AMR = false;
  d_pspec = 0;

  d_assignmentBasePatch = -1;
  d_oldAssignmentBasePatch = -1;
}

DynamicLoadBalancer::~DynamicLoadBalancer()
{
}

void DynamicLoadBalancer::collectParticlesForRegrid(const Grid* oldGrid, const vector<vector<Region> >& newGridRegions, 
                                                    vector<vector<double> >& costs)
{
  // collect particles from the old grid's patches onto processor 0 and then distribute them
  // (it's either this or do 2 consecutive load balances).  For now, it's safe to assume that
  // if there is a new level or a new patch there are no particles there.

  int num_procs = d_myworld->size();
  int myrank = d_myworld->myrank();
  int num_patches = 0;
  for (unsigned i = 0; i < newGridRegions.size(); i++)
    num_patches += newGridRegions[i].size();

  vector<int> recvcounts(num_procs,0); // init the counts to 0
  int totalsize = 0;

  DataWarehouse* dw = d_scheduler->get_dw(0);
  if (dw == 0)
    return;

  vector<PatchInfo> subpatchParticles;
  unsigned grid_index = 0;
  for(unsigned l=0;l<newGridRegions.size();l++){
    const vector<Region>& level = newGridRegions[l];
    for (unsigned r = 0; r < level.size(); r++, grid_index++) {
      const Region& region = level[r];;

      if (l >= (unsigned) oldGrid->numLevels()) {
        // new patch - no particles yet
        recvcounts[0]++;
        totalsize++;
        if (d_myworld->myrank() == 0) {
          PatchInfo pi(grid_index, 0);
          subpatchParticles.push_back(pi);
        }
        continue;
      }

      // find all the particles on old patches
      const LevelP oldLevel = oldGrid->getLevel(l);
      Level::selectType oldPatches;
      oldLevel->selectPatches(region.getLow(), region.getHigh(), oldPatches);

      if (oldPatches.size() == 0) {
        recvcounts[0]++;
        totalsize++;
        if (d_myworld->myrank() == 0) {
          PatchInfo pi(grid_index, 0);
          subpatchParticles.push_back(pi);
        }
        continue;
      }

      for (int i = 0; i < oldPatches.size(); i++) {
        const Patch* oldPatch = oldPatches[i];

        recvcounts[d_processorAssignment[oldPatch->getGridIndex()]]++;
        totalsize++;
        if (d_processorAssignment[oldPatch->getGridIndex()] == myrank) {
          IntVector low, high;
          //loop through the materials and add up the particles
          // the main difference between this and the above portion is that we need to grab the portion of the patch
          // that is intersected by the other patch
          low = Max(region.getLow(), oldPatch->getLowIndex());
          high = Min(region.getHigh(), oldPatch->getHighIndex());

          int thisPatchParticles = 0;
          if (dw) {
            //loop through the materials and add up the particles
            //   go through all materials since getting an MPMMaterial correctly would depend on MPM
            for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
              ParticleSubset* psubset = 0;
              if (dw->haveParticleSubset(m, oldPatch, low, high))
                psubset = dw->getParticleSubset(m, oldPatch, low, high);
              if (psubset)
                thisPatchParticles += psubset->numParticles();
            }
          }
          PatchInfo p(grid_index, thisPatchParticles);
          subpatchParticles.push_back(p);
        }
      }
    }
  }

  vector<int> num_particles(num_patches, 0);

  if (d_myworld->size() > 1) {
    //construct a mpi datatype for the PatchInfo
    MPI_Datatype particletype;
    MPI_Type_contiguous(2, MPI_INT, &particletype);
    MPI_Type_commit(&particletype);

    vector<PatchInfo> recvbuf(totalsize);
    vector<int> displs(num_procs,0);
    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

    MPI_Gatherv(&subpatchParticles[0], recvcounts[d_myworld->myrank()], particletype, &recvbuf[0],
                &recvcounts[0], &displs[0], particletype, 0, d_myworld->getComm());

    if ( d_myworld->myrank() == 0) {
      for (unsigned i = 0; i < recvbuf.size(); i++) {
        PatchInfo& spi = recvbuf[i];
        num_particles[spi.id] += spi.numParticles;
      }
    }
    // combine all the subpatches results
    MPI_Bcast(&num_particles[0], num_particles.size(), MPI_INT,0,d_myworld->getComm());
    MPI_Type_free(&particletype);
  }
  else {
    for (unsigned i = 0; i < subpatchParticles.size(); i++) {
      PatchInfo& spi = subpatchParticles[i];
      num_particles[spi.id] += spi.numParticles;
    }
  }

  if (dbg.active() && d_myworld->myrank() == 0) {
    for (unsigned i = 0; i < num_particles.size(); i++) {
      dbg << d_myworld->myrank() << "  Post gather index " << i << ": " << " numP : " << num_particles[i] << endl;
    }
  }
  // add the number of particles to the cost array
  unsigned cost_level = 0;
  unsigned cost_index = 0;
  for (unsigned i = 0; i < num_particles.size(); i++, cost_index++) {
    if (costs[cost_level].size() <= cost_index) {
      cost_index = 0;
      cost_level++;
    }
    costs[cost_level][cost_index] += num_particles[i]*d_particleCost;
  }
  // make sure that all regions got covered
  ASSERTEQ(cost_level, costs.size()-1);
  ASSERTEQ(cost_index, costs[cost_level].size());
}


void DynamicLoadBalancer::collectParticles(const Grid* grid, vector<vector<double> >& costs)
{
  if (d_processorAssignment.size() == 0)
    return; // if we haven't been through the LB yet, don't try this.

  if (d_myworld->myrank() == 0)
    dbg << " DLB::collectParticles\n";

  int num_patches = 0;
  for (int i = 0; i < grid->numLevels(); i++)
    num_patches += grid->getLevel(i)->numPatches();

  int num_procs = d_myworld->size();
  int myrank = d_myworld->myrank();
  // get how many particles were each patch had at the end of the last timestep
  //   gather from each proc - based on the last location

  DataWarehouse* dw = d_scheduler->get_dw(0);
  if (dw == 0)
    return;

  vector<PatchInfo> particleList;
  vector<int> num_particles(num_patches, 0);

  // find out how many particles per patch, and store that number
  // along with the patch number in particleList
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    for (Level::const_patchIterator iter = level->patchesBegin(); 
        iter != level->patchesEnd(); iter++) {
      Patch *patch = *iter;
      int id = patch->getGridIndex();
      if (d_processorAssignment[id] != myrank)
        continue;
      
      int thisPatchParticles = 0;

      if (dw) {
        //loop through the materials and add up the particles
        //   go through all materials since getting an MPMMaterial correctly would depend on MPM
        for (int m = 0; m < d_sharedState->getNumMatls(); m++) {
          if (dw->haveParticleSubset(m, patch))
            thisPatchParticles += dw->getParticleSubset(m, patch)->numParticles();
        }
      }
      // add to particle list
      PatchInfo p(id,thisPatchParticles);
      particleList.push_back(p);
      dbg << "  Pre gather " << id << " part: " << thisPatchParticles << endl;
    }
  }

  if (d_myworld->size() > 1) {
    //construct a mpi datatype for the PatchInfo
    MPI_Datatype particletype;
    MPI_Type_contiguous(2, MPI_INT, &particletype);
    MPI_Type_commit(&particletype);
    vector<int> displs(num_procs, 0);
    vector<int> recvcounts(num_procs,0); // init the counts to 0

    // order patches by processor #, to determine recvcounts easily
    vector<int> sorted_processorAssignment = d_processorAssignment;
    sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());
    vector<PatchInfo> all_particles(num_patches);

    for (int i = 0; i < (int)d_processorAssignment.size(); i++) {
      recvcounts[sorted_processorAssignment[i]]++;
    }

    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

    MPI_Allgatherv(&particleList[0], particleList.size(), particletype,
                    &all_particles[0], &recvcounts[0], &displs[0], particletype,
                    d_myworld->getComm());
    MPI_Type_free(&particletype);

    if (dbg.active() && d_myworld->myrank() == 0) {
      for (unsigned i = 0; i < all_particles.size(); i++) {
        PatchInfo& pi = all_particles[i];
        dbg << d_myworld->myrank() << "  Post gather index " << i << ": " << pi.id << " numP : " << pi.numParticles << endl;
      }
    }

    for (int i = 0; i < num_patches; i++) {
      num_particles[all_particles[i].id] = all_particles[i].numParticles;
    }
  }
  else {
    for (int i = 0; i < num_patches; i++)
      num_particles[particleList[i].id] = particleList[i].numParticles;
  }

  // add the number of particles to the cost array
  unsigned cost_level = 0;
  unsigned cost_index = 0;
  for (unsigned i = 0; i < num_particles.size(); i++, cost_index++) {
    if (costs[cost_level].size() <= cost_index) {
      cost_index = 0;
      cost_level++;
    }
    costs[cost_level][cost_index] += num_particles[i]*d_particleCost;
  }
  // make sure that all regions got covered
  ASSERTEQ(cost_level, costs.size()-1);
  ASSERTEQ(cost_index, costs[cost_level].size());

}

void DynamicLoadBalancer::useSFC(const LevelP& level, int* order)
{
  vector<DistributedIndex> indices; //output
  vector<double> positions;
  
  vector<int> recvcounts(d_myworld->size(), 0);

  //this should be removed when dimensions in shared state is done
  int dim=d_sharedState->getNumDims();
  int *dimensions=d_sharedState->getActiveDims();

  IntVector min_patch_size(INT_MAX,INT_MAX,INT_MAX);  

  // get the overall range in all dimensions from all patches
  IntVector high(INT_MIN,INT_MIN,INT_MIN);
  IntVector low(INT_MAX,INT_MAX,INT_MAX);
  
  for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) 
  {
    const Patch* patch = *iter;
   
    //calculate patchset bounds
    high = Max(high, patch->getInteriorCellHighIndex());
    low = Min(low, patch->getInteriorCellLowIndex());
    
    //calculate minimum patch size
    IntVector size=patch->getInteriorCellHighIndex()-patch->getInteriorCellLowIndex();
    min_patch_size=min(min_patch_size,size);
    
    //create positions vector
    int proc = (patch->getLevelIndex()*d_myworld->size())/level->numPatches();
    if(d_myworld->myrank()==proc)
    {
      Vector point=(patch->getInteriorCellLowIndex()+patch->getInteriorCellHighIndex()).asVector()/2.0;
      for(int d=0;d<dim;d++)
      {
        positions.push_back(point[dimensions[d]]);
      }
    }
  }
  
  //patchset dimensions
  IntVector range = high-low;
  
  //center of patchset
  Vector center=(high+low).asVector()/2.0;
 
  double r[3]={range[dimensions[0]],range[dimensions[1]],range[dimensions[2]]};
  double c[3]={center[dimensions[0]],center[dimensions[1]],center[dimensions[2]]};
  double delta[3]={min_patch_size[dimensions[0]],min_patch_size[dimensions[1]],min_patch_size[dimensions[2]]};

  //create SFC
  sfc.SetLocalSize(positions.size()/dim);
  sfc.SetDimensions(r);
  sfc.SetCenter(c);
  sfc.SetRefinementsByDelta(delta); 
  sfc.SetLocations(&positions);
  sfc.SetOutputVector(&indices);
  sfc.GenerateCurve();

  vector<int> displs(d_myworld->size(), 0);
  if(d_myworld->size()>1)  
  {
    int rsize=indices.size();
     
    //gather recv counts
    MPI_Allgather(&rsize,1,MPI_INT,&recvcounts[0],1,MPI_INT,d_myworld->getComm());
    
    
    for (unsigned i = 0; i < recvcounts.size(); i++)
    {
       recvcounts[i]*=sizeof(DistributedIndex);
    }
    for (unsigned i = 1; i < recvcounts.size(); i++)
    {
       displs[i] = recvcounts[i-1] + displs[i-1];
    }

    vector<DistributedIndex> rbuf(level->numPatches());

    //gather curve
    MPI_Allgatherv(&indices[0], recvcounts[d_myworld->myrank()], MPI_BYTE, &rbuf[0], &recvcounts[0], 
                   &displs[0], MPI_BYTE, d_myworld->getComm());

    indices.swap(rbuf);
  }

  //convert distributed indices to normal indices
  for(unsigned int i=0;i<indices.size();i++)
  {
    DistributedIndex di=indices[i];
    //order[i]=displs[di.p]/sizeof(DistributedIndex)+di.i;
    order[i]=(int)ceil((float)di.p*level->numPatches()/d_myworld->size())+di.i;
  }
 
  /*
  if(d_myworld->myrank()==0)
  {
    cout << "Warning checking SFC correctness\n";
  }
  for (int i = 0; i < level->numPatches(); i++) 
  {
    for (int j = i+1; j < level->numPatches(); j++)
    {
      if (order[i] == order[j]) 
      {
        cout << "Rank:" << d_myworld->myrank() <<  ":   ALERT!!!!!! index done twice: index " << i << " has the same value as index " << j << " " << order[i] << endl;
        throw InternalError("SFC unsuccessful", __FILE__, __LINE__);
      }
    }
  }
  */
}


bool DynamicLoadBalancer::assignPatchesFactor(const GridP& grid, bool force)
{
  doing << d_myworld->myrank() << "   APF\n";
  double time = Time::currentSeconds();

  vector<vector<double> > patch_costs;

  int num_procs = d_myworld->size();
  DataWarehouse* olddw = d_scheduler->get_dw(0);

  //get costs of the patches for load balancing
  getCosts(grid.get_rep(), patch_costs);
  
  int level_offset=0;
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    int num_patches = level->numPatches();
    vector<int> order(num_patches);
    double total_cost = 0;

    for (unsigned i = 0; i < patch_costs[l].size(); i++)
      total_cost += patch_costs[l][i];

    if (d_doSpaceCurve) {
      //cout << d_myworld->myrank() << "   Doing SFC level " << l << endl;
      useSFC(level, &order[0]);
    }
#if 1
    double avgCostPerProc = 0;
    avgCostPerProc = total_cost / num_procs;

    int currentProc = 0;
    double currentProcCost = 0;
    
    for (int p = 0; p < num_patches; p++) {
      int index;
      if (d_doSpaceCurve) {
        index = order[p];
        //cout << d_myworld->myrank() << ": mapping " << order[p] << " to " << index << " startingPatch:" << startingPatch << endl;
      }
      else {
        // not attempting space-filling curve
        index = p;
      }
      
      // assign the patch to a processor.  When we advance procs,
      // re-update the cost, so we use all procs (and don't go over)
      double patchCost = patch_costs[l][index];
      double notakeimb=fabs(currentProcCost-avgCostPerProc);
      double takeimb=fabs(currentProcCost+patchCost-avgCostPerProc);

      if (notakeimb<takeimb) {
        // move to next proc and add this patch
        currentProc++;
        d_tempAssignment[level_offset+index] = currentProc;
        //update average (this ensures we don't over/under fill to much)
        total_cost -= currentProcCost;
        avgCostPerProc = total_cost / (num_procs-currentProc);
        currentProcCost = patchCost;
      }
      else {
        // add patch to currentProc
        ASSERTRANGE(currentProc,0,d_myworld->size());
        d_tempAssignment[level_offset+index] = currentProc;
        currentProcCost += patchCost;
      }
      if (d_myworld->myrank() == 0)
        lb << "Level " << l << " Patch " << index << "-> proc " << currentProc 
            << " PatchCost: " << patchCost << ", ProcCost: "
            << currentProcCost << " group cost " << total_cost << "  avg cost " << avgCostPerProc << endl;
    }
#else
    //partition patches recursivly into halves until I have my patchset
    int remainingProcessors=num_procs;
    int remainingPatches=num_patches;
    int remainingCost=total_cost;
    int startingPatch=0;
    int startingProc=0;

    while(remainingProcessors>1 && remainingPatches>0)
    {
      //compute a split point where both sides have approximatly the same cost
      int halfProc=remainingProcessors/2+remainingProcessors%2;
      double halfCost=remainingCost/2;
      double cost=0;
      int p=0;
      double takeimb=fabs(cost+patch_costs[l][order[startingPatch+p]]-halfCost);
      double notakeimb=fabs(cost-halfCost);

      //if we do not have enough patches on the left side for all processors 
      //  or if there is enough patches for the right side
      //     and taking the patch causes less imbalance than not taking it
      while(p<halfProc || remainingPatches-p>remainingProcessors-halfProc && notakeimb>takeimb)
      {
        //assign this patch to left side
        cost+=patch_costs[l][order[startingPatch+p++]];
        
        //break out if there are no more patches
        if(p==remainingPatches)
           break;
        
        //update imbalance
        takeimb=fabs(cost+patch_costs[l][order[startingPatch+p]]-halfCost);
        notakeimb=fabs(cost-halfCost);
      }

      if(lb.active())
      {
        lb << d_myworld->myrank() << "  RemainingPatches:" << remainingPatches << " RemainingCost:" << remainingCost << " RemainingProcs:" << remainingProcessors << " cost:" << cost
              << " halfCost:" << halfCost << " halfProc:" << halfProc << " startProc:" << startingProc << " startingPatch:" << startingPatch
              << endl;
      }
      if(d_myworld->myrank()<startingProc+halfProc)
      {
        //continue on left side
        remainingProcessors=halfProc;
        remainingPatches=p;
        remainingCost=cost;
        //startingProc remains the same
        //startingPatch remains the same
      }
      else
      {
        //continue on right side
        remainingProcessors-=halfProc;
        remainingPatches-=p;
        remainingCost-=cost;
        startingProc+=halfProc;
        startingPatch+=p;
      }
    }
    
    if(lb.active() )
      lb << d_myworld->myrank() << " Level:" << l << " Final Cost:" << remainingCost  << " startingPatch:" << startingPatch << endl;
    
    vector<int> startPatch(num_procs+1,-999999);
    startPatch[num_procs]=num_patches;
    
    ASSERT((startingProc==d_myworld->myrank() && remainingProcessors==1) || remainingPatches==0);
    //allgather patchsets
    MPI_Allgather(&startingPatch,1,MPI_INT,&startPatch[0],1,MPI_INT,d_myworld->getComm());

    //set d_tempAssignment array
    int processor=0;
    for(int p=0;p<num_patches;p++)
    {
      while(p>=startPatch[processor+1])
         processor++;
    
      if(lb.active() && d_myworld->myrank()==0 )
        lb << "On Level:" << l << " Patch:" << order[p] << " assigning to:" << processor << " cost:" << patch_costs[l][order[p]] << endl;
      
      d_tempAssignment[level_offset+order[p]] = processor;
    }
    
#endif
    
    if(stats.active() && d_myworld->myrank()==0)
    {
      //calculate lb stats:
      int totalCost=0;
      vector<int> procCosts(num_procs,0);
      for(int p=0;p<num_patches;p++)
      {
        totalCost+=patch_costs[l][p];
        procCosts[d_tempAssignment[level_offset+p]]+=patch_costs[l][p];
      }
      
      double meanCost=totalCost/num_procs;
      double minCost=procCosts[0];
      double maxCost=procCosts[0];
      for(int p=0;p<num_procs;p++)
      {
        if(minCost>procCosts[p])
           minCost=procCosts[p];
        else if(maxCost<procCosts[p])
           maxCost=procCosts[p];
      }
      double stdCost=0;
      double sumXsquared=0;
      for(int p=0;p<num_procs;p++)
      {
        double diffSquared=procCosts[p]-meanCost;
        diffSquared*=diffSquared;
        sumXsquared+=diffSquared;
      }
      stdCost=sqrt(sumXsquared/num_procs);

      stats << "LoadBalance Stats level(" << l << "):"  << " Mean:" << meanCost << " relstdev:" << stdCost/meanCost << " Min:" << minCost << " Max:" << maxCost << endl;
    }  
    
    level_offset+=num_patches;
  }

  bool doLoadBalancing = force || thresholdExceeded(patch_costs);

  time = Time::currentSeconds() - time;
  if (d_myworld->myrank() == 0)
    dbg << " Time to LB: " << time << endl;
  doing << d_myworld->myrank() << "   APF END\n";
  return doLoadBalancing;
}

bool DynamicLoadBalancer::thresholdExceeded(const vector<vector<double> >& patch_costs)
{
  // add up the costs each processor for the current assignment
  // and for the temp assignment, then calculate the standard deviation
  // for both.  If (curStdDev / tmpStdDev) > threshold, return true,
  // and have possiblyDynamicallyRelocate change the load balancing
  
  int num_procs = d_myworld->size();
  vector<double> currentProcCosts(num_procs);
  vector<double> tempProcCosts(num_procs);

  // add the number of particles to the cost array
  unsigned cost_level = 0;
  unsigned cost_index = 0;
  for (unsigned i = 0; i < d_tempAssignment.size(); i++, cost_index++) {
    if (patch_costs[cost_level].size() <= cost_index) {
      cost_index = 0;
      cost_level++;
    }
    currentProcCosts[d_processorAssignment[i]] += patch_costs[cost_level][cost_index];
    tempProcCosts[d_tempAssignment[i]] += patch_costs[cost_level][cost_index];
  }
  
  // use (avg/max) for lb%
  double avg_current = 0;
  double max_current = 0;
  double avg_temp = 0;
  double max_temp = 0;

  for (int i = 0; i < d_myworld->size(); i++) {
    if (currentProcCosts[i] > max_current) max_current = currentProcCosts[i];
    if (tempProcCosts[i] > max_temp) max_temp = tempProcCosts[i];
    avg_current += currentProcCosts[i];
    avg_temp += tempProcCosts[i];
  }
  avg_current /= d_myworld->size();
  avg_temp /= d_myworld->size();
  
  double curLB = avg_current/max_current;
  double tmpLB = avg_temp/max_temp;

  if (d_myworld->myrank() == 0)
    dbg << "CurrLB: " << curLB << " tmp " << tmpLB
        << " threshold: " << tmpLB-curLB << " minT " << d_lbThreshold << endl;

  // if tmp - cur is positive, it is an improvement
  if (tmpLB - curLB > d_lbThreshold)
    return true;
  else
    return false;
}


bool DynamicLoadBalancer::assignPatchesRandom(const GridP&, bool force)
{
  // this assigns patches in a random form - every time we re-load balance
  // We get a random seed on the first proc and send it out (so all procs
  // generate the same random numbers), and assign the patches accordingly
  // Not a good load balancer - useful for performance comparisons
  // because we should be able to come up with one better than this

  int seed;

  if (d_myworld->myrank() == 0)
    seed = (int) Time::currentSeconds();
  
  MPI_Bcast(&seed, 1, MPI_INT,0,d_myworld->getComm());

  srand(seed);

  int num_procs = d_myworld->size();
  int num_patches = (int)d_processorAssignment.size();

  vector<int> proc_record(num_procs,0);
  int max_ppp = num_patches / num_procs;

  for (int i = 0; i < num_patches; i++) {
    int proc = (int) (((float) rand()) / RAND_MAX * num_procs);
    int newproc = proc;

    // only allow so many patches per proc.  Linear probe if necessary,
    // but make sure every proc has some work to do
    while (proc_record[newproc] >= max_ppp) {
      newproc++;
      if (newproc >= num_procs) newproc = 0;
      if (proc == newproc) 
        // let each proc have more - we've been through all procs
        max_ppp++;
    }
    proc_record[newproc]++;
    d_tempAssignment[i] = newproc;
  }
  return true;
}

bool DynamicLoadBalancer::assignPatchesCyclic(const GridP&, bool force)
{
  // this assigns patches in a cyclic form - every time we re-load balance
  // we move each patch up one proc - this obviously isn't a very good
  // lb technique, but it tests its capabilities pretty well.

  int num_procs = d_myworld->size();
  for (unsigned i = 0; i < d_tempAssignment.size(); i++) {
    d_tempAssignment[i] = (d_processorAssignment[i] + 1 ) % num_procs;
  }
  return true;
}


int
DynamicLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch)
{
  ASSERTRANGE(patch->getID(), d_assignmentBasePatch, d_assignmentBasePatch + (int) d_processorAssignment.size());
  int proc = d_processorAssignment[patch->getRealPatch()->getGridIndex()];
  //cout << group->myrank() << " Requesting patch " << patch->getGridIndex()
  //   << " which is stored on processor " << proc << endl;
  //int proc = (patch->getLevelIndex()*num_procs)/patch->getLevel()->numPatches();
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}

int
DynamicLoadBalancer::getOldProcessorAssignment(const VarLabel* var, 
						const Patch* patch, 
                                                const int /*matl*/)
{

  if (var && var->typeDescription()->isReductionVariable()) {
    return d_myworld->myrank();
  }

  // on an initial-regrid-timestep, this will get called from createNeighborhood
  // and can have a patch with a higher index than we have
  if ((int)patch->getID() < d_oldAssignmentBasePatch || patch->getID() >= d_oldAssignmentBasePatch + (int)d_oldAssignment.size())
    return -patch->getID();
  
  if (patch->getGridIndex() >= (int) d_oldAssignment.size())
    return -999;

  int proc = d_oldAssignment[patch->getGridIndex()];
  //cout << d_myworld->myrank() << " Requesting patch " <<patch->getGridIndex()
  //   << " which *used to be* stored on processor " << proc << endl;
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
  //return getPatchwiseProcessorAssignment(patch, d_myworld);
}

bool 
DynamicLoadBalancer::needRecompile(double /*time*/, double /*delt*/, 
				    const GridP& grid)
{
  double time = d_sharedState->getElapsedTime();
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();

  bool do_check = false;

  if (d_lbTimestepInterval != 0 && timestep >= d_lastLbTimestep + d_lbTimestepInterval) {
    d_lastLbTimestep = timestep;
    do_check = true;
  }
  else if (d_lbInterval != 0 && time >= d_lastLbTime + d_lbInterval) {
    d_lastLbTime = time;
    do_check = true;
  }
  else if ((time == 0 && d_collectParticles == true) || d_checkAfterRestart) {
    // do AFTER initialization timestep too (no matter how much init regridding),
    // so we can compensate for new particles
    do_check = true;
    d_checkAfterRestart = false;
  }

  if (dbg.active() && d_myworld->myrank() == 0)
    dbg << d_myworld->myrank() << " DLB::NeedRecompile: check=" << do_check << " ts: " << timestep << " " << d_lbTimestepInterval << " t " << time << " " << d_lbInterval << " last: " << d_lastLbTimestep << " " << d_lastLbTime << endl;

  // if it determines we need to re-load-balance, recompile
  if (do_check && possiblyDynamicallyReallocate(grid, check)) {
    doing << d_myworld->myrank() << " PLB - scheduling recompile " <<endl;
    return true;
  }
  else {
    d_oldAssignment = d_processorAssignment;
    d_oldAssignmentBasePatch = d_assignmentBasePatch;
    doing << d_myworld->myrank() << " PLB - NOT scheduling recompile " <<endl;
    return false;
  }
} 

void
DynamicLoadBalancer::restartInitialize(DataArchive* archive, int time_index, ProblemSpecP& pspec, string tsurl, const GridP& grid)
{
  // here we need to grab the uda data to reassign patch dat  a to the 
  // processor that will get the data
  int num_patches = 0;
  const Patch* first_patch = *(grid->getLevel(0)->patchesBegin());
  int startingID = first_patch->getID();
  int prevNumProcs = 0;

  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    num_patches += level->numPatches();
  }

  d_processorAssignment.resize(num_patches);
  d_assignmentBasePatch = startingID;
  for (unsigned i = 0; i < d_processorAssignment.size(); i++)
    d_processorAssignment[i]= -1;

  if (archive->queryPatchwiseProcessor(first_patch, time_index) != -1) {
    // for uda 1.1 - if proc is saved with the patches
    for(int l=0;l<grid->numLevels();l++){
      const LevelP& level = grid->getLevel(l);
      for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) {
        d_processorAssignment[(*iter)->getID()-startingID] = archive->queryPatchwiseProcessor(*iter, time_index) % d_myworld->size();
      }
    }
  } // end queryPatchwiseProcessor
  else {
    // before uda 1.1
    // strip off the timestep.xml
    string dir = tsurl.substr(0, tsurl.find_last_of('/')+1);

    ASSERT(pspec != 0);
    ProblemSpecP datanode = pspec->findBlock("Data");
    if(datanode == 0)
      throw InternalError("Cannot find Data in timestep", __FILE__, __LINE__);
    for(ProblemSpecP n = datanode->getFirstChild(); n != 0; 
        n=n->getNextSibling()){
      if(n->getNodeName() == "Datafile") {
        map<string,string> attributes;
        n->getAttributes(attributes);
        string proc = attributes["proc"];
        if (proc != "") {
          int procnum = atoi(proc.c_str());
          if (procnum+1 > prevNumProcs)
            prevNumProcs = procnum+1;
          string datafile = attributes["href"];
          if(datafile == "")
            throw InternalError("timestep href not found", __FILE__, __LINE__);
          
          string dataxml = dir + datafile;
          // open the datafiles
          ProblemSpecReader psr(dataxml);

          ProblemSpecP dataDoc = psr.readInputFile();
          if (!dataDoc)
            throw InternalError("Cannot open data file", __FILE__, __LINE__);
          for(ProblemSpecP r = dataDoc->getFirstChild(); r != 0; r=r->getNextSibling()){
            if(r->getNodeName() == "Variable") {
              int patchid;
              if(!r->get("patch", patchid) && !r->get("region", patchid))
                throw InternalError("Cannot get patch id", __FILE__, __LINE__);
              if (d_processorAssignment[patchid-startingID] == -1) {
                // assign the patch to the processor
                // use the grid index
                d_processorAssignment[patchid - startingID] = procnum % d_myworld->size();
              }
            }
          }            
          
        }
      }
    }
  } // end else...
  for (unsigned i = 0; i < d_processorAssignment.size(); i++) {
    if (d_processorAssignment[i] == -1)
      cout << "index " << i << " == -1\n";
    ASSERT(d_processorAssignment[i] != -1);
  }
  d_oldAssignment = d_processorAssignment;
  d_oldAssignmentBasePatch = d_assignmentBasePatch;

  if (prevNumProcs != d_myworld->size() || d_outputNthProc > 1) {
    if (d_myworld->myrank() == 0) dbg << "  Original run had " << prevNumProcs << ", this has " << d_myworld->size() << endl;
    d_checkAfterRestart = true;
  }

  if (d_myworld->myrank() == 0) {
    int startPatch = (*grid->getLevel(0)->patchesBegin())->getID();
    dbg << d_myworld->myrank() << " check after restart: " << d_checkAfterRestart << "\n";
    if (lb.active()) {
      for (unsigned i = 0; i < d_processorAssignment.size(); i++) {
        lb <<d_myworld-> myrank() << " patch " << i << " (real " << i+startPatch << ") -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") - " << d_processorAssignment.size() << ' ' << d_oldAssignment.size() << "\n";
      }
    }
  }
}

void DynamicLoadBalancer::sortPatches(vector<Region> &patches, vector<double> &costs)
{
  if(d_doSpaceCurve) //reorder by sfc
  {
    int *dimensions=d_sharedState->getActiveDims();
    vector<double> positions;
    vector<DistributedIndex> indices;
    int mode;
    unsigned int num_procs=d_myworld->size();
    int myRank=d_myworld->myrank();

    //decide to run serial or parallel
    if(num_procs==1 || patches.size()<num_procs) //compute in serial if the number of patches is small
    {
      mode=SERIAL;
      num_procs=1;
      myRank=0;
    }
    else 
    {
      mode=PARALLEL;
    }
    //create locations array on each processor
    for(unsigned int p=0;p<patches.size();p++)
    {
      int proc = (p*num_procs)/patches.size();
      if(proc==myRank)
      {
        Vector center=(patches[p].getHigh()+patches[p].getLow()).asVector()/2;
        for(int d=0;d<d_sharedState->getNumDims();d++)
        {
          positions.push_back(center[dimensions[d]]);
        }
      }
    }
    //generate curve
    sfc.SetOutputVector(&indices);
    sfc.SetLocations(&positions);
    sfc.SetLocalSize(positions.size()/d_sharedState->getNumDims());
    sfc.GenerateCurve(mode);

    vector<int> displs(num_procs, 0);
    vector<int> recvcounts(num_procs,0); // init the counts to 0
    //gather curve onto each processor
    if(mode==PARALLEL) 
    {
      //gather curve
      int rsize=indices.size();
      //gather recieve size
      MPI_Allgather(&rsize,1,MPI_INT,&recvcounts[0],1,MPI_INT,d_myworld->getComm());

      for (unsigned i = 0; i < recvcounts.size(); i++)
      {
        recvcounts[i]*=sizeof(DistributedIndex);
      }
      for (unsigned i = 1; i < recvcounts.size(); i++)
      {
        displs[i] = recvcounts[i-1] + displs[i-1];
      }
     
      vector<DistributedIndex> rbuf(patches.size());

      //gather curve
      MPI_Allgatherv(&indices[0], recvcounts[myRank], MPI_BYTE, &rbuf[0], &recvcounts[0],
                    &displs[0], MPI_BYTE, d_myworld->getComm());
      
      indices.swap(rbuf);
    }
    
    //reorder patches
    vector<Region> reorderedPatches(patches.size());
    vector<double> reorderedCosts(patches.size());
    for(unsigned int i=0;i<indices.size();i++)
    {
        DistributedIndex di=indices[i];
        //index is equal to the diplacement for the processor converted to struct size plus the local index
        //int index=displs[di.p]/sizeof(DistributedIndex)+di.i;
        int index=(int)ceil((float)di.p*patches.size()/d_myworld->size())+di.i;
        reorderedPatches[i]=patches[index];
        reorderedCosts[i]=costs[index];
    }
    patches=reorderedPatches;
    costs=reorderedCosts;
  }
  //random? cyclicic?
}

// There are 2 cases which call this function:
// 1) from possiblyDynamicallyReallocate, not regrid
// 2) from possiblyDynamicallyReallocate, after regrid
// 
//    patches only matters during a regrid.  In this case, 'grid' is the old grid.
void DynamicLoadBalancer::getCosts(const Grid* grid, vector<vector<double> >&costs)
{
  costs.clear();
  costs.resize(grid->numLevels());
  for (int l = 0; l < grid->numLevels(); l++) 
  {
    for(int p = 0; p < grid->getLevel(l)->numPatches(); p++)
      {
        costs[l].push_back(d_patchCost+grid->getLevel(l)->getPatch(p)->getVolume()*d_cellCost);
      }
    }
    if (d_collectParticles && d_scheduler->get_dw(0) != 0) 
    {
      collectParticles(grid, costs);
    }
  
  if (dbg.active() && d_myworld->myrank() == 0) {
    for (unsigned l = 0; l < costs.size(); l++)
      for (unsigned p = 0; p < costs[l].size(); p++)
        dbg << "L"  << l << " P " << p << " cost " << costs[l][p] << endl;
  }
}

bool DynamicLoadBalancer::possiblyDynamicallyReallocate(const GridP& grid, int state)
{
  TAU_PROFILE("DynamicLoadBalancer::possiblyDynamicallyReallocate()", " ", TAU_USER);

  if (d_myworld->myrank() == 0)
    dbg << d_myworld->myrank() << " In DLB, state " << state << endl;

  double start = Time::currentSeconds();

  bool changed = false;
  bool force = false;
  // don't do on a restart unless procs changed between runs.  For restarts, this is 
  // called mainly to update the perProc Patch sets (at the bottom)
  if (state != restart) {
    if (state != check) {
      force = true;
      if (d_lbTimestepInterval != 0) {
        d_lastLbTimestep = d_sharedState->getCurrentTopLevelTimeStep();
      }
      else if (d_lbInterval != 0) {
        d_lastLbTime = d_sharedState->getElapsedTime();
      }
    }
    d_oldAssignment = d_processorAssignment;
    d_oldAssignmentBasePatch = d_assignmentBasePatch;
    
    bool dynamicAllocate = false;
    //temp assignment can be set if the regridder has already called the load balancer
    if(d_tempAssignment.empty())
    {
      int num_patches = 0;
      for(int l=0;l<grid->numLevels();l++){
        const LevelP& level = grid->getLevel(l);
        num_patches += level->numPatches();
      }
    
      d_tempAssignment.resize(num_patches);
      if (d_myworld->myrank() == 0)
        doing << d_myworld->myrank() << "  Checking whether we need to LB\n";
      switch (d_dynamicAlgorithm) {
        case patch_factor_lb:  dynamicAllocate = assignPatchesFactor(grid, force); break;
        case cyclic_lb:        dynamicAllocate = assignPatchesCyclic(grid, force); break;
        case random_lb:        dynamicAllocate = assignPatchesRandom(grid, force); break;
      }
    }
    else  //regridder has called dynamic load balancer so we must dynamically Allocate
    {
      dynamicAllocate=true;
    }

    if (dynamicAllocate || state != check) {
      //d_oldAssignment = d_processorAssignment;
      changed = true;
      d_processorAssignment = d_tempAssignment;
      d_assignmentBasePatch = (*grid->getLevel(0)->patchesBegin())->getID();

      if (state == init) {
        // set it up so the old and new are in same place
        d_oldAssignment = d_processorAssignment;
        d_oldAssignmentBasePatch = d_assignmentBasePatch;
      }
      if (lb.active()) {
        int num_procs = d_myworld->size();
        int myrank = d_myworld->myrank();
        if (myrank == 0) {
          LevelP curLevel = grid->getLevel(0);
          Level::const_patchIterator iter = curLevel->patchesBegin();
          lb << "  Changing the Load Balance\n";
          for (unsigned int i = 0; i < d_processorAssignment.size(); i++) {
            lb << myrank << " patch " << i << " (real " << (*iter)->getID() << ") -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") patch size: "  << (*iter)->getGridIndex() << " " << ((*iter)->getHighIndex() - (*iter)->getLowIndex()) << "\n";
            IntVector range = ((*iter)->getHighIndex() - (*iter)->getLowIndex());
            iter++;
            if (iter == curLevel->patchesEnd() && i+1 < d_processorAssignment.size()) {
              curLevel = curLevel->getFinerLevel();
              iter = curLevel->patchesBegin();
            }
          }
        }
      }
    }
  }
  d_tempAssignment.resize(0);
  // this must be called here (it creates the new per-proc patch sets) even if DLB does nothing.  Don't move or return earlier.
  LoadBalancerCommon::possiblyDynamicallyReallocate(grid, (changed || state == restart) ? regrid : check);
  d_sharedState->loadbalancerTime += Time::currentSeconds() - start;
  return changed;
}

void
DynamicLoadBalancer::problemSetup(ProblemSpecP& pspec, SimulationStateP& state)
{
  LoadBalancerCommon::problemSetup(pspec, state);

  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  string dynamicAlgo;
  double interval = 0;
  int timestepInterval = 0;
  double threshold = 0.0;
  bool spaceCurve = false;
  
  if (p != 0) {
    // if we have DLB, we know the entry exists in the input file...
    if(!p->get("timestepInterval", timestepInterval))
      timestepInterval = 0;
    if (timestepInterval != 0 && !p->get("interval", interval))
      interval = 0.0; // default
    p->getWithDefault("dynamicAlgorithm", dynamicAlgo, "static");
    p->getWithDefault("cellCost", d_cellCost, 2);
    p->getWithDefault("particleCost", d_particleCost, 1);
    p->getWithDefault("patchCost", d_patchCost, 0);
    p->getWithDefault("gainThreshold", threshold, 0.0);
    p->getWithDefault("doSpaceCurve", spaceCurve, false);
  }

  if (dynamicAlgo == "cyclic")
    d_dynamicAlgorithm = cyclic_lb;
  else if (dynamicAlgo == "random")
    d_dynamicAlgorithm = random_lb;
  else if (dynamicAlgo == "patchFactor") {
    d_dynamicAlgorithm = patch_factor_lb;
    d_collectParticles = false;
  }
  else if (dynamicAlgo == "patchFactorParticles" || dynamicAlgo == "particle3") {
    // particle3 is for backward-compatibility
    d_dynamicAlgorithm = patch_factor_lb;
    d_collectParticles = true;
  }
  else {
    if (d_myworld->myrank() == 0)
     cout << "Invalid Load Balancer Algorithm: " << dynamicAlgo
           << "\nPlease select 'cyclic', 'random', 'patchFactor' (default), or 'patchFactorParticles'\n"
           << "\nUsing 'patchFactor' load balancer\n";
    d_dynamicAlgorithm = patch_factor_lb;
  }
  d_lbInterval = interval;
  d_lbTimestepInterval = timestepInterval;
  d_doSpaceCurve = spaceCurve;
  d_lbThreshold = threshold;
  
  ASSERT(d_sharedState->getNumDims()>0 || d_sharedState->getNumDims()<4);
  //set curve parameters that do not change between timesteps
  sfc.SetNumDimensions(d_sharedState->getNumDims());
  sfc.SetMergeMode(1);
  sfc.SetCleanup(BATCHERS);
  sfc.SetMergeParameters(3000,500,2,.15);  //Should do this by profiling
}
