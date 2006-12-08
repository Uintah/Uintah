#include <Packages/Uintah/CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/SFC.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Time.h>
#include <Core/Exceptions/InternalError.h>

#include <iostream> // debug only

using namespace Uintah;
using namespace SCIRun;
using std::cerr;
static DebugStream doing("DynamicLoadBalancer_doing", false);
static DebugStream lb("DynamicLoadBalancer_lb", false);
static DebugStream dbg("DynamicLoadBalancer", false);

DynamicLoadBalancer::DynamicLoadBalancer(const ProcessorGroup* myworld)
   : LoadBalancerCommon(myworld)
{
  d_lbInterval = 0.0;
  d_lastLbTime = 0.0;
  d_lbTimestepInterval = 0;
  d_lastLbTimestep = 0;
  d_timeRefineWeight = false;
  d_checkAfterRestart = false;
  d_levelIndependent = true;

  d_dynamicAlgorithm = patch_factor_lb;  
  d_do_AMR = false;
  d_pspec = 0;

  d_assignmentBasePatch = -1;
  d_oldAssignmentBasePatch = -1;
}

DynamicLoadBalancer::~DynamicLoadBalancer()
{
}

void DynamicLoadBalancer::collectParticles(const GridP& grid, std::vector<PatchInfo>& allParticles)
{
  if (d_processorAssignment.size() == 0)
    return; // if we haven't been through the LB yet, don't try this.

  if (d_myworld->myrank() == 0)
    dbg << " DLB::collectParticles\n";

  int numProcs = d_myworld->size();
  int myrank = d_myworld->myrank();
  // get how many particles were each patch had at the end of the last timestep
  //   gather from each proc - based on the last location

  DataWarehouse* dw = d_scheduler->get_dw(0);
  if (dw == 0)
    return;
  const Grid* oldGrid = dw->getGrid();

  //construct a mpi datatype for the PatchInfo
  MPI_Datatype particletype;
  //if (numProcs > 1) {
  MPI_Type_contiguous(3, MPI_INT, &particletype);
  MPI_Type_commit(&particletype);
    //}

  if (*grid.get_rep() == *oldGrid) {
    // order patches by processor #
    vector<int> sorted_processorAssignment = d_processorAssignment;
    sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());

    vector<int> displs(numProcs, 0);
    vector<int> recvcounts(numProcs,0); // init the counts to 0
    vector<PatchInfo> particleList;

    for (int i = 0; i < (int)d_processorAssignment.size(); i++) {
      recvcounts[sorted_processorAssignment[i]]++;
    }

    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

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
        PatchInfo p(id,thisPatchParticles,0);
        particleList.push_back(p);

      }
    }

    if (d_myworld->myrank() == 0) {
      int totalRecv = 0;
      for (unsigned int i = 0; i < recvcounts.size(); i++) {
        totalRecv += recvcounts[i]; 
      }
    }
    MPI_Allgatherv(&particleList[0], particleList.size(), particletype,
	        &allParticles[0], &recvcounts[0], &displs[0], particletype,
	        d_myworld->getComm());

  }
  else {
    // collect particles from the old grid's patches onto processor 0 and then distribute them
    // (it's either this or do 2 consecutive load balances).  For now, it's safe to assume that
    // if there is a new level or a new patch there are no particles there.

    vector<int> displs(numProcs,0);
    vector<int> recvcounts(numProcs,0); // init the counts to 0
    int totalsize = 0;

    vector<PatchInfo> subpatchParticles;
    for(int l=0;l<grid->numLevels();l++){
      const LevelP level = grid->getLevel(l);
      for (Level::const_patchIterator iter = level->patchesBegin(); 
           iter != level->patchesEnd(); iter++) {
        Patch *patch = *iter;

        if (l >= oldGrid->numLevels()) {
          // new patch - no particles yet
          recvcounts[0]++;
          totalsize++;
          if (d_myworld->myrank() == 0) {
            PatchInfo pi(patch->getGridIndex(), 0, 0);
            subpatchParticles.push_back(pi);
          }
          continue;
        }

        // find all the particles on old patches
        const LevelP oldLevel = oldGrid->getLevel(l);
        Level::selectType oldPatches;
        oldLevel->selectPatches(patch->getLowIndex(), patch->getHighIndex(), oldPatches);

        if (oldPatches.size() == 0) {
          recvcounts[0]++;
          totalsize++;
          if (d_myworld->myrank() == 0) {
            PatchInfo pi(patch->getGridIndex(), 0, 0);
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
            low = Max(patch->getLowIndex(), oldPatch->getLowIndex());
            high = Min(patch->getHighIndex(), oldPatch->getHighIndex());

            int thisPatchParticles = 0;
            if (dw) {
              const MaterialSet *matls = d_sharedState->allMPMMaterials();
              const MaterialSubset *ms;
              if (matls) {
                ms = matls->getSubset(0);
                int size = ms->size();
                for (int matl = 0; matl < size; matl++) {
                  ParticleSubset* psubset = 0;
                  if (dw->haveParticleSubset(matl, oldPatch, low, high))
                    psubset = dw->getParticleSubset(matl, oldPatch, low, high);
                  if (psubset)
                    thisPatchParticles += psubset->numParticles();
                }
              }
            }
            PatchInfo p(patch->getGridIndex(), thisPatchParticles, 0);
            subpatchParticles.push_back(p);
          }
        }
      }
    }

    vector<PatchInfo> recvbuf(totalsize);
    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

    MPI_Gatherv(&subpatchParticles[0], recvcounts[d_myworld->myrank()], particletype, &recvbuf[0],
                &recvcounts[0], &displs[0], particletype, 0, d_myworld->getComm());

    if ( d_myworld->myrank() == 0) {
      for (unsigned i = 0; i < recvbuf.size(); i++) {
        PatchInfo& spi = recvbuf[i];
        PatchInfo& pi = allParticles[spi.id];
        pi.id = spi.id;
        pi.numParticles += spi.numParticles;
      }
    }
    // combine all the subpatches results
    MPI_Bcast(&allParticles[0], 3*allParticles.size(), MPI_INT,0,d_myworld->getComm());
  }
  MPI_Type_free(&particletype);
  if (dbg.active() && d_myworld->myrank() == 0) {
    for (unsigned i = 0; i < allParticles.size(); i++) {
      PatchInfo& pi = allParticles[i];
      dbg << d_myworld->myrank() << "  Post gather index " << i << ": " << pi.id << " numP : " << pi.numParticles << endl;
    }
  }
}

void DynamicLoadBalancer::useSFC(const LevelP& level, DistributedIndex* output)
{
  // output needs to be at least the number of patches in the level

  vector<DistributedIndex> indices; //output
  vector<int> recvcounts(d_myworld->size(), 0);

  // positions will be in float triplets
  vector<float> positions;


  // get the overall range in all dimensions from all patches
  IntVector high(INT_MIN,INT_MIN,INT_MIN);
  IntVector low(INT_MAX,INT_MAX,INT_MAX);
  
  for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) 
  {
    const Patch* patch = *iter;
    
    high = Max(high, patch->getInteriorCellHighIndex());
    low = Min(low, patch->getInteriorCellLowIndex());

  }
  IntVector range = high-low;
  Vector center;
  center[0]=(high[0]+low[0])/2.0;
  center[1]=(high[1]+low[1])/2.0;
  center[2]=(high[2]+low[2])/2.0;
  IntVector dimensions;
  int dim=0;
  for(int d=0;d<3;d++)
  {
     if(range[d]>1)
     {
        dimensions[dim]=d;
        dim++;
     }
  }
  
  if( (level->numPatches()<d_myworld->size()) || d_myworld->size()==1)  //do in serial
  {
    for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) 
    {
      // use center*2, like PatchRangeTree
      const Patch* patch = *iter;

      IntVector ipos = patch->getInteriorCellLowIndex()+patch->getInteriorCellHighIndex();
      Vector pos(ipos[0]/ 2.0, ipos[1] / 2.0, ipos[2] / 2.0);

      for(int d=0;d<dim;d++)
      {
        positions.push_back(pos[dimensions[d]]);
      }
    }
    if(dim==3)
    {
      SFC3f curve(HILBERT, d_myworld);
      curve.SetLocalSize(level->numPatches());
      curve.SetDimensions(range.x(), range.y(), range.z());
      curve.SetLocations(&positions);
      curve.SetOutputVector(&indices);
      curve.SetCenter(center.x(),center.y(), center.z());
      curve.GenerateCurve(SERIAL);
    }
    else if (dim==2)
    {
      SFC2f curve(HILBERT, d_myworld);   
      curve.SetLocalSize(level->numPatches());
      curve.SetDimensions(range[dimensions[0]], range[dimensions[1]]);
      curve.SetLocations(&positions);
      curve.SetOutputVector(&indices);
      curve.SetCenter(center[dimensions[0]],center[dimensions[1]]);
      curve.GenerateCurve(SERIAL);
    }
    else
    {
      SFC1f curve(d_myworld);
      curve.SetLocalSize(level->numPatches());
      curve.SetDimensions(range[dimensions[0]]);
      curve.SetLocations(&positions);
      curve.SetOutputVector(&indices);
      curve.SetCenter(center[dimensions[0]]);
      curve.GenerateCurve(SERIAL);
    }
    memcpy(output,&indices[0],sizeof(DistributedIndex)*level->numPatches());
  }
  else  //calculate in parallel
  {
    IntVector min_patch_size(INT_MAX,INT_MAX,INT_MAX);  
    // go through the patches, to place them on processors and to calculate the high
    for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) 
    {
      // use center*2, like PatchRangeTree
      const Patch* patch = *iter;
    
      IntVector size=patch->getInteriorCellHighIndex()-patch->getInteriorCellLowIndex();
      for(int d=0;d<3;d++)
      {
        if(size[d]<min_patch_size[d])
        {
          min_patch_size[d]=size[d];
        }
      }
      // since the new levels and patches haven't been assigned processors yet, go through the
      // patches and determine processor base this way.
      int proc = (patch->getLevelIndex()*d_myworld->size())/level->numPatches();
      recvcounts[proc]++;
      if (d_myworld->myrank() == proc) 
      {
        IntVector ipos = patch->getInteriorCellLowIndex()+patch->getInteriorCellHighIndex();
        Vector pos(ipos[0]/ 2.0, ipos[1] / 2.0, ipos[2] / 2.0);

        for(int d=0;d<dim;d++)
        {
          positions.push_back(pos[dimensions[d]]);
        }
      }
    }

    if(dim==3)
    {
      SFC3f curve(HILBERT, d_myworld);
      curve.SetLocalSize(positions.size()/3);
      curve.SetDimensions(range.x(), range.y(), range.z());
      curve.SetRefinementsByDelta(min_patch_size.x(),min_patch_size.y(),min_patch_size.z()); 
      curve.SetLocations(&positions);
      curve.SetOutputVector(&indices);
      curve.SetCenter(center.x(),center.y(), center.z());
      curve.SetMergeMode(1);
      curve.SetCleanup(BATCHERS);
      curve.SetMergeParameters(3000,500,2,.15);
      curve.GenerateCurve();
    }
    else if(dim==2)
    {
      SFC2f curve(HILBERT, d_myworld);
      curve.SetLocalSize(positions.size()/2);
      curve.SetDimensions(range[dimensions[0]],range[dimensions[1]]);
      curve.SetRefinementsByDelta(min_patch_size[dimensions[0]],min_patch_size[dimensions[1]]); 
      curve.SetLocations(&positions);
      curve.SetOutputVector(&indices);
      curve.SetCenter(center[dimensions[0]],center[dimensions[1]]);
      curve.SetMergeMode(1);
      curve.SetCleanup(BATCHERS);
      curve.SetMergeParameters(3000,500,2,.15);
      curve.GenerateCurve();
    }
    else
    {
      SFC1f curve(d_myworld);
      curve.SetLocalSize(positions.size());
      curve.SetDimensions(range[dimensions[0]]);
      curve.SetRefinementsByDelta(min_patch_size[dimensions[0]]); 
      curve.SetLocations(&positions);
      curve.SetOutputVector(&indices);
      curve.SetCenter(center[dimensions[0]]);
      curve.SetMergeMode(1);
      curve.SetCleanup(BATCHERS);
      curve.SetMergeParameters(3000,500,2,.15);
      curve.GenerateCurve();
    }

    //indices comes back in the size of each proc's patch set, pointing to the index
    // gather it all into one array
    
    int rsize=indices.size();
     
    //gather recv counts
    MPI_Allgather(&rsize,1,MPI_INT,&recvcounts[0],1,MPI_INT,d_myworld->getComm());
    
    vector<int> displs(d_myworld->size(), 0);
    
    for (unsigned i = 0; i < recvcounts.size(); i++)
    {
       recvcounts[i]*=sizeof(DistributedIndex);
    }
    for (unsigned i = 1; i < recvcounts.size(); i++)
    {
       displs[i] = recvcounts[i-1] + displs[i-1];
    }
    //gather curve
    MPI_Allgatherv(&indices[0], recvcounts[d_myworld->myrank()], MPI_BYTE, output, &recvcounts[0], 
                   &displs[0], MPI_BYTE, d_myworld->getComm());
  } 
  /*
  for (int i = 0; i < level->numPatches(); i++) 
  {
    for (int j = i+1; j < level->numPatches(); j++)
    {
      if (output[i].p == output[j].p && output[i].i == output[j].i) 
      {
        cout << "Rank:" << d_myworld->myrank() <<  ":   ALERT!!!!!! index done twice: index " << i << " has the same value as index " << j << " " << output[i].p << ":" << output[j].i << endl;
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

  int numProcs = d_myworld->size();
  int numPatches = 0;
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    numPatches += level->numPatches();
  }

  vector<PatchInfo> allParticles(numPatches, PatchInfo(0,0,0));
  if (d_collectParticles && (d_processorAssignment.size() != 0) && d_scheduler->get_dw(0) != 0)
    collectParticles(grid, allParticles);
  else {
    for (int i = 0; i < numPatches; i++) {
      allParticles[i] = PatchInfo(i,0,0);
    }
  }

  vector<Patch*> patchset;
  vector<double> patch_costs;

  // these variables are "groups" of costs.  If we are doing level independent, then
  // the variables will be size one, and we can share functionality
  vector<double> groupCost(1,0);
  vector<double> avgCostPerProc(1,0);
  vector<int> groupSize(1,0);
  vector<DistributedIndex> sfc(numPatches);

  if (d_levelIndependent) {
    groupCost.resize(grid->numLevels());
    avgCostPerProc.resize(grid->numLevels());
    groupSize.resize(grid->numLevels());
  }
  // make a list of Patch*'s and costs per patch

  sort(allParticles.begin(), allParticles.end(), PatchCompare());
  int timeWeight = 1;
  int index = 0;
  int startingPatch = 0;
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    double levelcost = 0;

    if (l > 0 && d_timeRefineWeight && !d_sharedState->isLockstepAMR()) {
      timeWeight *= level->getRefinementRatioMaxDim(); 
    }

    if (d_doSpaceCurve) {
      //cout << d_myworld->myrank() << "   Doing SFC level " << l << endl;
      useSFC(level, &sfc[startingPatch]);
    }

    for (Level::const_patchIterator iter = level->patchesBegin(); 
        iter != level->patchesEnd(); iter++) {

      Patch* patch = *iter;
      int id = patch->getGridIndex();

      ASSERTEQ(id, allParticles[id].id);
      patchset.push_back(patch);
      IntVector range = patch->getHighIndex() - patch->getLowIndex();
      double cost = allParticles[id].numParticles + d_cellFactor * 
        range.x() * range.y() * range.z();
      cost *= timeWeight;
      if ( d_myworld->myrank() == 0)
        dbg << d_myworld->myrank() << "  Patch: " << id << " cost: " << cost << " " << allParticles[id].numParticles << " " << range.x() * range.y() * range.z() << " " << d_cellFactor << " TW " << timeWeight << endl;
      patch_costs.push_back(cost);
      levelcost += cost;
    }
    groupCost[index] += levelcost;
    groupSize[index] += level->numPatches();
    if (d_levelIndependent) {
      index++;
    }
    startingPatch += level->numPatches();
  }

  for (unsigned i = 0; i < groupCost.size(); i++)
    avgCostPerProc[i] = groupCost[i] / numProcs;
  

  // TODO - bring back excessive patch clumps?
  startingPatch = 0;
  for (unsigned i = 0; i < groupCost.size(); i++) {
    int currentProc = 0;
    double currentProcCost = 0;
    
    for (int p = startingPatch; p < groupSize[i] + startingPatch; p++) {
      int index;
      if (d_doSpaceCurve) {
        //index = sfc[p] + startingPatch;
        index = (int)ceil( double(sfc[p].p*groupSize[i]) /d_myworld->size() ) +sfc[p].i + startingPatch;
        //cout << d_myworld->myrank() << ": mapping " << sfc[p].p << ":" << sfc[p].i << " to " << index << endl;
      }
      else {
        // not attempting space-filling curve
        index = p;
      }
      if (allParticles[index].assigned)
        continue;
      
      // assign the patch to a processor.  When we advance procs,
      // re-update the cost, so we use all procs (and don't go over)
      double patchCost = patch_costs[index];

      double imb1=fabs(currentProcCost-avgCostPerProc[i]);
      double imb2=fabs(currentProcCost+patchCost-avgCostPerProc[i]);
              
      if (imb1<imb2) {
        // move to next proc and add this patch
        currentProc++;
        d_tempAssignment[index] = currentProc;
        groupCost[i] -= currentProcCost;
        avgCostPerProc[i] = groupCost[i] / (numProcs-currentProc);
        currentProcCost = patchCost;
      }
      else {
        // add patch to currentProc
        d_tempAssignment[index] = currentProc;
        currentProcCost += patchCost;
      }
      if (d_myworld->myrank() == 0)
        dbg << "Patch " << index << "-> proc " << currentProc 
            << " PatchCost: " << patchCost << ", ProcCost: " 
            << currentProcCost << " group cost " << groupCost[i] << "  avg cost " << avgCostPerProc[i]
            << ", idcheck: " << patchset[index]->getGridIndex() << " (" << patchset[index]->getID() << ")" << endl;
    }
    startingPatch += groupSize[i];
  }

  bool doLoadBalancing = force || thresholdExceeded(patch_costs);

  time = Time::currentSeconds() - time;
  if (d_myworld->myrank() == 0)
    dbg << " Time to LB: " << time << endl;
  doing << d_myworld->myrank() << "   APF END\n";
  MPI_Barrier(d_myworld->getComm());
  return doLoadBalancing;
}

bool DynamicLoadBalancer::thresholdExceeded(const vector<double>& patch_costs)
{
  // add up the costs each processor for the current assignment
  // and for the temp assignment, then calculate the standard deviation
  // for both.  If (curStdDev / tmpStdDev) > threshold, return true,
  // and have possiblyDynamicallyRelocate change the load balancing
  
  int numProcs = d_myworld->size();
  vector<double> currentProcCosts(numProcs);
  vector<double> tempProcCosts(numProcs);
  
  for (unsigned i = 0; i < d_tempAssignment.size(); i++) {
    currentProcCosts[d_processorAssignment[i]] += patch_costs[i];
    tempProcCosts[d_tempAssignment[i]] += patch_costs[i];
  }
  
  // use the std dev formula:
  // sqrt((n*sum_of_squares - sum squared)/n squared)
  double avg_current = 0;
  double max_current = 0;
  double avg_temp = 0;
  double max_temp = 0;

  for (int i = 0; i < d_myworld->size(); i++) {
    avg_current += currentProcCosts[i];
    if (currentProcCosts[i] > max_current) max_current = currentProcCosts[i];
    if (tempProcCosts[i] > max_temp) max_temp = currentProcCosts[i];
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

  int numProcs = d_myworld->size();
  int numPatches = (int)d_processorAssignment.size();

  vector<int> proc_record(numProcs,0);
  int max_ppp = numPatches / numProcs;

  for (int i = 0; i < numPatches; i++) {
    int proc = (int) (((float) rand()) / RAND_MAX * numProcs);
    int newproc = proc;

    // only allow so many patches per proc.  Linear probe if necessary,
    // but make sure every proc has some work to do
    while (proc_record[newproc] >= max_ppp) {
      newproc++;
      if (newproc >= numProcs) newproc = 0;
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

  int numProcs = d_myworld->size();
  for (unsigned i = 0; i < d_tempAssignment.size(); i++) {
    d_tempAssignment[i] = (d_processorAssignment[i] + 1 ) % numProcs;
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
  //int proc = (patch->getLevelIndex()*numProcs)/patch->getLevel()->numPatches();
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
DynamicLoadBalancer::restartInitialize(ProblemSpecP& pspec, string tsurl, const GridP& grid)
{
  // here we need to grab the uda data to reassign patch data to the 
  // processor that will get the data
  int numPatches = 0;
  int startingID = (*(grid->getLevel(0)->patchesBegin()))->getID();
  int prevNumProcs = 0;

  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    numPatches += level->numPatches();
  }

  d_processorAssignment.resize(numPatches);
  d_assignmentBasePatch = startingID;
  for (unsigned i = 0; i < d_processorAssignment.size(); i++)
    d_processorAssignment[i]= -1;

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

bool DynamicLoadBalancer::possiblyDynamicallyReallocate(const GridP& grid, int state)
{
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
    int numProcs = d_myworld->size();
    int numPatches = 0;
    
    for(int l=0;l<grid->numLevels();l++){
      const LevelP& level = grid->getLevel(l);
      numPatches += level->numPatches();
    }
    
    int myrank = d_myworld->myrank();
    
    d_oldAssignment = d_processorAssignment;
    d_oldAssignmentBasePatch = d_assignmentBasePatch;
    d_tempAssignment.clear();
    d_tempAssignment.resize(numPatches);
    if (d_myworld->myrank() == 0)
      doing << d_myworld->myrank() << "  Checking whether we need to LB\n";
    bool dynamicAllocate = false;
    switch (d_dynamicAlgorithm) {
    case patch_factor_lb:  dynamicAllocate = assignPatchesFactor(grid, force); break;
    case cyclic_lb:        dynamicAllocate = assignPatchesCyclic(grid, force); break;
    case random_lb:        dynamicAllocate = assignPatchesRandom(grid, force); break;
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
        if (myrank == 0) {
          LevelP curLevel = grid->getLevel(0);
          Level::const_patchIterator iter = curLevel->patchesBegin();
          lb << "  Changing the Load Balance\n";
          vector<int> costs(numProcs);
          for (int i = 0; i < numPatches; i++) {
            lb << myrank << " patch " << i << " (real " << (*iter)->getID() << ") -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") patch size: "  << (*iter)->getGridIndex() << " " << ((*iter)->getHighIndex() - (*iter)->getLowIndex()) << "\n";
            IntVector range = ((*iter)->getHighIndex() - (*iter)->getLowIndex());
            costs[d_processorAssignment[i]] += range.x() * range.y() * range.z();
            iter++;
            if (iter == curLevel->patchesEnd() && i+1 < numPatches) {
              curLevel = curLevel->getFinerLevel();
              iter = curLevel->patchesBegin();
            }
          }
          for (int i = 0; i < numProcs; i++) {
            lb << myrank << " proc " << i << "  has cost: " << costs[i] << endl;
          }
        }
      }
      
    }
  }
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
  double cellFactor = .1;
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
    p->getWithDefault("cellFactor", cellFactor, .1);
    p->getWithDefault("gainThreshold", threshold, 0.0);
    p->getWithDefault("doSpaceCurve", spaceCurve, false);
    p->getWithDefault("timeRefinementWeight", d_timeRefineWeight, false);
    p->getWithDefault("levelIndependent", d_levelIndependent, true);
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
  d_cellFactor = cellFactor;
}
