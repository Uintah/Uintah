#include <Packages/Uintah/CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
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
#include <stack>
#include <vector>
using namespace Uintah;
using namespace SCIRun;
using std::cerr;
static DebugStream doing("DynamicLoadBalancer_doing", false);
static DebugStream lb("DynamicLoadBalancer_lb", false);
static DebugStream dbg("DynamicLoadBalancer", false);

DynamicLoadBalancer::DynamicLoadBalancer(const ProcessorGroup* myworld)
   : LoadBalancerCommon(myworld), sfc(myworld)
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

  int numProcs = d_myworld->size();
  int numPatches = 0;
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    numPatches += level->numPatches();
  }

  vector<PatchInfo> allPatches(numPatches, PatchInfo(0,0,0));
  if (d_collectParticles && (d_processorAssignment.size() != 0) && d_scheduler->get_dw(0) != 0)
    collectParticles(grid, allPatches);
  else {
    for (int i = 0; i < numPatches; i++) {
      allPatches[i] = PatchInfo(i,0,0);
    }
  }

  vector<Patch*> patchset;
  vector<double> patch_costs;

  // these variables are "groups" of costs.  If we are doing level independent, then
  // the variables will be size one, and we can share functionality
  vector<double> groupCost(1,0);
  vector<double> avgCostPerProc(1,0);
  vector<int> groupSize(1,0);
  vector<int> order(numPatches);

  if (d_levelIndependent) {
    groupCost.resize(grid->numLevels());
    avgCostPerProc.resize(grid->numLevels());
    groupSize.resize(grid->numLevels());
  }
  // make a list of Patch*'s and costs per patch

  sort(allPatches.begin(), allPatches.end(), PatchCompare());
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
      useSFC(level, &order[startingPatch]);
    }

    for (Level::const_patchIterator iter = level->patchesBegin(); 
        iter != level->patchesEnd(); iter++) {

      Patch* patch = *iter;
      int id = patch->getGridIndex();

      ASSERTEQ(id, allPatches[id].id);
      patchset.push_back(patch);
      IntVector range = patch->getHighIndex() - patch->getLowIndex();
      double cost = allPatches[id].numParticles + d_cellFactor * 
        range.x() * range.y() * range.z();
      cost *= timeWeight;
      if ( d_myworld->myrank() == 0)
        dbg << d_myworld->myrank() << "  Patch: " << id << " cost: " << cost << " " << allPatches[id].numParticles << " " << range.x() * range.y() * range.z() << " " << d_cellFactor << " TW " << timeWeight << endl;
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
        index = order[p] + startingPatch;
        //cout << d_myworld->myrank() << ": mapping " << order[p] << " to " << index << " startingPatch:" << startingPatch << endl;
      }
      else {
        // not attempting space-filling curve
        index = p;
      }
      
      ASSERT(allPatches[index].assigned==false);
      
      // assign the patch to a processor.  When we advance procs,
      // re-update the cost, so we use all procs (and don't go over)
      double patchCost = patch_costs[index];
      double notakeimb=fabs(currentProcCost-avgCostPerProc[i]);
      double takeimb=fabs(currentProcCost+patchCost-avgCostPerProc[i]);
              
      if (notakeimb<takeimb) {
        // move to next proc and add this patch
        currentProc++;
        d_tempAssignment[index] = currentProc;
        //update average (this ensures we don't over/under fill to much)
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

void DynamicLoadBalancer::sortPatches(vector<Region> &patches)
{
  if(d_doSpaceCurve) //reorder by sfc
  {
    int *dimensions=d_sharedState->getActiveDims();
    vector<double> positions;
    vector<DistributedIndex> indices;
    int mode;
    unsigned int numProcs=d_myworld->size();
    int myRank=d_myworld->myrank();

    //decide to run serial or parallel
    if(numProcs==1 || patches.size()<numProcs) //compute in serial if the number of patches is small
    {
      mode=SERIAL;
      numProcs=1;
      myRank=0;
    }
    else 
    {
      mode=PARALLEL;
    }
    //create locations array on each processor
    for(unsigned int p=0;p<patches.size();p++)
    {
      int proc = (p*numProcs)/patches.size();
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

    vector<int> displs(numProcs, 0);
    vector<int> recvcounts(numProcs,0); // init the counts to 0
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
    for(unsigned int i=0;i<indices.size();i++)
    {
        DistributedIndex di=indices[i];
        //index is equal to the diplacement for the processor converted to struct size plus the local index
        //int index=displs[di.p]/sizeof(DistributedIndex)+di.i;
        int index=(int)ceil((float)di.p*patches.size()/d_myworld->size())+di.i;
        reorderedPatches[i]=patches[index];
    }
  }
  //random? cyclicic?
}
void DynamicLoadBalancer::getCosts(const LevelP& level, const vector<Region> &patches, vector<double> &costs)
{
  //level can be null?  do we need level?  or old grid instead maybe?  
  costs.resize(0);
  //cout << "need to implment costs\n";
  for(vector<Region>::const_iterator patch=patches.begin();patch!=patches.end();patch++)
  {
    costs.push_back(patch->getVolume());
  }
}
void DynamicLoadBalancer::dynamicallyLoadBalanceAndSplit(const GridP& oldGrid, SizeList min_patch_size,vector<vector<Region> > &levels, bool canSplit)
{
  double start = Time::currentSeconds();

  int *dimensions=d_sharedState->getActiveDims();
  
  d_tempAssignment.resize(0);

  //loop over levels
  for(unsigned int l=0;l<levels.size();l++)
  {
    vector<Region> &patches=levels[l];
    vector<double> costs;

    //set SFC parameters here so they are the same if we need to sort after splitting later
    if (d_doSpaceCurve)        
    {

      Region bounds=*patches.begin();
      //find domain bounds
      for(vector<Region>::iterator patch=patches.begin();patch<patches.end();patch++)
      {
         bounds.low()=Min(patch->low(),bounds.low());
         bounds.high()=Max(patch->high(),bounds.high());
      }
      Vector range=(bounds.high()-bounds.low()).asVector();
      Vector center=(bounds.high()+bounds.low()).asVector()/2;
      double r[3]={range[dimensions[0]],range[dimensions[1]],range[dimensions[2]]};
      double c[3]={center[dimensions[0]],center[dimensions[1]],center[dimensions[2]]};
      double delta[3]={min_patch_size[l][dimensions[0]],min_patch_size[l][dimensions[1]],min_patch_size[l][dimensions[2]]};
    
      sfc.SetDimensions(r);
      sfc.SetCenter(c);
      sfc.SetRefinementsByDelta(delta);
    }
    sortPatches(patches);

    LevelP level=0;

    if(l<(unsigned int)oldGrid->numLevels())
    {
      level=oldGrid->getLevel(l);
    }
    //get costs (get costs will determine algorithm)  
    getCosts(level,patches,costs);

    double totalCost=0;
    double targetCost=0;
    double currentCost=0;
    for(unsigned int i=0;i<costs.size();i++)
    {
      totalCost+=costs[i];
    }
    targetCost=totalCost/d_myworld->size();
   
    vector<Region> assignedPatches;
    stack<Region> unassignedPatches;
    stack<double> unassignedPatchesCost;
    vector<Region>::iterator toAssignPatch=patches.begin();
    vector<double>::iterator toAssignCost=costs.begin();

    int currentProc=0;
    int numProcs=d_myworld->size();
    while(toAssignPatch!=patches.end() || !unassignedPatches.empty())
    {
      Region patch;
      double cost;

      //get patch that needs assignment
      if(!unassignedPatches.empty())    //first grab from stack where we place patches after splitting
      {
        patch=unassignedPatches.top();
        cost=unassignedPatchesCost.top();
        unassignedPatches.pop();
        unassignedPatchesCost.pop();
      }
      else                              //then grab from original patch list
      {
        patch=*toAssignPatch;
        cost=*toAssignCost;
        toAssignPatch++;
        toAssignCost++;
      }
      
      if(currentCost+cost<targetCost)  //if patch fits in current processor
      {
        //assign to current proc
        d_tempAssignment.push_back(currentProc);
        assignedPatches.push_back(patch);
        
        //update vars
        currentCost+=cost;
        totalCost-=cost;
      }
      else
      {
        //calculate number of possible patches in each dimension
        int numPossiblePatches=1;
        IntVector size;
        int dim=0; 
        if(canSplit && l!=0)
        {
          size=(patch.high()-patch.low())/min_patch_size[l];
      
          //find maximum dimension
          for(int d=1;d<3;d++)
          {
            if(size[d]>size[dim])
              dim=d;
          }
          numPossiblePatches=size[0]*size[1]*size[2];
        }
        double minCost=cost/numPossiblePatches;  //estimate of the cost of the next patch if we split fully

        if(canSplit && l!=0 && size[dim]>1 && currentCost+minCost<=targetCost) //if can be split and splitting should help
        {
          //calculate split point
          int mid=patch.getLow()[dim]+(int(size[dim])/2)*min_patch_size[l][dim];
          //split patch
          Region left=patch,right=patch;
          left.high()[dim]=mid;
          right.low()[dim]=mid;
          vector<Region> newpatches;
          newpatches.push_back(left);
          newpatches.push_back(right);
           
          //sort both patches in serial
          sortPatches(newpatches);
          
          //place in reverse order on to assign stack
          unassignedPatches.push(newpatches[1]);
          unassignedPatches.push(newpatches[0]);

          //derive costs by a percentage of old costs
            //ideally we would recalculate costs but particles cause a problem currently
          double newCost=cost * double(newpatches[0].getVolume())/patch.getVolume();

          unassignedPatchesCost.push(newCost);
          unassignedPatchesCost.push(cost-newCost);
        }
        else  //cannot be split so attempt to assign it to currentProc
        {
          double takeimb=fabs(currentCost+cost-targetCost);
          double notakeimb=fabs(currentCost-targetCost);

          if(notakeimb<takeimb) //taking patch would cause more imbalance then not taking it
          {
            //move to next proc
            currentProc++;
           
            //place patch back in queue for assignment
            unassignedPatches.push(patch);
            unassignedPatchesCost.push(cost);

            //update vars
            currentCost=0;
            targetCost=totalCost/(numProcs-currentProc);
          }
          else  //take patch as it causes the least imbalance
          {
            //assign to this proc
            d_tempAssignment.push_back(currentProc);
            assignedPatches.push_back(patch);

            //update vars
            currentCost+=cost;
            totalCost-=cost;
          } 
        } //end if can be split else
      } //end if fits onto the processor without splitting else
    } //end while() patches to assign
    
    //update patchlist
    patches=assignedPatches;

  }  //end for each level
  d_sharedState->loadbalancerTime += Time::currentSeconds() - start;
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
    d_oldAssignment = d_processorAssignment;
    d_oldAssignmentBasePatch = d_assignmentBasePatch;
    
    bool dynamicAllocate = false;
    //temp assignment can be set if the regridder has already called the load balancer
    if(d_tempAssignment.empty())
    {
      int numPatches = 0;
      for(int l=0;l<grid->numLevels();l++){
        const LevelP& level = grid->getLevel(l);
        numPatches += level->numPatches();
      }
    
      d_tempAssignment.resize(numPatches);
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
        int numProcs = d_myworld->size();
        int myrank = d_myworld->myrank();
        if (myrank == 0) {
          LevelP curLevel = grid->getLevel(0);
          Level::const_patchIterator iter = curLevel->patchesBegin();
          lb << "  Changing the Load Balance\n";
          vector<int> costs(numProcs);
          for (unsigned int i = 0; i < d_processorAssignment.size(); i++) {
            lb << myrank << " patch " << i << " (real " << (*iter)->getID() << ") -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") patch size: "  << (*iter)->getGridIndex() << " " << ((*iter)->getHighIndex() - (*iter)->getLowIndex()) << "\n";
            IntVector range = ((*iter)->getHighIndex() - (*iter)->getLowIndex());
            costs[d_processorAssignment[i]] += range.x() * range.y() * range.z();
            iter++;
            if (iter == curLevel->patchesEnd() && i+1 < d_processorAssignment.size()) {
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
  
  ASSERT(d_sharedState->getNumDims()>0 || d_sharedState->getNumDims()<4);
  //set curve parameters that do not change between timesteps
  sfc.SetNumDimensions(d_sharedState->getNumDims());
  sfc.SetMergeMode(1);
  sfc.SetCleanup(BATCHERS);
  sfc.SetMergeParameters(3000,500,2,.15);  //Should do this by profiling
}
