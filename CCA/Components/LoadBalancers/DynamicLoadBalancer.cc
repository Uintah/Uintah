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

  d_dynamicAlgorithm = static_lb;  
  d_state = checkLoadBalance;
  d_do_AMR = false;
  d_pspec = 0;
}

DynamicLoadBalancer::~DynamicLoadBalancer()
{
}

void DynamicLoadBalancer::collectParticles(const GridP& grid, std::vector<PatchInfo>& allParticles)
{
  if (d_myworld->myrank() == 0)
    dbg << " DLB::collectParticles\n";

  int numProcs = d_myworld->size();
  int myrank = d_myworld->myrank();
  // get how many particles were each patch had at the end of the last timestep
  //   gather from each proc - based on the last location

  DataWarehouse* dw = d_scheduler->get_dw(0);
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

    vector<int> displs;
    vector<int> recvcounts(numProcs,0); // init the counts to 0
    vector<PatchInfo> particleList;

    int offsetProc = 0;
    for (int i = 0; i < (int)d_processorAssignment.size(); i++) {
      // set the offsets for the MPI_Gatherv
      if ( offsetProc == sorted_processorAssignment[i]) {
        displs.push_back(i);
        offsetProc++;
      }
      recvcounts[sorted_processorAssignment[i]]++;
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
          //        const MaterialSet *matls = sc->getMaterialSet();
          const MaterialSet *matls = d_sharedState->allMPMMaterials();
          const MaterialSubset *ms;
          if (matls) {
            ms = matls->getSubset(0);
            int size = ms->size();
            for (int matl = 0; matl < size; matl++) {
              ParticleSubset* psubset = 0;
              if (dw->haveParticleSubset(matl, patch))
                psubset = dw->getParticleSubset(matl, patch);
              if (psubset)
                thisPatchParticles += psubset->numParticles();
            }
          }
        }
        // add to particle list
        PatchInfo p(id,thisPatchParticles,0);
        particleList.push_back(p);

      }
    }
    
    MPI_Allgatherv(&particleList[0],(int)particleList.size(),particletype,
	        &allParticles[0], &recvcounts[0], &displs[0], particletype,
	        d_myworld->getComm());

    //cout << "Post gather\n";
    //for (i = 0; i < (int)allParticles.size(); i++)
    //cout << "Patch: " << allParticles[i].id
    //     << " -> " << allParticles[i].numParticles << endl;
    
  }
  else {
    // collect particles from the old grid's patches onto processor 0 and then distribute them
    // (it's either this or do 2 consecutive load balances).  For now, it's safe to assume that
    // if there is a new level or a new patch there are no particles there.

    vector<PatchInfo> subpatchParticles;
    for(int l=0;l<grid->numLevels();l++){
      const LevelP level = grid->getLevel(l);
      for (Level::const_patchIterator iter = level->patchesBegin(); 
           iter != level->patchesEnd(); iter++) {
        Patch *patch = *iter;

        if (l >= oldGrid->numLevels()) {
          // new patch - no particles yet
          PatchInfo pi(patch->getGridIndex(), 0, 0);
          subpatchParticles.push_back(pi);
          continue;
        }

        // find all the particles on old patches
        const LevelP oldLevel = oldGrid->getLevel(l);
        Level::selectType oldPatches;
        oldLevel->selectPatches(patch->getLowIndex(), patch->getHighIndex(), oldPatches);
        for (int i = 0; i < oldPatches.size(); i++) {
          const Patch* oldPatch = oldPatches[i];

          // three cases, send to proc 0, recv on proc 0, or store it (if on proc 0 and patch belongs to proc 0)
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
            if (myrank == 0) {
              // store it - don't send or recv
              subpatchParticles.push_back(p);
            }
            else {
              // send it to 0
              // expand this proc's array so we don't have to send the size
              subpatchParticles.push_back(p);
              MPI_Ssend(&p.id, 3, MPI_INT, 0, patch->getGridIndex()*12+oldPatch->getGridIndex(), d_myworld->getComm());
            }
          }
          else if (myrank == 0) {
            // recv it
            int src = d_processorAssignment[oldPatch->getGridIndex()];
            subpatchParticles.push_back(PatchInfo());
            MPI_Status status;
            MPI_Recv(&subpatchParticles[subpatchParticles.size()-1], 3, MPI_INT, src, 
                     patch->getGridIndex()*12+oldPatch->getGridIndex(), d_myworld->getComm(), &status);
          }
          else {
            // expand this proc's array so we don't have to send the size
            subpatchParticles.push_back(PatchInfo());
          }
        }
      }
    }
    // combine all the subpatches results
    MPI_Bcast(&subpatchParticles[0], 3*subpatchParticles.size(), MPI_INT,0,d_myworld->getComm());
    for (unsigned i = 0; i < subpatchParticles.size(); i++) {
      PatchInfo& spi = subpatchParticles[i];
      PatchInfo& pi = allParticles[spi.id];
      pi.id = spi.id;
      pi.numParticles += spi.numParticles;
      if (d_myworld->myrank() == 0)
        dbg << d_myworld->myrank() << "  Post gather " << spi.id << " numP : " << spi.numParticles << " total " << pi.numParticles << endl;
    }
  }
  MPI_Type_free(&particletype);
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
  if (d_collectParticles)
    collectParticles(grid, allParticles);
  else {
    for (int i = 0; i < numPatches; i++) {
      allParticles[i] = PatchInfo(i,0,0);
    }
  }

  vector<Patch*> patchset;
  vector<float> patch_costs;
  float avg_costPerProc = 0;
  float totalCost = 0;

  // make a list of Patch*'s and costs per patch

  sort(allParticles.begin(), allParticles.end(), PatchCompare());
  int timeWeight = 1;
  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);

    if (l > 0 && d_timeRefineWeight)
      timeWeight *= d_sharedState->timeRefinementRatio();


    for (Level::const_patchIterator iter = level->patchesBegin(); 
        iter != level->patchesEnd(); iter++) {
      Patch* patch = *iter;
      int id = patch->getGridIndex();

      ASSERTEQ(id, allParticles[id].id);
      patchset.push_back(patch);
      IntVector range = patch->getHighIndex() - patch->getLowIndex();
      float cost = allParticles[id].numParticles + d_cellFactor * 
        range.x() * range.y() * range.z();
      cost *= timeWeight;
      if ( d_myworld->myrank() == 0)
        dbg << d_myworld->myrank() << "  Patch: " << id << " cost: " << cost << " " << allParticles[id].numParticles << " " << range.x() * range.y() * range.z() << " " << d_cellFactor << " TW " << timeWeight << endl;
      patch_costs.push_back(cost);
      totalCost += cost;
    }
  }
    
  avg_costPerProc = totalCost / numProcs;
  

  // assume patches are in order of creation, and find find patch 
  // continuity based on their position.  Since patches are created z
  // then y then x, we will go down the z axis, when we get to the
  // end of the row, we will come up the z axis, and so forth, going
  // up a column, until we get to the top, and then we will come down
  // the next column, thus resulting in a continuous set of patches

  int x = 0;
  int y = 0;
  int z = 0;

  int y_patches = -1;
  int z_patches = -1;
  bool wrap_y = false;
  bool wrap_z = false;

  int currentProc = 0;
  float currentProcCost = 0;
  float origTotalCost = totalCost;
  
  // first pass - grab all patches that have a significant cost and 
  // give them their own processor

  /*
  for (int p = 0; p < numPatches; p++) {
    if (patch_costs[p] > .9 * avg_costPerProc) {
      d_tempAssignment[p] = currentProc;
      allParticles[p].assigned = true;
      totalCost -= patch_costs[p];
      dbg << "Patch " << p << "-> proc " << currentProc << " Cost: " 
          << patch_costs[p] << endl;
      currentProc++;
    }
  }
  */

  if (currentProc > 0) {
    int optimal_procs = (int)(totalCost /
      ((origTotalCost - totalCost) / currentProc) +  currentProc);
    if (d_myworld->myrank() == 0) {
      dbg << "This simulation would probably perform better on " 
          << optimal_procs << "-" << optimal_procs+1 << " processors\n";
      dbg << "  or rather - origCost " << origTotalCost << ", costLeft " 
          << totalCost << " after " << currentProc << " procs " << endl;
    }
  }
  avg_costPerProc = totalCost / (numProcs-currentProc);

  IntVector lastIndex = patchset[0]->getLowIndex();
  for (int p = 0; p < numPatches; p++, z++) {
    int index;
    if (d_doSpaceCurve) {
      IntVector low = patchset[p]->getLowIndex();
      if (low.z() <= lastIndex.z() && z>0) {
        // we wrapped in z
        if (z_patches == -1 && z > 0)
          z_patches = p;
        wrap_z = !wrap_z;
        z = 0;
        y++;
      }
      if (low.y() <= lastIndex.y() && y > 0 && z_patches != -1 && z == 0) {
        // we wrapped in y
        if (y_patches == -1)
          y_patches = p/z_patches;
        wrap_y = !wrap_y;
        y = 0;
        x++;
      }
      // we should do something here to compare in x (for different levels 
      // or boxes)
      
      lastIndex = low;
      
      //translate x,y, and z into a number
      index = x * (y_patches*z_patches) + 
        ((wrap_y) ? (y_patches-1-y)*z_patches : y*z_patches) +
        ((wrap_z) ? z_patches-1-z : z);
    }
    else {
      // not attempting space-filling curve
      index = p;
    }
    if (allParticles[index].assigned)
      continue;

    // assign the patch to a processor.  When we advance procs,
    // re-update the cost, so we use all procs (and don't go over)
    float patchCost = patch_costs[index];
    if (currentProcCost > avg_costPerProc ||
        (currentProcCost + patchCost > avg_costPerProc *1.1&&
          currentProcCost >=  .7*avg_costPerProc)) {
      // move to next proc and add this patch
      currentProc++;
      d_tempAssignment[index] = currentProc;
      totalCost -= currentProcCost;
      avg_costPerProc = totalCost / (numProcs-currentProc);
      currentProcCost = patchCost;
      if (d_myworld->myrank() == 0)
        dbg << "Patch " << index << "-> proc " << currentProc 
            << " PatchCost: " << patchCost << ", ProcCost: " 
            << currentProcCost 
            << ", idcheck: " << patchset[index]->getGridIndex() << endl;
    }
    else {
      // add patch to currentProc
      d_tempAssignment[index] = currentProc;
      currentProcCost += patchCost;
      if (d_myworld->myrank() == 0)
        dbg << "Patch " << index << "-> proc " << currentProc 
            << " PatchCost: " << patchCost << ", ProcCost: " 
            << currentProcCost 
            << ", idcheck: " << patchset[index]->getGridIndex() << endl;
    }
  }

  bool doLoadBalancing = force || thresholdExceeded(patch_costs);

  time = Time::currentSeconds() - time;
  if (d_myworld->myrank() == 0)
    dbg << " Time to LB: " << time << endl;
  doing << d_myworld->myrank() << "   APF END\n";
  return doLoadBalancing;
}

bool DynamicLoadBalancer::thresholdExceeded(const vector<float>& patch_costs)
{
  // add up the costs each processor for the current assignment
  // and for the temp assignment, then calculate the standard deviation
  // for both.  If (curStdDev / tmpStdDev) > threshold, return true,
  // and have possiblyDynamicallyRelocate change the load balancing
  
  int numProcs = d_myworld->size();
  vector<float> currentProcCosts(numProcs);
  vector<float> tempProcCosts(numProcs);
  
  for (unsigned i = 0; i < d_tempAssignment.size(); i++) {
    currentProcCosts[d_processorAssignment[i]] += patch_costs[i];
    tempProcCosts[d_tempAssignment[i]] += patch_costs[i];
  }
  
  // use the std dev formula:
  // sqrt((n*sum_of_squares - sum squared)/n squared)
  float sum_of_current = 0;
  float sum_of_current_squares = 0;
  float sum_of_temp = 0;
  float sum_of_temp_squares = 0;

  for (int i = 0; i < d_myworld->size(); i++) {
    sum_of_current += currentProcCosts[i];
    sum_of_current_squares += currentProcCosts[i]*currentProcCosts[i];
    sum_of_temp += tempProcCosts[i];
    sum_of_temp_squares += tempProcCosts[i]*tempProcCosts[i];
  }
  
  float curStdDev = sqrt((numProcs*sum_of_current_squares - sum_of_current*sum_of_current)/(numProcs*numProcs));
  float tmpStdDev = sqrt((numProcs*sum_of_temp_squares - sum_of_temp*sum_of_temp)/(numProcs*numProcs));

  if (d_myworld->myrank() == 0)
    dbg << "CurrStdDev: " << curStdDev << " tmp " << tmpStdDev 
        << " threshold: " << curStdDev/tmpStdDev << " minT " << d_lbThreshold << endl;

  // if cur / tmp is greater than 1, it is an improvement
  if (curStdDev / tmpStdDev >= d_lbThreshold)
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
  if (d_state == restartLoadBalance) {
    // on a restart, nothing happens on the first execute, and then a recompile
    // happens, but we already have the LB set up how we want from restart
    // initialize, so do nothing here
    d_state = postLoadBalance;
    d_lastLbTime = d_sharedState->getElapsedTime();
    d_lastLbTimestep = d_sharedState->getCurrentTopLevelTimeStep();
    return false;
  }
  if (d_dynamicAlgorithm == static_lb && d_state != postLoadBalance)
    // should only happen on the first timestep
    return d_state != idle; 

  int old_state = d_state;

  double time = d_sharedState->getElapsedTime();
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();

  //cout << d_myworld->myrank() << " PLB recompile: "
  // << time << d_lbTimestepInterval << ' ' << ' ' << d_lastLbTimestep
  //   << timestep << ' ' << d_lbInterval << ' ' << ' ' << d_lastLbTime << endl;

  if (d_lbTimestepInterval != 0 && timestep >= d_lastLbTimestep + d_lbTimestepInterval) {
    d_lastLbTimestep = timestep;
    d_state = checkLoadBalance;
  }
  else if (d_lbInterval != 0 && time >= d_lastLbTime + d_lbInterval) {
    d_lastLbTime = time;
    d_state = checkLoadBalance;
  }
  
  // if we check for lb every timestep, but don't, we still need to recompile
  // if we load balanced on the last timestep
  if (possiblyDynamicallyReallocate(grid, false) || old_state == postLoadBalance) {
    doing << d_myworld->myrank() << " PLB - scheduling recompile " <<endl;
    return true;
  }
  else {
    doing << d_myworld->myrank() << " PLB - NOT scheduling recompile " <<endl;
    return false;
  }
    } 

void
DynamicLoadBalancer::restartInitialize(ProblemSpecP& pspec, string tsurl, const GridP& grid)
{
  // here we need to grab the uda data to reassign patch data to the 
  // processor that will get the data
  d_state = idle;

  int numPatches = 0;
  int startingID = (*(grid->getLevel(0)->patchesBegin()))->getID();

  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    numPatches += level->numPatches();
  }

  d_processorAssignment.resize(numPatches);

  d_state = idle;
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

  if (dbg.active()) {
    if (d_myworld->myrank() == 0) {
      dbg << d_myworld->myrank() << " POST RESTART\n";
      for (unsigned i = 0; i < d_processorAssignment.size(); i++) {
        dbg <<d_myworld-> myrank() << " patch " << i << " -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") - " << d_processorAssignment.size() << ' ' << d_oldAssignment.size() << "\n";
      }
    }
  }
}

bool DynamicLoadBalancer::possiblyDynamicallyReallocate(const GridP& grid, bool force)
{
  doing << d_myworld->myrank() << " In PLB - State: " << d_state << endl;

  if (force)
    d_state = checkLoadBalance;

  // if the last timestep was a load balance, we need to recompile the 
  // task graph
  if (d_state == postLoadBalance) {
    d_oldAssignment = d_processorAssignment;
    d_state = idle;
    doing << d_myworld->myrank() << "  Changing LB state from postLB to idle\n";
  }
  else if (d_state != idle) {
    int numProcs = d_myworld->size();
    int numPatches = 0;
    
    for(int l=0;l<grid->numLevels();l++){
      const LevelP& level = grid->getLevel(l);
      numPatches += level->numPatches();
    }
    
    int myrank = d_myworld->myrank();
    
    DataWarehouse* dw = d_scheduler->get_dw(0);
    
    if (dw == 0) {
      // on the first timestep, just assign the patches in a simple fashion
      // then on the next timestep do the real work
      d_processorAssignment.resize(numPatches);
      
      for(int l=0;l<grid->numLevels();l++){
        const LevelP& level = grid->getLevel(l);
        
        for (Level::const_patchIterator iter = level->patchesBegin(); 
             iter != level->patchesEnd(); iter++) {
          Patch *patch = *iter;
          int patchid = patch->getGridIndex();
          int levelidx = patch->getLevelIndex();
          d_processorAssignment[patchid] = levelidx * numProcs / patch->getLevel()->numPatches();
          ASSERTRANGE(patchid,0,numPatches);
        }
      }
      
      if (dw == 0)
        d_oldAssignment = d_processorAssignment;
      if (d_dynamicAlgorithm != static_lb)
        d_state = checkLoadBalance;  // l.b. on next timestep
      else
        d_state = postLoadBalance;
    }
    else {
      //d_oldAssignment = d_processorAssignment;
      d_tempAssignment.clear();
      d_tempAssignment.resize(numPatches);
      if (d_myworld->myrank() == 0)
      doing << d_myworld->myrank() << "  Checking whether we need to LB\n";
      bool dynamicAllocate = false;
      switch (d_dynamicAlgorithm) {
      case patch_factor_lb:  dynamicAllocate = assignPatchesFactor(grid, force); break;
      case cyclic_lb:        dynamicAllocate = assignPatchesCyclic(grid, force); break;
      case random_lb:        dynamicAllocate = assignPatchesRandom(grid, force); break;
        // static_lb does nothing
      }
      if (dynamicAllocate || force) {
        d_oldAssignment = d_processorAssignment;
        d_processorAssignment = d_tempAssignment;
        
        d_state = postLoadBalance;
        
        if (lb.active()) {
          if (myrank == 0) {
            LevelP curLevel = grid->getLevel(0);
            Level::const_patchIterator iter = curLevel->patchesBegin();
            lb << "  Changing the Load Balance\n";
            vector<int> costs(numProcs);
            for (int i = 0; i < numPatches; i++) {
              lb << myrank << " patch " << i << " -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") patch size: "  << (*iter)->getGridIndex() << " " << ((*iter)->getHighIndex() - (*iter)->getLowIndex()) << "\n";
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
      else {
        d_state = idle;
      }
    }
  }

  // this must be called here (it creates the new per-proc patch sets) even if DLB does nothing.  Don't move or return earlier.
  LoadBalancerCommon::possiblyDynamicallyReallocate(grid, d_state != idle);
  return d_state != idle;
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
    if(!p->get("timestepInterval", timestepInterval))
      timestepInterval = 0;
    if (timestepInterval != 0 && !p->get("interval", interval))
      interval = 0.0; // default
    p->getWithDefault("dynamicAlgorithm", dynamicAlgo, "static");
    p->getWithDefault("cellFactor", cellFactor, .1);
    p->getWithDefault("gainThreshold", threshold, 0.0);
    p->getWithDefault("doSpaceCurve", spaceCurve, false);
    p->getWithDefault("timeRefinementWeight", d_timeRefineWeight, false);
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
  else if (dynamicAlgo == "static")
    d_dynamicAlgorithm = static_lb;
  else {
    if (d_myworld->myrank() == 0)
     cout << "Invalid Load Balancer Algorithm: " << dynamicAlgo
           << "\nPlease select 'cyclic', 'random', 'patchFactor', \n"
           << "'patchFactorParticles' ('particle3'), or 'static' (default)\n"
           << "\nUsing 'static' load balancer\n";
    d_dynamicAlgorithm = static_lb;
  }
  d_lbInterval = interval;
  d_lbTimestepInterval = timestepInterval;
  d_doSpaceCurve = spaceCurve;
  d_lbThreshold = threshold;
  d_cellFactor = cellFactor;
}
