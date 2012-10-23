/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
//Allgatherv currently performs poorly on Kraken.  
//This hack changes the Allgatherv to an allgather 
//by padding the digits
#define AG_HACK  

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/LoadBalancers/ParticleLoadBalancer.h>
#include <Core/Grid/Grid.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Time.h>
#include <Core/Exceptions/InternalError.h>
#include <CCA/Components/LoadBalancers/CostModeler.h>
#include <CCA/Components/LoadBalancers/CostModelForecaster.h>

#include <iostream> // debug only
#include <stack>
#include <vector>
using namespace Uintah;
using namespace SCIRun;
using namespace std;
static DebugStream doing("ParticleLoadBalancer_doing", false);
static DebugStream lb("ParticleLoadBalancer_lb", false);
static DebugStream dbg("ParticleLoadBalancer", false);
static DebugStream stats("LBStats",false);
static DebugStream times("LBTimes",false);
static DebugStream lbout("LBOut",false);

//if defined the space-filling curve will be computed in parallel, this may not be a good idea because the time to compute 
//the space-filling curve is so small that it might not parallelize well.
#define SFC_PARALLEL  

ParticleLoadBalancer::ParticleLoadBalancer(const ProcessorGroup* myworld)
  : LoadBalancerCommon(myworld), sfc(myworld)
{
  d_lbInterval = 0.0;
  d_lastLbTime = 0.0;
  d_lbTimestepInterval = 0;
  d_lastLbTimestep = 0;
  d_checkAfterRestart = false;
  d_pspec = 0;

  d_assignmentBasePatch = -1;
  d_oldAssignmentBasePatch = -1;

}

ParticleLoadBalancer::~ParticleLoadBalancer()
{
}

void ParticleLoadBalancer::collectParticlesForRegrid(const Grid* oldGrid, const vector<vector<Region> >& newGridRegions, vector<vector<int> >& particles)
{
  // collect particles from the old grid's patches onto processor 0 and then distribute them
  // (it's either this or do 2 consecutive load balances).  For now, it's safe to assume that
  // if there is a new level or a new patch there are no particles there.

  int numProcs = d_myworld->size();
  int myrank = d_myworld->myrank();
  int num_patches = 0;

  particles.resize(newGridRegions.size());
  for (unsigned i = 0; i < newGridRegions.size(); i++)
  {
    particles[i].resize(newGridRegions[i].size());
    num_patches += newGridRegions[i].size();
  }
  
  vector<int> recvcounts(numProcs,0); // init the counts to 0
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
          low = Max(region.getLow(), oldPatch->getExtraCellLowIndex());
          high = Min(region.getHigh(), oldPatch->getExtraCellHighIndex());

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
    vector<int> displs(numProcs,0);
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

  //add the number of particles to the cost array
  unsigned level = 0;
  unsigned index = 0;

  for (unsigned i = 0; i < num_particles.size(); i++, index++) {
    if (particles[level].size() <= index) {
      index = 0;
      level++;
    }
    particles[level][index] += num_particles[i];
  }

  if (dbg.active() && d_myworld->myrank() == 0) {
    for (unsigned i = 0; i < num_particles.size(); i++) {
      dbg << d_myworld->myrank() << "  Post gather index " << i << ": " << " numP : " << num_particles[i] << endl;
    }
  }

}

void ParticleLoadBalancer::collectParticles(const Grid* grid, vector<vector<int> >& particles)
{
  particles.resize(grid->numLevels());
  for(int l=0;l<grid->numLevels();l++)
  {
    particles[l].resize(grid->getLevel(l)->numPatches());
    particles[l].assign(grid->getLevel(l)->numPatches(),0);
  }
  if (d_processorAssignment.size() == 0)
    return; // if we haven't been through the LB yet, don't try this.

  if (d_myworld->myrank() == 0)
    dbg << " DLB::collectParticles\n";

  int num_patches = 0;
  for (int i = 0; i < grid->numLevels(); i++)
    num_patches += grid->getLevel(i)->numPatches();

  int numProcs = d_myworld->size();
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
  for(int l=0;l<grid->numLevels();l++) {
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
    vector<int> displs(numProcs, 0);
    vector<int> recvcounts(numProcs,0); // init the counts to 0

    // order patches by processor #, to determine recvcounts easily
    //vector<int> sorted_processorAssignment = d_processorAssignment;
    //sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());
    vector<PatchInfo> all_particles(num_patches);

    for (int i = 0; i < (int)d_processorAssignment.size(); i++) {
      recvcounts[d_processorAssignment[i]]+=sizeof(PatchInfo);
    }

    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

#ifdef AG_HACK
    //compute maximum elements across all processors
    int max_size=recvcounts[0];
    for(int p=1;p<d_myworld->size();p++)
      if(max_size<recvcounts[p])
        max_size=recvcounts[p];

    //create temporary vectors
    vector<PatchInfo> particleList2(particleList), all_particles2;
    particleList2.resize(max_size/sizeof(PatchInfo));
    all_particles2.resize(particleList2.size()*d_myworld->size());

    //gather regions
    MPI_Allgather(&particleList2[0],max_size,MPI_BYTE,&all_particles2[0],max_size, MPI_BYTE,d_myworld->getComm());

    //copy to original vectors
    int j=0;
    for(int p=0;p<d_myworld->size();p++)
    {
      int start=particleList2.size()*p;
      int end=start+recvcounts[p]/sizeof(PatchInfo);
      for(int i=start;i<end;i++)
        all_particles[j++]=all_particles2[i];          
    }

    //free memory
    particleList2.clear();
    all_particles2.clear();

#else
    MPI_Allgatherv(&particleList[0], particleList.size()*sizeof(PatchInfo),  MPI_BYTE,
        &all_particles[0], &recvcounts[0], &displs[0], MPI_BYTE,
        d_myworld->getComm());
#endif    
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
    {
      num_particles[particleList[i].id] = particleList[i].numParticles;
    }
  }

  // add the number of particles to the particles array
  for (int l = 0, i=0; l < grid->numLevels(); l++) {
    unsigned num_patches=grid->getLevel(l)->numPatches();
    for(unsigned p =0; p<num_patches; p++,i++)
    {
      particles[l][p]=num_particles[i];
    }
  }
}

void ParticleLoadBalancer::useSFC(const LevelP& level, int* order)
{
  MALLOC_TRACE_TAG_SCOPE("ParticleLoadBalancer::useSFC");
  vector<DistributedIndex> indices; //output
  vector<double> positions;

  //this should be removed when dimensions in shared state is done
  int dim=d_sharedState->getNumDims();
  int *dimensions=d_sharedState->getActiveDims();

  IntVector min_patch_size(INT_MAX,INT_MAX,INT_MAX);  

  // get the overall range in all dimensions from all patches
  IntVector high(INT_MIN,INT_MIN,INT_MIN);
  IntVector low(INT_MAX,INT_MAX,INT_MAX);
#ifdef SFC_PARALLEL 
  vector<int> originalPatchCount(d_myworld->size(),0); //store how many patches each patch has originally
#endif
  for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) 
  {
    const Patch* patch = *iter;
   
    //calculate patchset bounds
    high = Max(high, patch->getCellHighIndex());
    low = Min(low, patch->getCellLowIndex());
    
    //calculate minimum patch size
    IntVector size=patch->getCellHighIndex()-patch->getCellLowIndex();
    min_patch_size=min(min_patch_size,size);
    
    //create positions vector

#ifdef SFC_PARALLEL
    //place in long longs to avoid overflows with large numbers of patches and processors
    long long pindex=patch->getLevelIndex();
    long long num_patches=d_myworld->size();
    long long proc = (pindex*num_patches) /(long long)level->numPatches();

    ASSERTRANGE(proc,0,d_myworld->size());
    if(d_myworld->myrank()==(int)proc)
    {
      Vector point=(patch->getCellLowIndex()+patch->getCellHighIndex()).asVector()/2.0;
      for(int d=0;d<dim;d++)
      {
        positions.push_back(point[dimensions[d]]);
      }
    }
    originalPatchCount[proc]++;
#else
    Vector point=(patch->getCellLowIndex()+patch->getCellHighIndex()).asVector()/2.0;
    for(int d=0;d<dim;d++)
    {
      positions.push_back(point[dimensions[d]]);
    }
#endif

  }

#ifdef SFC_PARALLEL
  //compute patch starting locations
  vector<int> originalPatchStart(d_myworld->size(),0);
  for(int p=1;p<d_myworld->size();p++)
  {
    originalPatchStart[p]=originalPatchStart[p-1]+originalPatchCount[p-1];
  }
#endif

  //patchset dimensions
  IntVector range = high-low;
  
  //center of patchset
  Vector center=(high+low).asVector()/2.0;
 
  double r[3]={(double)range[dimensions[0]], (double)range[dimensions[1]], (double)range[dimensions[2]]};
  double c[3]={(double)center[dimensions[0]],(double)center[dimensions[1]], (double)center[dimensions[2]]};
  double delta[3]={(double)min_patch_size[dimensions[0]], (double)min_patch_size[dimensions[1]], (double)min_patch_size[dimensions[2]]};


  //create SFC
  sfc.SetDimensions(r);
  sfc.SetCenter(c);
  sfc.SetRefinementsByDelta(delta); 
  sfc.SetLocations(&positions);
  sfc.SetOutputVector(&indices);
  
#ifdef SFC_PARALLEL
  sfc.SetLocalSize(originalPatchCount[d_myworld->myrank()]);
  sfc.GenerateCurve();
#else
  sfc.SetLocalSize(level->numPatches());
  sfc.GenerateCurve(SERIAL);
#endif
  
#ifdef SFC_PARALLEL
  if(d_myworld->size()>1)  
  {
    vector<int> recvcounts(d_myworld->size(), 0);
    vector<int> displs(d_myworld->size(), 0);
    
    for (unsigned i = 0; i < recvcounts.size(); i++)
    {
      displs[i]=originalPatchStart[i]*sizeof(DistributedIndex);
      if(displs[i]<0)
        throw InternalError("Displacments < 0",__FILE__,__LINE__);
      recvcounts[i]=originalPatchCount[i]*sizeof(DistributedIndex);
      if(recvcounts[i]<0)
        throw InternalError("Recvcounts < 0",__FILE__,__LINE__);
    }

    vector<DistributedIndex> rbuf(level->numPatches());

    //gather curve
    MPI_Allgatherv(&indices[0], recvcounts[d_myworld->myrank()], MPI_BYTE, &rbuf[0], &recvcounts[0], 
                   &displs[0], MPI_BYTE, d_myworld->getComm());

    indices.swap(rbuf);
  
  }

  //convert distributed indices to normal indices
  for(size_t i=0;i<indices.size();i++)
  {
    DistributedIndex di=indices[i];
    order[i]=originalPatchStart[di.p]+di.i;
  }
#else
  //write order array
  for(size_t i=0;i<indices.size();i++)
  {
    order[i]=indices[i].i;
  }
#endif

#if 0
  cout << "SFC order: ";
  for (int i = 0; i < level->numPatches(); i++) 
  {
    cout << order[i] << " ";
  }
  cout << endl;
#endif
#if 0
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
#endif
}
void ParticleLoadBalancer::assignPatches( const vector<double> &previousProcCosts, const vector<double> &patchCosts, vector<int> &patches, vector<int> &assignments )
{
  if(patches.size()==0)
    return;

  int numProcs=d_myworld->size();

  //resize vectors to appropriate size
  assignments.resize(patches.size());

  double totalCost=0; //the total amount of work 
  
  //add previous assignment costs to total cost
  for(size_t p=0;p<previousProcCosts.size();p++)
    totalCost+=previousProcCosts[p];

  //add in work that we are assignming
  for(size_t p=0;p<patches.size();p++)
    totalCost+=patchCosts[patches[p]];

  //compute average work per processor (which is our target)
  double avgCostPerProc=totalCost/numProcs;
  double bestMaxCost=totalCost; //best max across all processors
  int minProcLoc = -1; //which processor has the best max
  double myStoredMax=DBL_MAX;
  double improvement; //how much the max was improrved from the last iteration

  vector<int> tempAssignments(patches.size(),0);
  assignments.resize(patches.size());
  do
  {
    double currentMaxCost=0;  
    //give each processor a different max to try to achieve
    double myMaxCost =bestMaxCost-(bestMaxCost-avgCostPerProc)/d_myworld->size()*(double)d_myworld->myrank();
    
    vector<double> currentProcCosts(numProcs,0);
    double remainingCost=totalCost;
    int currentProc=0;

    //assign patches to processors
    for(size_t p=0;p<patches.size();p++)
    {
      // assign the patch to a processor.  When we advance procs,
      // re-update the cost, so we use all procs (and don't go over)
      double patchCost = patchCosts[patches[p]];
      double notakeimb=fabs(previousProcCosts[currentProc]+currentProcCosts[currentProc]-avgCostPerProc);
      double takeimb=fabs(previousProcCosts[currentProc]+currentProcCosts[currentProc]+patchCost-avgCostPerProc);

      if ( previousProcCosts[currentProc]+currentProcCosts[currentProc]+patchCost<myMaxCost && takeimb<=notakeimb) {
        // add patch to currentProc
        tempAssignments[p] = currentProc;
        currentProcCosts[currentProc] += patchCost;
      }
      else {
        //if maximum is better then update the max
        if(previousProcCosts[currentProc]+currentProcCosts[currentProc]>currentMaxCost)
          currentMaxCost=previousProcCosts[currentProc]+currentProcCosts[currentProc];

        //subtract currentProc's cost from remaining cost
        remainingCost -= (currentProcCosts[currentProc]+previousProcCosts[currentProc]);
        

        // move to next proc 
        currentProc++;
        
        //update average (this ensures we don't over/under fill to much)
        avgCostPerProc = remainingCost / (numProcs-currentProc);

        //if currentProc to large then load balance is invalid so break out
        if(currentProc>=numProcs)
          break;


        //retry assingment loop on this patch
        p--;
      }
    }
    //check if last proc is the max
    if(currentProc<numProcs && previousProcCosts[currentProc]+currentProcCosts[currentProc]>currentMaxCost)
      currentMaxCost=previousProcCosts[currentProc]+currentProcCosts[currentProc];


    //if the max was lowered and the assignments are valid
    if(currentMaxCost<bestMaxCost && currentProc<numProcs)
    {
      assignments.swap(tempAssignments);

      //update the max I have stored
      myStoredMax=currentMaxCost;

    }

    double_int maxInfo(myStoredMax,d_myworld->myrank());
    double_int min;

    //gather the maxes
    //change to all reduce with loc
    if(numProcs>1)
      MPI_Allreduce(&maxInfo,&min,1,MPI_DOUBLE_INT,MPI_MINLOC,d_myworld->getComm());    
    else
      min=maxInfo;

    improvement=bestMaxCost-min.val;

    if(min.val<bestMaxCost)
    {
      //set hardMax
      bestMaxCost=min.val;
      //set minloc
      minProcLoc=min.loc;
    }

    //compute max on proc
  } while (improvement>0); 
    
  if(minProcLoc!=-1 && numProcs>1)
  {
    //broadcast load balance
    MPI_Bcast(&assignments[0],assignments.size(),MPI_INT,minProcLoc,d_myworld->getComm());
  }
#if 0
  if(d_myworld->myrank()==0)
  {
    cout << " Assignments in function: ";
    for(int p=0;p<assignments.size();p++)
    {
      cout << assignments[p] << " ";
    }
    cout << endl;
  }
#endif
}

bool ParticleLoadBalancer::loadBalanceGrid(const GridP& grid, bool force)
{
  MALLOC_TRACE_TAG_SCOPE("ParticleLoadBalancer::loadBalanceGrid");
  doing << d_myworld->myrank() << "   APF\n";
  vector<vector<double> > cellCosts;
  vector<vector<double> > particleCosts;

  int numProcs = d_myworld->size();

  getCosts(grid.get_rep(),particleCosts,cellCosts);

  //for each level
  for(int l=0;l<grid->numLevels();l++)
  {
    const LevelP& level = grid->getLevel(l);
    int num_patches = level->numPatches();

    //sort the patches in SFC order
    vector<int> order(num_patches);
    useSFC(level, &order[0]);

    vector<int> cellPatches, particlePatches;

    //split patches into particle/cell patches
    for(int p=0;p<num_patches;p++)
    {
      if(particleCosts[l][order[p]]>cellCosts[l][order[p]])
        particlePatches.push_back(order[p]);
      else
        cellPatches.push_back(order[p]);
    }

    proc0cout << "ParticleLoadBalancer: ParticlePatches: " << particlePatches.size() << " cellPatches: " << cellPatches.size() << endl;

    vector<double> procCosts(numProcs);
    vector<int> assignments;
    
    //assign particlePatches
    assignPatches( procCosts, particleCosts[l], particlePatches,assignments);
    //for each particlePatch
    
    for(size_t p=0;p<particlePatches.size();p++)
    {
      int patch=particlePatches[p];
      int proc=assignments[p];
      //set assignment
      d_tempAssignment[patch]=proc;
      //update procCosts (using cellCost)
      procCosts[proc]+=cellCosts[l][patch];
    }

    assignments.clear();
    
    //assign cellPatches
    assignPatches( procCosts, cellCosts[l], cellPatches, assignments);
    //for each cellPatch
    for(size_t p=0;p<cellPatches.size();p++)
    {
      int patch=cellPatches[p];
      int proc=assignments[p];
      //set assignment
      d_tempAssignment[patch]=proc;
    }

  }

  if(stats.active() && d_myworld->myrank()==0)
  {
    double cellImb, partImb;

    //calculate lb stats based on particles
    cellImb=computeImbalance(cellCosts);
    //calculate lb stats based on cells
    partImb=computeImbalance(particleCosts);

    stats << "Load Imbalance, cellImb: " << cellImb << " partImb: " << partImb << endl;
  }  

  //need to rewrite thresholdExceeded to take into account cells and particles
  bool doLoadBalancing = force || thresholdExceeded(cellCosts,particleCosts);

  return doLoadBalancing;
}

double ParticleLoadBalancer::computeImbalance(const vector<vector<double> >& costs)
{
  int numProcs = d_myworld->size();
  int numLevels = costs.size();
  vector<vector<double> > tempProcCosts(numLevels);
  
  //compute the assignment costs
  int i=0;
  for(int l=0;l<numLevels;l++)
  {
    tempProcCosts[l].resize(numProcs);
    tempProcCosts[l].assign(numProcs,0);

    for(int p=0;p<(int)costs[l].size();p++,i++)
    {
      tempProcCosts[l][d_tempAssignment[i]] += costs[l][p];
    }
  }
#if 0
  if(d_myworld->myrank()==0)
  {
    for(int l=0;l<numLevels;l++)
    {
      cout << "ProcCosts: level: " << l << ", ";
      for(int p=0;p<numProcs;p++)
        cout << tempProcCosts[l][p] << " ";
      cout << endl;
    }
  }
#endif

  double total_max_temp=0, total_avg_temp=0;

  for(int i=0;i<d_myworld->size();i++)
  {
    double temp_cost=0;
    for(int l=0;l<numLevels;l++)
    {
      temp_cost+=tempProcCosts[l][i];
    }
    if(temp_cost>total_max_temp)
      total_max_temp=temp_cost;
    total_avg_temp+=temp_cost;
  }
  total_avg_temp/=d_myworld->size();

  return (total_max_temp-total_avg_temp)/total_avg_temp;
}
double ParticleLoadBalancer::computePercentImprovement(const vector<vector<double> >& costs, double &avg, double &max)
{
  int numProcs = d_myworld->size();
  int numLevels = costs.size();

  vector<vector<double> > currentProcCosts(numLevels);
  vector<vector<double> > tempProcCosts(numLevels);
  
  //compute the assignment costs
  int i=0;
  for(int l=0;l<numLevels;l++)
  {
    currentProcCosts[l].resize(numProcs);
    tempProcCosts[l].resize(numProcs);

    currentProcCosts[l].assign(numProcs,0);
    tempProcCosts[l].assign(numProcs,0);

    for(int p=0;p<(int)costs[l].size();p++,i++)
    {
      currentProcCosts[l][d_processorAssignment[i]] += costs[l][p];
      tempProcCosts[l][d_tempAssignment[i]] += costs[l][p];
    }
  }
  
  double total_max_current=0, total_avg_current=0;
  double total_max_temp=0, total_avg_temp=0;

  for(int i=0;i<d_myworld->size();i++)
  {
    double current_cost=0, temp_cost=0;
    for(int l=0;l<numLevels;l++)
    {
      current_cost+=currentProcCosts[l][i];
      temp_cost+=tempProcCosts[l][i];
    }
    if(current_cost>total_max_current)
      total_max_current=current_cost;
    if(temp_cost>total_max_temp)
      total_max_temp=temp_cost;
    total_avg_current+=current_cost;
    total_avg_temp+=temp_cost;
  }
  total_avg_current/=d_myworld->size();
  total_avg_temp/=d_myworld->size();

  max=total_max_temp;
  avg=total_avg_temp;

  //return the percent improvement
  return (total_max_current-total_max_temp)/total_max_current;
}

bool ParticleLoadBalancer::thresholdExceeded(const vector<vector<double> >& cellCosts, const vector<vector<double> > & partCosts)
{

  double cellMax=0, cellAvg=0, partMax=0, partAvg=0;
  double cellImp=computePercentImprovement(cellCosts,cellAvg,cellMax);
  double partImp=computePercentImprovement(partCosts,partAvg,partMax);
  
  if (d_myworld->myrank() == 0)
    stats << "Total:"  << " Load Balance:  Cell Improvement:" << cellImp << " Particle Improvement:"  << partImp << endl;

  if((cellImp+partImp)/2>d_lbThreshold)
  {
    return true;
  }
  else
  {
    return false;
  }

}

int
ParticleLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch)
{
  // if on a copy-data timestep and we ask about an old patch, that could cause problems
  if (d_sharedState->isCopyDataTimestep() && patch->getRealPatch()->getID() < d_assignmentBasePatch)
    return -patch->getID();
 
  ASSERTRANGE(patch->getRealPatch()->getID(), d_assignmentBasePatch, d_assignmentBasePatch + (int) d_processorAssignment.size());
  int proc = d_processorAssignment[patch->getRealPatch()->getGridIndex()];

  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}

int
ParticleLoadBalancer::getOldProcessorAssignment(const VarLabel* var, 
						const Patch* patch, 
                                                const int /*matl*/)
{

  if (var && var->typeDescription()->isReductionVariable()) {
    return d_myworld->myrank();
  }

  // on an initial-regrid-timestep, this will get called from createNeighborhood
  // and can have a patch with a higher index than we have
  if ((int)patch->getRealPatch()->getID() < d_oldAssignmentBasePatch || patch->getRealPatch()->getID() >= d_oldAssignmentBasePatch + (int)d_oldAssignment.size())
    return -9999;
  
  if (patch->getGridIndex() >= (int) d_oldAssignment.size())
    return -999;

  int proc = d_oldAssignment[patch->getRealPatch()->getGridIndex()];
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}

bool 
ParticleLoadBalancer::needRecompile(double /*time*/, double /*delt*/, 
				    const GridP& grid)
{
  double time = d_sharedState->getElapsedTime();
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();

  bool do_check = false;
#if 1
  if (d_lbTimestepInterval != 0 && timestep >= d_lastLbTimestep + d_lbTimestepInterval) {
    d_lastLbTimestep = timestep;
    do_check = true;
  }
  else if (d_lbInterval != 0 && time >= d_lastLbTime + d_lbInterval) {
    d_lastLbTime = time;
    do_check = true;
  }
  else if (time == 0 || d_checkAfterRestart) {
    // do AFTER initialization timestep too (no matter how much init regridding),
    // so we can compensate for new particles
    do_check = true;
    d_checkAfterRestart = false;
  }
#endif

//  if (dbg.active() && d_myworld->myrank() == 0)
//    dbg << d_myworld->myrank() << " DLB::NeedRecompile: check=" << do_check << " ts: " << timestep << " " << d_lbTimestepInterval << " t " << time << " " << d_lbInterval << " last: " << d_lastLbTimestep << " " << d_lastLbTime << endl;

  // if it determines we need to re-load-balance, recompile
  if (do_check && possiblyDynamicallyReallocate(grid, check)) {
    doing << d_myworld->myrank() << " PLB - scheduling recompile " <<endl;
    return true;
  }
  else {
    d_oldAssignment = d_processorAssignment;
    d_oldAssignmentBasePatch = d_assignmentBasePatch;
    return false;
  }
} 

void
ParticleLoadBalancer::restartInitialize( DataArchive* archive, int time_index, ProblemSpecP& pspec,
                                        string tsurl, const GridP& grid )
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

          ProblemSpecP dataDoc = ProblemSpecReader().readInputFile( dataxml );
          if( !dataDoc ) {
            throw InternalError( string( "Cannot open data file: " ) + dataxml, __FILE__, __LINE__);
          }
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
    dbg << d_myworld->myrank() << " check after restart: " << d_checkAfterRestart << "\n";
#if 0
    int startPatch = (int) (*grid->getLevel(0)->patchesBegin())->getID();
    if (lb.active()) {
      for (unsigned i = 0; i < d_processorAssignment.size(); i++) {
        lb <<d_myworld-> myrank() << " patch " << i << " (real " << i+startPatch << ") -> proc " 
           << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") - " 
           << d_processorAssignment.size() << ' ' << d_oldAssignment.size() << "\n";
      }
    }
#endif
  }
}

//if it is not a regrid the patch information is stored in grid, if it is during a regrid the patch information is stored in patches
void ParticleLoadBalancer::getCosts(const Grid* grid, vector<vector<double> >&particle_costs, vector<vector<double> > &cell_costs)
{
  MALLOC_TRACE_TAG_SCOPE("ParticleLoadBalancer::getCosts");
  particle_costs.clear();
  cell_costs.clear();
    
  vector<vector<int> > num_particles;

  DataWarehouse* olddw = d_scheduler->get_dw(0);
  bool on_regrid = olddw != 0 && grid != olddw->getGrid();

  //collect the number of particles on each processor into num_particles
  if(on_regrid)
  {
    vector<vector<Region> > regions;
    // prepare the list of regions
    for (int l = 0; l < grid->numLevels(); l++) {
      regions.push_back(vector<Region>());
      for (int p = 0; p < grid->getLevel(l)->numPatches(); p++) {
        const Patch* patch = grid->getLevel(l)->getPatch(p);
        regions[l].push_back(Region(patch->getCellLowIndex(), patch->getCellHighIndex()));
      }
    }
    collectParticlesForRegrid(olddw->getGrid(),regions,num_particles);
  }
  else
  {
    collectParticles(grid, num_particles);
  }

  //for each patch set the costs equal to the number of cells and number of particles
  for (int l = 0; l < grid->numLevels(); l++) 
  {
    LevelP level=grid->getLevel(l);
    cell_costs.push_back(vector<double>());
    particle_costs.push_back(vector<double>());
    for (int p = 0; p < grid->getLevel(l)->numPatches(); p++) 
    {
      cell_costs[l].push_back(level->getPatch(p)->getNumCells()*d_cellCost);
      particle_costs[l].push_back(num_particles[l][p]*d_particleCost);
    }
#if 0
    if(d_myworld->myrank()==0)
    {
      cout << " Level: " << l << " cellCosts: ";
      for (int p = 0; p < grid->getLevel(l)->numPatches(); p++) 
      {
        cout << cell_costs[l][p] << " ";
      }
      cout << endl;
      cout << " Level: " << l << " particleCosts: ";
      for (int p = 0; p < grid->getLevel(l)->numPatches(); p++) 
      {
        cout << particle_costs[l][p] << " ";
      }
      cout << endl;
    }
#endif
  }
}

bool ParticleLoadBalancer::possiblyDynamicallyReallocate(const GridP& grid, int state)
{
  MALLOC_TRACE_TAG_SCOPE("ParticleLoadBalancer::possiblyDynamicallyReallocate");
  TAU_PROFILE("ParticleLoadBalancer::possiblyDynamicallyReallocate()", " ", TAU_USER);

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
      dynamicAllocate = loadBalanceGrid(grid, force); 
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
        int myrank = d_myworld->myrank();
        if (myrank == 0) {
          LevelP curLevel = grid->getLevel(0);
          Level::const_patchIterator iter = curLevel->patchesBegin();
          lb << "  Changing the Load Balance\n";
          for (size_t i = 0; i < d_processorAssignment.size(); i++) {
            lb << myrank << " patch " << i << " (real " << (*iter)->getID() << ") -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") patch size: "  << (*iter)->getNumExtraCells() << " low:" << (*iter)->getExtraCellLowIndex() << " high: " << (*iter)->getExtraCellHighIndex() <<"\n";
            IntVector range = ((*iter)->getExtraCellHighIndex() - (*iter)->getExtraCellLowIndex());
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
ParticleLoadBalancer::problemSetup(ProblemSpecP& pspec, GridP& grid,  SimulationStateP& state)
{
  proc0cout << "Warning the ParticleLoadBalancer is experimental, use at your own risk\n";

  LoadBalancerCommon::problemSetup(pspec, grid, state);
  
  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  double interval = 0;
  int timestepInterval = 10;
  double threshold = 0.0;
  bool spaceCurve = false;

  if (p != 0) {
    // if we have DLB, we know the entry exists in the input file...
    if(!p->get("timestepInterval", timestepInterval))
      timestepInterval = 0;
    if (timestepInterval != 0 && !p->get("interval", interval))
      interval = 0.0; // default
    p->getWithDefault("gainThreshold", threshold, 0.05);
    p->getWithDefault("doSpaceCurve", spaceCurve, true);
    p->getWithDefault("particleCost",d_particleCost, 2);
    p->getWithDefault("cellCost",d_cellCost, 1);
   
  }

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

