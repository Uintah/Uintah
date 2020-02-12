/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/LoadBalancers/ParticleLoadBalancer.h>

#include <CCA/Components/LoadBalancers/CostModeler.h>
#include <CCA/Components/LoadBalancers/CostModelForecaster.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/Timers/Timers.hpp>

#include <iostream> // debug only
#include <stack>
#include <vector>

using namespace Uintah;

namespace {
  DebugStream doing( "ParticleLoadBalancer_doing", "LoadBalancers", "", false );
  DebugStream lb(    "ParticleLoadBalancer_lb",    "LoadBalancers", "", false );
  DebugStream dbg(   "ParticleLoadBalancer",       "LoadBalancers", "", false );
}

ParticleLoadBalancer::ParticleLoadBalancer( const ProcessorGroup * myworld ) :
  LoadBalancerCommon(myworld)
{
  m_lb_interval = 0.0;
  m_last_lb_simTime = 0.0;
  m_lb_timeStep_interval = 0;
  m_last_lb_timeStep = 0;
  m_check_after_restart = false;
  d_pspec = 0;

  m_assignment_base_patch = -1;
  m_old_assignment_base_patch = -1;
}

ParticleLoadBalancer::~ParticleLoadBalancer()
{
}

void
ParticleLoadBalancer::collectParticlesForRegrid( const Grid* oldGrid, const std::vector<std::vector<Region> >& newGridRegions, std::vector<std::vector<int> >& particles )
{
  // collect particles from the old grid's patches onto processor 0 and then distribute them
  // (it's either this or do 2 consecutive load balances).  For now, it's safe to assume that
  // if there is a new level or a new patch there are no particles there.

  int numProcs = d_myworld->nRanks();
  int myRank = d_myworld->myRank();
  int num_patches = 0;

  particles.resize(newGridRegions.size());
  for (unsigned i = 0; i < newGridRegions.size(); i++)
  {
    particles[i].resize(newGridRegions[i].size());
    num_patches += newGridRegions[i].size();
  }
  
  std::vector<int> recvcounts(numProcs,0); // init the counts to 0
  int totalsize = 0;

  DataWarehouse* dw = m_scheduler->get_dw(0);
  if (dw == 0)
    return;

  std::vector<PatchInfo> subpatchParticles;
  unsigned grid_index = 0;
  for(unsigned l=0;l<newGridRegions.size();l++){
    const std::vector<Region>& level = newGridRegions[l];
    for (unsigned r = 0; r < level.size(); r++, grid_index++) {
      const Region& region = level[r];;

      if (l >= (unsigned) oldGrid->numLevels()) {
        // new patch - no particles yet
        recvcounts[0]++;
        totalsize++;
        if (d_myworld->myRank() == 0) {
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
        if (d_myworld->myRank() == 0) {
          PatchInfo pi(grid_index, 0);
          subpatchParticles.push_back(pi);
        }
        continue;
      }

      for (unsigned int i = 0; i < oldPatches.size(); i++) {
        const Patch* oldPatch = oldPatches[i];

        recvcounts[m_processor_assignment[oldPatch->getGridIndex()]]++;
        totalsize++;
        if (m_processor_assignment[oldPatch->getGridIndex()] == myRank) {
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
            for (unsigned int m = 0; m < m_materialManager->getNumMatls(); m++) {
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

  std::vector<int> num_particles(num_patches, 0);

  if (d_myworld->nRanks() > 1) {
    //construct a mpi datatype for the PatchInfo
    MPI_Datatype particletype;
    Uintah::MPI::Type_contiguous(2, MPI_INT, &particletype);
    Uintah::MPI::Type_commit(&particletype);

    std::vector<PatchInfo> recvbuf(totalsize);
    std::vector<int> displs(numProcs,0);
    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

    Uintah::MPI::Gatherv(&subpatchParticles[0], recvcounts[d_myworld->myRank()], particletype, &recvbuf[0],
        &recvcounts[0], &displs[0], particletype, 0, d_myworld->getComm());

    if ( d_myworld->myRank() == 0) {
      for (unsigned i = 0; i < recvbuf.size(); i++) {
        PatchInfo& spi = recvbuf[i];
        num_particles[spi.m_id] += spi.m_num_particles;
      }
    }
    // combine all the subpatches results
    Uintah::MPI::Bcast(&num_particles[0], num_particles.size(), MPI_INT,0,d_myworld->getComm());
    Uintah::MPI::Type_free(&particletype);
  }
  else {
    for (unsigned i = 0; i < subpatchParticles.size(); i++) {
      PatchInfo& spi = subpatchParticles[i];
      num_particles[spi.m_id] += spi.m_num_particles;
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

  if (dbg.active() && d_myworld->myRank() == 0) {
    for (unsigned i = 0; i < num_particles.size(); i++) {
      dbg << d_myworld->myRank() << "  Post gather index " << i << ": " << " numP : " << num_particles[i] << std::endl;
    }
  }

}

void ParticleLoadBalancer::collectParticles(const Grid* grid, std::vector<std::vector<int> >& particles)
{
  particles.resize(grid->numLevels());
  for(int l=0;l<grid->numLevels();l++)
  {
    particles[l].resize(grid->getLevel(l)->numPatches());
    particles[l].assign(grid->getLevel(l)->numPatches(),0);
  }
  if (m_processor_assignment.size() == 0)
    return; // if we haven't been through the LB yet, don't try this.

  if (d_myworld->myRank() == 0)
    dbg << " DLB::collectParticles\n";

  int num_patches = 0;
  for (int i = 0; i < grid->numLevels(); i++)
    num_patches += grid->getLevel(i)->numPatches();

  int numProcs = d_myworld->nRanks();
  int myRank = d_myworld->myRank();
  // get how many particles were each patch had at the end of the last timestep
  //   gather from each proc - based on the last location

  DataWarehouse* dw = m_scheduler->get_dw(0);
  if (dw == 0)
    return;

  std::vector<PatchInfo> particleList;
  std::vector<int> num_particles(num_patches, 0);

  // find out how many particles per patch, and store that number
  // along with the patch number in particleList
  for(int l=0;l<grid->numLevels();l++) {
    const LevelP& level = grid->getLevel(l);
    for (Level::const_patch_iterator iter = level->patchesBegin(); 
        iter != level->patchesEnd(); iter++) {
      Patch *patch = *iter;
      int id = patch->getGridIndex();
      if (m_processor_assignment[id] != myRank)
        continue;
      int thisPatchParticles = 0;

      if (dw) {
        //loop through the materials and add up the particles
        //   go through all materials since getting an MPMMaterial correctly would depend on MPM
        for (unsigned int m = 0; m < m_materialManager->getNumMatls(); m++) {
          if (dw->haveParticleSubset(m, patch))
            thisPatchParticles += dw->getParticleSubset(m, patch)->numParticles();
        }
      }
      // add to particle list
      PatchInfo p(id,thisPatchParticles);
      particleList.push_back(p);
      dbg << "  Pre gather " << id << " part: " << thisPatchParticles << std::endl;
    }
  }

  if (d_myworld->nRanks() > 1) {
    //construct a mpi datatype for the PatchInfo
    std::vector<int> displs(numProcs, 0);
    std::vector<int> recvcounts(numProcs,0); // init the counts to 0

    // order patches by processor #, to determine recvcounts easily
    //std::vector<int> sorted_processorAssignment = d_processorAssignment;
    //sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());
    std::vector<PatchInfo> all_particles(num_patches);

    for (int i = 0; i < (int)m_processor_assignment.size(); i++) {
      recvcounts[m_processor_assignment[i]]+=sizeof(PatchInfo);
    }

    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

    Uintah::MPI::Allgatherv(&particleList[0], particleList.size()*sizeof(PatchInfo),  MPI_BYTE,
                   &all_particles[0], &recvcounts[0], &displs[0], MPI_BYTE, d_myworld->getComm());

    if (dbg.active() && d_myworld->myRank() == 0) {
      for (unsigned i = 0; i < all_particles.size(); i++) {
        PatchInfo& pi = all_particles[i];
        dbg << d_myworld->myRank() << "  Post gather index " << i << ": " << pi.m_id << " numP : " << pi.m_num_particles << std::endl;
      }
    }
    for (int i = 0; i < num_patches; i++) {
      num_particles[all_particles[i].m_id] = all_particles[i].m_num_particles;
    }
  }
  else {
    for (int i = 0; i < num_patches; i++)
    {
      num_particles[particleList[i].m_id] = particleList[i].m_num_particles;
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

void
ParticleLoadBalancer::assignPatches( const std::vector<double> &previousProcCosts, const std::vector<double> &patchCosts, std::vector<int> &patches, std::vector<int> &assignments )
{
  if(patches.size()==0)
    return;

  int numProcs=d_myworld->nRanks();

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

  std::vector<int> tempAssignments(patches.size(),0);
  assignments.resize(patches.size());
  do
  {
    double currentMaxCost=0;  
    //give each processor a different max to try to achieve
    double myMaxCost =bestMaxCost-(bestMaxCost-avgCostPerProc)/d_myworld->nRanks()*(double)d_myworld->myRank();
    
    std::vector<double> currentProcCosts(numProcs,0);
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

    double_int maxInfo(myStoredMax,d_myworld->myRank());
    double_int min;

    //gather the maxes
    //change to all reduce with loc
    if(numProcs>1)
      Uintah::MPI::Allreduce(&maxInfo,&min,1,MPI_DOUBLE_INT,MPI_MINLOC,d_myworld->getComm());    
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
    Uintah::MPI::Bcast(&assignments[0],assignments.size(),MPI_INT,minProcLoc,d_myworld->getComm());
  }
#if 0
  if(d_myworld->myRank()==0)
  {
    std::cout << " Assignments in function: ";
    for(int p=0;p<assignments.size();p++)
    {
      std::cout << assignments[p] << " ";
    }
    std::cout << std::endl;
  }
#endif
}

bool ParticleLoadBalancer::loadBalanceGrid(const GridP& grid, bool force)
{
  doing << d_myworld->myRank() << "   APF\n";
  std::vector<std::vector<double> > cellCosts;
  std::vector<std::vector<double> > particleCosts;

  int numProcs = d_myworld->nRanks();

  getCosts(grid.get_rep(),particleCosts,cellCosts);

  //for each level
  for(int l=0;l<grid->numLevels();l++)
  {
    const LevelP& level = grid->getLevel(l);
    int num_patches = level->numPatches();

    //sort the patches in SFC order
    std::vector<int> order(num_patches);
    useSFC(level, &order[0]);

    std::vector<int> cellPatches, particlePatches;

    //split patches into particle/cell patches
    for(int p=0;p<num_patches;p++)
    {
      if(particleCosts[l][order[p]]>cellCosts[l][order[p]])
        particlePatches.push_back(order[p]);
      else
        cellPatches.push_back(order[p]);
    }

    proc0cout << "ParticleLoadBalancer: ParticlePatches: " << particlePatches.size() << " cellPatches: " << cellPatches.size() << std::endl;

    std::vector<double> procCosts(numProcs);
    std::vector<int> assignments;
    
    //assign particlePatches
    assignPatches( procCosts, particleCosts[l], particlePatches,assignments);
    //for each particlePatch
    
    for(size_t p=0;p<particlePatches.size();p++)
    {
      int patch=particlePatches[p];
      int proc=assignments[p];
      //set assignment
      m_temp_assignment[patch]=proc;
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
      m_temp_assignment[patch]=proc;
    }

  }

  if(stats.active() && d_myworld->myRank()==0)
  {
    double cellImb, partImb;

    //calculate lb stats based on particles
    cellImb=computeImbalance(cellCosts);
    //calculate lb stats based on cells
    partImb=computeImbalance(particleCosts);

    stats << "Load Imbalance, cellImb: " << cellImb << " partImb: " << partImb << std::endl;
  }  

  //need to rewrite thresholdExceeded to take into account cells and particles
  bool doLoadBalancing = force || thresholdExceeded(cellCosts,particleCosts);

  return doLoadBalancing;
}

double ParticleLoadBalancer::computeImbalance(const std::vector<std::vector<double> >& costs)
{
  int numProcs = d_myworld->nRanks();
  int numLevels = costs.size();
  std::vector<std::vector<double> > tempProcCosts(numLevels);
  
  //compute the assignment costs
  int i=0;
  for(int l=0;l<numLevels;l++)
  {
    tempProcCosts[l].resize(numProcs);
    tempProcCosts[l].assign(numProcs,0);

    for(int p=0;p<(int)costs[l].size();p++,i++)
    {
      tempProcCosts[l][m_temp_assignment[i]] += costs[l][p];
    }
  }
#if 0
  if(d_myworld->myRank()==0)
  {
    for(int l=0;l<numLevels;l++)
    {
      std::cout << "ProcCosts: level: " << l << ", ";
      for(int p=0;p<numProcs;p++)
        std::cout << tempProcCosts[l][p] << " ";
      std::cout << std::endl;
    }
  }
#endif

  double total_max_temp=0, total_avg_temp=0;

  for(int i=0;i<d_myworld->nRanks();i++)
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
  total_avg_temp/=d_myworld->nRanks();

  return (total_max_temp-total_avg_temp)/total_avg_temp;
}
double ParticleLoadBalancer::computePercentImprovement(const std::vector<std::vector<double> >& costs, double &avg, double &max)
{
  int numProcs = d_myworld->nRanks();
  int numLevels = costs.size();

  std::vector<std::vector<double> > currentProcCosts(numLevels);
  std::vector<std::vector<double> > tempProcCosts(numLevels);
  
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
      currentProcCosts[l][m_processor_assignment[i]] += costs[l][p];
      tempProcCosts[l][m_temp_assignment[i]] += costs[l][p];
    }
  }
  
  double total_max_current=0, total_avg_current=0;
  double total_max_temp=0, total_avg_temp=0;

  for(int i=0;i<d_myworld->nRanks();i++)
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
  total_avg_current/=d_myworld->nRanks();
  total_avg_temp/=d_myworld->nRanks();

  max=total_max_temp;
  avg=total_avg_temp;

  //return the percent improvement
  return (total_max_current-total_max_temp)/total_max_current;
}

bool ParticleLoadBalancer::thresholdExceeded(const std::vector<std::vector<double> >& cellCosts, const std::vector<std::vector<double> > & partCosts)
{

  double cellMax=0, cellAvg=0, partMax=0, partAvg=0;
  double cellImp=computePercentImprovement(cellCosts,cellAvg,cellMax);
  double partImp=computePercentImprovement(partCosts,partAvg,partMax);
  
  if (d_myworld->myRank() == 0)
    stats << "Total:"  << " Load Balance:  Cell Improvement:" << cellImp << " Particle Improvement:"  << partImp << std::endl;

  if((cellImp+partImp)/2>d_lbThreshold)
  {
    return true;
  }
  else
  {
    return false;
  }

}

bool 
ParticleLoadBalancer::needRecompile(const GridP& grid)
{
  const int timeStep   = m_application->getTimeStep();
  const double simTime = m_application->getSimTime();

  bool do_check = false;
#if 1
  if (m_lb_timeStep_interval != 0 &&
      timeStep >= m_last_lb_timeStep + m_lb_timeStep_interval) {
    m_last_lb_timeStep = timeStep;
    do_check = true;
  }
  else if (m_lb_interval != 0 && simTime >= m_last_lb_simTime + m_lb_interval) {
    m_last_lb_simTime = simTime;
    do_check = true;
  }
  else if (simTime == 0 || m_check_after_restart) {
    // do AFTER initialization timestep too (no matter how much init regridding),
    // so we can compensate for new particles
    do_check = true;
    m_check_after_restart = false;
  }
#endif

//  if (dbg.active() && d_myworld->myRank() == 0)
//    dbg << d_myworld->myRank() << " DLB::NeedRecompile: check=" << do_check << " ts: " << timestep << " " << m_lb_timeStep_interval << " t " << time << " " << d_lbInterval << " last: " << m_last_lb_timeStep << " " << d_lastLbTime << std::endl;

  // if it determines we need to re-load-balance, recompile
  if (do_check && possiblyDynamicallyReallocate(grid, LoadBalancer::CHECK_LB)) {
    doing << Uintah::Parallel::getMPIRank() << " PLB - scheduling recompile\n";
    return true;
  }
  else {
    m_old_assignment = m_processor_assignment;
    m_old_assignment_base_patch = m_assignment_base_patch;
    return false;
  }
} 

//if it is not a regrid the patch information is stored in grid, if it is during a regrid the patch information is stored in patches
void ParticleLoadBalancer::getCosts(const Grid* grid, std::vector<std::vector<double> >&particle_costs, std::vector<std::vector<double> > &cell_costs)
{
  particle_costs.clear();
  cell_costs.clear();
    
  std::vector<std::vector<int> > num_particles;

  DataWarehouse* olddw = m_scheduler->get_dw(0);
  bool on_regrid = olddw != 0 && grid != olddw->getGrid();

  //collect the number of particles on each processor into num_particles
  if(on_regrid)
  {
    std::vector<std::vector<Region> > regions;
    // prepare the list of regions
    for (int l = 0; l < grid->numLevels(); l++) {
      regions.push_back(std::vector<Region>());
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
    cell_costs.push_back(std::vector<double>());
    particle_costs.push_back(std::vector<double>());
    for (int p = 0; p < grid->getLevel(l)->numPatches(); p++) 
    {
      cell_costs[l].push_back(level->getPatch(p)->getNumCells()*d_cellCost);
      particle_costs[l].push_back(num_particles[l][p]*d_particleCost);
    }
#if 0
    if( Uintah::Parallel::getMPIRank() == 0 )
    {
      std::cout << " Level: " << l << " cellCosts: ";
      for (int p = 0; p < grid->getLevel(l)->numPatches(); p++) 
      {
        std::cout << cell_costs[l][p] << " ";
      }
      std::cout << std::endl;
      std::cout << " Level: " << l << " particleCosts: ";
      for (int p = 0; p < grid->getLevel(l)->numPatches(); p++) 
      {
        std::cout << particle_costs[l][p] << " ";
      }
      std::cout << std::endl;
    }
#endif
  }
}

bool
ParticleLoadBalancer::possiblyDynamicallyReallocate(const GridP& grid, int state)
{
  if( Uintah::Parallel::getMPIRank() == 0 ) {
    dbg << d_myworld->myRank() << " In DLB, state " << state << "\n";
  }

  Timers::Simple timer;
  timer.start();

  const int timeStep   = m_application->getTimeStep();
  const double simTime = m_application->getSimTime();

  bool changed = false;
  bool force = false;

  // don't do on a restart unless procs changed between runs.  For restarts, this is 
  // called mainly to update the perProc Patch sets (at the bottom)
  if( state != LoadBalancer::RESTART_LB ) {
    if( state != LoadBalancer::CHECK_LB ) {
      force = true;
      if (m_lb_timeStep_interval != 0) {
        m_last_lb_timeStep = timeStep;
      }
      else if (m_lb_interval != 0) {
        m_last_lb_simTime = simTime;
      }
    }
    m_old_assignment = m_processor_assignment;
    m_old_assignment_base_patch = m_assignment_base_patch;
    
    bool dynamicAllocate = false;
    //temp assignment can be set if the regridder has already called the load balancer
    if(m_temp_assignment.empty())
    {
      int num_patches = 0;
      for(int l=0;l<grid->numLevels();l++){
        const LevelP& level = grid->getLevel(l);
        num_patches += level->numPatches();
      }
    
      m_temp_assignment.resize(num_patches);
      dynamicAllocate = loadBalanceGrid(grid, force); 
    }
    else  //regridder has called dynamic load balancer so we must dynamically Allocate
    {
      dynamicAllocate=true;
    }

    if( dynamicAllocate || state != LoadBalancer::CHECK_LB ) {
      //d_oldAssignment = d_processorAssignment;
      changed = true;
      m_processor_assignment = m_temp_assignment;
      m_assignment_base_patch = (*grid->getLevel(0)->patchesBegin())->getID();

      if( state == LoadBalancer::INIT_LB ) {
        // set it up so the old and new are in same place
        m_old_assignment = m_processor_assignment;
        m_old_assignment_base_patch = m_assignment_base_patch;
      }
        
      if (lb.active()) {
        int myRank = Uintah::Parallel::getMPIRank();
        if (myRank == 0) {
          LevelP curLevel = grid->getLevel(0);
          Level::const_patch_iterator iter = curLevel->patchesBegin();
          lb << "  Changing the Load Balance\n";
          
          for (size_t i = 0; i < m_processor_assignment.size(); i++) {
            lb << myRank << " patch " << i << " (real " << (*iter)->getID() << ") -> proc " << m_processor_assignment[i] << " (old " << m_old_assignment[i] << ") patch size: "  << (*iter)->getNumExtraCells() << " low:" << (*iter)->getExtraCellLowIndex() << " high: " << (*iter)->getExtraCellHighIndex() <<"\n";
            IntVector range = ((*iter)->getExtraCellHighIndex() - (*iter)->getExtraCellLowIndex());
            iter++;
            if (iter == curLevel->patchesEnd() && i+1 < m_processor_assignment.size()) {
              curLevel = curLevel->getFinerLevel();
              iter = curLevel->patchesBegin();
            }
          }
        }
      }
    }
  }
  m_temp_assignment.resize(0);
  
  // logic to setting flag
  int flag = LoadBalancer::CHECK_LB;
  if ( changed || state == LoadBalancer::RESTART_LB ){
    flag = LoadBalancer::REGRID_LB;
  }
  
  // this must be called here (it creates the new per-proc patch sets)
  // even if DLB does nothing.  Don't move or return earlier.
  LoadBalancerCommon::possiblyDynamicallyReallocate( grid, flag );

  (*d_runtimeStats)[LoadBalancerTime] += timer().seconds();
  
  return changed;
}

void
ParticleLoadBalancer::problemSetup(ProblemSpecP& pspec, GridP& grid, const MaterialManagerP& materialManager)
{
  proc0cout << "Warning the ParticleLoadBalancer is experimental, use at your own risk\n";

  LoadBalancerCommon::problemSetup( pspec, grid, materialManager );
  
  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  double interval = 0;
  int timestepInterval = 10;
  double threshold = 0.0;
  bool spaceCurve = false;

  if( p != nullptr ) {
    // if we have DLB, we know the entry exists in the input file...
    if( !p->get("timestepInterval", timestepInterval) ) {
      timestepInterval = 0;
    }
    if( timestepInterval != 0 && !p->get("interval", interval) ) {
      interval = 0.0; // default
    }
    p->getWithDefault("gainThreshold", threshold, 0.05);
    p->getWithDefault("doSpaceCurve", spaceCurve, true);
    p->getWithDefault("particleCost",d_particleCost, 2);
    p->getWithDefault("cellCost",d_cellCost, 1);
   
  }

  m_lb_timeStep_interval = timestepInterval;
  m_do_space_curve = spaceCurve;
  d_lbThreshold = threshold;

  // Set curve parameters that do not change between timesteps
  ASSERT( m_numDims > 0 || m_numDims < 4);
  
  m_sfc.SetNumDimensions(m_numDims);
  m_sfc.SetMergeMode(1);
  m_sfc.SetCleanup(BATCHERS);
  m_sfc.SetMergeParameters(3000,500,2,.15);  //Should do this by profiling
}

