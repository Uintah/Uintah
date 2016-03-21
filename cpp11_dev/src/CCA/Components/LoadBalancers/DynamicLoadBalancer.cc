/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>

#include <CCA/Components/LoadBalancers/CostModeler.h>
#include <CCA/Components/LoadBalancers/CostModelForecaster.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Regridder.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Timers/Timers.hpp>

#include <iostream> // debug only
#include <stack>
#include <vector>

using namespace Uintah;
using namespace std;

static DebugStream doing( "DynamicLoadBalancer_doing", false );
static DebugStream lb(    "DynamicLoadBalancer_lb", false );
static DebugStream dbg(   "DynamicLoadBalancer", false );
static DebugStream stats( "LBStats", false );
static DebugStream times( "LBTimes", false );
static DebugStream lbout( "LBOut", false );

double lbtimes[5] = {0,0,0,0,0};

DynamicLoadBalancer::DynamicLoadBalancer( const ProcessorGroup * myworld )
  : LoadBalancerCommon{myworld}
  , d_costForecaster{0}
{
  d_lbInterval = 0.0;
  d_lastLbTime = 0.0;
  d_lbTimestepInterval = 0;
  d_lastLbTimestep = 0;
  d_checkAfterRestart = false;

  d_dynamicAlgorithm = patch_factor_lb;
  d_collectParticles = false;

  d_do_AMR = false;
  d_pspec = 0;

  d_assignmentBasePatch = -1;
  d_oldAssignmentBasePatch = -1;
}
//______________________________________________________________________
//
DynamicLoadBalancer::~DynamicLoadBalancer()
{
  if( d_costForecaster ) {
    delete d_costForecaster;
    d_costForecaster = 0;
  }
}
//______________________________________________________________________
//
void
DynamicLoadBalancer::collectParticlesForRegrid( const Grid                     * oldGrid,
                                                const vector< vector<Region> > & newGridRegions,
                                                vector< vector<int> >          & particles )
{
  // Collect particles from the old grid's patches onto processor 0 and then distribute them
  // (it's either this or do 2 consecutive load balances).  For now, it's safe to assume that
  // if there is a new level or a new patch there are no particles there.

  int num_procs = d_myworld->size();
  int myrank = d_myworld->myrank();
  int num_patches = 0;

  particles.resize(newGridRegions.size());
  for( unsigned i = 0; i < newGridRegions.size(); i++ ) {
    particles[i].resize(newGridRegions[i].size());
    num_patches += newGridRegions[i].size();
  }

  if( !d_collectParticles ) {
    return;
  }

  vector<int> recvcounts(num_procs,0); // init the counts to 0
  int totalsize = 0;

  DataWarehouse* dw = d_scheduler->get_dw(0);
  if (dw == 0) {
    return;
  }

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
    MPI::Type_contiguous(2, MPI_INT, &particletype);
    MPI::Type_commit(&particletype);

    vector<PatchInfo> recvbuf(totalsize);
    vector<int> displs(num_procs,0);
    for (unsigned i = 1; i < displs.size(); i++) {
      displs[i] = displs[i-1]+recvcounts[i-1];
    }

    MPI::Gatherv(&subpatchParticles[0], recvcounts[d_myworld->myrank()], particletype, &recvbuf[0],
        &recvcounts[0], &displs[0], particletype, 0, d_myworld->getComm());

    if ( d_myworld->myrank() == 0) {
      for (unsigned i = 0; i < recvbuf.size(); i++) {
        PatchInfo& spi = recvbuf[i];
        num_particles[spi.id] += spi.numParticles;
      }
    }
    // combine all the subpatches results
    MPI::Bcast(&num_particles[0], num_particles.size(), MPI_INT,0,d_myworld->getComm());
    MPI::Type_free(&particletype);
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
//______________________________________________________________________
//
void
DynamicLoadBalancer::collectParticles( const Grid                  * grid,
                                             vector< vector<int> > & particles )
{
  particles.resize(grid->numLevels());
  for(int l=0;l<grid->numLevels();l++) {
    particles[l].resize(grid->getLevel(l)->numPatches());
    particles[l].assign(grid->getLevel(l)->numPatches(),0);
  }
  if( d_processorAssignment.size() == 0 ) {
    return; // if we haven't been through the LB yet, don't try this.
  }

  //if we are not supposed to collect particles just return
  if( !d_collectParticles || !d_scheduler->get_dw(0) ) {
    return;
  }

  if( d_myworld->myrank() == 0 ) {
    dbg << " DLB::collectParticles\n";
  }

  int num_patches = 0;
  for (int i = 0; i < grid->numLevels(); i++) {
    num_patches += grid->getLevel(i)->numPatches();
  }

  int num_procs = d_myworld->size();
  int myrank = d_myworld->myrank();
  // get how many particles were each patch had at the end of the last timestep
  //   gather from each proc - based on the last location

  DataWarehouse* dw = d_scheduler->get_dw(0);
  if( dw == 0 ) {
    return;
  }

  vector<PatchInfo> particleList;
  vector<int> num_particles(num_patches, 0);

  // Find out how many particles per patch, and store that number
  // along with the patch number in particleList.
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
    vector<int> displs(num_procs, 0);
    vector<int> recvcounts(num_procs,0); // init the counts to 0

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


    MPI::Allgatherv(&particleList[0], particleList.size()*sizeof(PatchInfo),  MPI_BYTE,
                   &all_particles[0], &recvcounts[0], &displs[0], MPI_BYTE, d_myworld->getComm());

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

//______________________________________________________________________
//
bool
DynamicLoadBalancer::assignPatchesFactor( const GridP & grid, bool force )
{
  doing << d_myworld->myrank() << "   APF\n";
  vector<vector<double> > patch_costs;

  //------------------------------< begin load balance timing >------------------------------
  Timers::Simple lb_total_timer;

  for (int i = 0; i < 5; i++) {
    lbtimes[i] = 0;
  }

  Timers::Simple individual_lb_times;

  lbtimes[0] += individual_lb_times().seconds();
  individual_lb_times.reset();

  static int lbiter = -1;  //counter to identify which regrid
  lbiter++;

  int num_procs = d_myworld->size();

  getCosts(grid.get_rep(), patch_costs);

  lbtimes[1] += individual_lb_times().seconds();
  individual_lb_times.reset();

  int level_offset = 0;

  vector<double> totalProcCosts(num_procs, 0);
  vector<double> procCosts(num_procs, 0);
  vector<double> previousProcCosts(num_procs, 0);

  double previous_total_cost = 0;

  //__________________________________
  //
  for (int l = 0; l < grid->numLevels(); l++) {

    const LevelP& level = grid->getLevel(l);
    int num_patches = level->numPatches();
    vector<int> order(num_patches);
    double total_cost = 0;

    for (unsigned i = 0; i < patch_costs[l].size(); i++) {
      total_cost += patch_costs[l][i];
    }

    if (d_doSpaceCurve) {
      //cout << d_myworld->myrank() << "   Doing SFC level " << l << endl;
      useSFC(level, &order[0]);
    }

    lbtimes[2] += individual_lb_times().seconds();
    individual_lb_times.reset();

    //hard maximum cost for assigning a patch to a processor
    double avgCost = (total_cost + previous_total_cost) / num_procs;
    double hardMaxCost = total_cost;
    double myMaxCost = hardMaxCost - (hardMaxCost - avgCost) / d_myworld->size() * (double)d_myworld->myrank();
    double myStoredMax = DBL_MAX;
    int minProcLoc = -1;

    if (!force)  //use initial load balance to start the iteration
    {
      //compute costs of current load balance
      vector<double> currentProcCosts(num_procs);

      for (int p = 0; p < (int)patch_costs[l].size(); p++) {
        //copy assignment from current load balance
        d_tempAssignment[level_offset + p] = d_processorAssignment[level_offset + p];
        //add assignment to current costs
        currentProcCosts[d_processorAssignment[level_offset + p]] += patch_costs[l][p];
      }

      //compute maximum of current load balance
      hardMaxCost = currentProcCosts[0];
      for (int i = 1; i < num_procs; i++) {
        if (currentProcCosts[i] + previousProcCosts[i] > hardMaxCost) {
          hardMaxCost = currentProcCosts[i] + previousProcCosts[i];
        }
      }
      double range = hardMaxCost - avgCost;
      myStoredMax = hardMaxCost;
      myMaxCost = hardMaxCost - range / d_myworld->size() * (double)d_myworld->myrank();
    }

    //temperary vector to assign the load balance in
    vector<int> temp_assignment(d_tempAssignment);
    vector<int> maxList(num_procs);

    //__________________________________
    //iterate the load balancing algorithm until the max can no longer be lowered
    int iter = 0;
    double improvement = 1;

    while (improvement > 0) {
      double remainingCost = total_cost + previous_total_cost;
      double avgCostPerProc = remainingCost / num_procs;

      int currentProc = 0;
      vector<double> currentProcCosts(num_procs, 0);
      double currentMaxCost = 0;

      for (int p = 0; p < num_patches; p++) {
        int index;

        if (d_doSpaceCurve) {
          index = order[p];
        } else {
          // not attempting space-filling curve
          index = p;
        }

        // assign the patch to a processor.  When we advance procs,
        // re-update the cost, so we use all procs (and don't go over)
        double patchCost = patch_costs[l][index];
        double notakeimb = fabs(previousProcCosts[currentProc] + currentProcCosts[currentProc] - avgCostPerProc);
        double takeimb = fabs(previousProcCosts[currentProc] + currentProcCosts[currentProc] + patchCost - avgCostPerProc);

        if (previousProcCosts[currentProc] + currentProcCosts[currentProc] + patchCost < myMaxCost && takeimb <= notakeimb) {
          // add patch to currentProc
          temp_assignment[level_offset + index] = currentProc;
          currentProcCosts[currentProc] += patchCost;
        } else {
          if (previousProcCosts[currentProc] + currentProcCosts[currentProc] > currentMaxCost) {
            currentMaxCost = previousProcCosts[currentProc] + currentProcCosts[currentProc];
          }

          //subtract currentProc's cost from remaining cost
          remainingCost -= (currentProcCosts[currentProc] + previousProcCosts[currentProc]);

          // move to next proc
          currentProc++;

          //if currentProc to large then load balance is invalid so break out
          if (currentProc >= num_procs) {
            break;
          }

          //assign patch to currentProc
          temp_assignment[level_offset + index] = currentProc;

          //update average (this ensures we don't over/under fill to much)
          avgCostPerProc = remainingCost / (num_procs - currentProc);
          currentProcCosts[currentProc] = patchCost;

        }
      }

      //check if last proc is the max
      if (currentProc < num_procs && previousProcCosts[currentProc] + currentProcCosts[currentProc] > currentMaxCost) {
        currentMaxCost = previousProcCosts[currentProc] + currentProcCosts[currentProc];
      }

      //__________________________________
      //if the max was lowered and the assignments are valid
      if (currentMaxCost < myStoredMax && currentProc < num_procs) {

#if 1
        //take this assignment
        for (int p = 0; p < num_patches; p++) {
          d_tempAssignment[level_offset + p] = temp_assignment[level_offset + p];
        }
#else
        d_tempAssignment.swap(temp_assignment);
#endif
        //update myMaxCost
        myStoredMax = currentMaxCost;
      }

      double_int maxInfo(myStoredMax, d_myworld->myrank());
      double_int min;

      //gather the maxes
      //change to all reduce with loc
      if (num_procs > 1) {
        MPI::Allreduce(&maxInfo, &min, 1, MPI_DOUBLE_INT, MPI_MINLOC, d_myworld->getComm());
      } else {
        min = maxInfo;
      }

      //set improvement
      improvement = hardMaxCost - min.val;

      if (min.val < hardMaxCost) {
        //set hardMax
        hardMaxCost = min.val;
        //set minloc
        minProcLoc = min.loc;
      }

      //compute average cost per proc
      double average = (total_cost + previous_total_cost) / num_procs;

      //set new myMax by having each processor search at even intervals in the range
      double range = hardMaxCost - average;
      myMaxCost = hardMaxCost - range / d_myworld->size() * (double)d_myworld->myrank();
      iter++;
    }

    lbtimes[3] += individual_lb_times().seconds();
    individual_lb_times.reset();

    if (minProcLoc != -1 && num_procs > 1) {
      //broadcast load balance
      MPI::Bcast(&d_tempAssignment[0], d_tempAssignment.size(), MPI_INT, minProcLoc, d_myworld->getComm());
    }

    if (!d_levelIndependent) {
      //update previousProcCost

      //loop through the assignments for this level and add costs to previousProcCosts
      for (int p = 0; p < num_patches; p++) {
        previousProcCosts[d_tempAssignment[level_offset + p]] += patch_costs[l][p];
      }
      previous_total_cost += total_cost;
    }

    //__________________________________
    //    debugging output
    if (stats.active() && d_myworld->myrank() == 0) {
      //calculate lb stats:
      double totalCost = 0;
      vector<double> procCosts(num_procs, 0);
      vector<int> patchCounts(num_procs, 0);

      for (int p = 0; p < num_patches; p++) {
        int me = d_tempAssignment[level_offset + p];
        totalCost += patch_costs[l][p];
        procCosts[me] += patch_costs[l][p];
        totalProcCosts[me] += patch_costs[l][p];
        patchCounts[me]++;
      }

      double meanCost = totalCost / num_procs;
      double minCost = procCosts[0];
      double maxCost = procCosts[0];
      int maxProc = 0;

      stats << "LoadBalance ProcCosts Level(" << l << ") ";

      // compute min & max
      for (int p = 0; p < num_procs; p++) {
        if (minCost > procCosts[p]) {
          minCost = procCosts[p];
        } else if (maxCost < procCosts[p]) {
          maxCost = procCosts[p];
          maxProc = p;
        }
        stats << p << ":" << procCosts[p] << " ";
      }
      stats << "\nLoadBalance Stats level(" << l << "):" << " Mean: " << meanCost << " Min: " << minCost << " Max: " << maxCost
            << " Imbalance:" << 1 - meanCost / maxCost << " max on:" << maxProc << endl;
    }

    if (lbout.active() && d_myworld->myrank() == 0) {
      for (int p = 0; p < num_patches; p++) {
        int index;  //compute order index
        if (d_doSpaceCurve) {
          index = order[p];
        } else {
          // not attempting space-filling curve
          index = p;
        }

        IntVector sum = (level->getPatch(index)->getCellLowIndex() + level->getPatch(index)->getCellHighIndex());

        Vector loc(sum.x() / 2.0, sum.y() / 2.0, sum.z() / 2.0);
        //output load balance information
        lbout << lbiter << " " << l << " " << index << " " << d_tempAssignment[level_offset + index] << " " << patch_costs[l][index]
              << " " << loc.x() << " " << loc.y() << " " << loc.z() << endl;
      }
    }

    level_offset += num_patches;
    lbtimes[4] += individual_lb_times().seconds();
    individual_lb_times.reset();
  }

  //__________________________________
  //
  if (stats.active() && d_myworld->myrank() == 0) {
    double meanCost = 0;
    double minCost = totalProcCosts[0];
    double maxCost = totalProcCosts[0];
    int maxProc = 0;

    for (int p = 0; p < num_procs; p++) {
      meanCost += totalProcCosts[p];

      if (minCost > totalProcCosts[p]) {
        minCost = totalProcCosts[p];
      } else if (maxCost < totalProcCosts[p]) {
        maxCost = totalProcCosts[p];
        maxProc = p;
      }
    }
    meanCost /= num_procs;

    stats << "LoadBalance Stats total:" << " Mean: " << meanCost << " Min: " << minCost << " Max: " << maxCost << " Imbalance: "
          << 1 - meanCost / maxCost << " max proc on: " << maxProc << endl;

  }

  //__________________________________
  //
  if (times.active()) {
    double avg[5] = { 0 };

    MPI::Reduce(lbtimes, avg, 5, MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm());

    if (d_myworld->myrank() == 0) {
      cout << "LoadBalance Avg Times: ";
      for (int i = 0; i < 5; i++) {
        avg[i] /= d_myworld->size();
        cout << avg[i] << " ";
      }
      cout << endl;
    }

    double max[5] = { 0 };

    MPI::Reduce(lbtimes, max, 5, MPI_DOUBLE, MPI_MAX, 0, d_myworld->getComm());

    if (d_myworld->myrank() == 0) {
      cout << "LoadBalance Max Times: ";
      for (int i = 0; i < 5; i++) {
        cout << max[i] << " ";
      }
      cout << endl;
    }
  }

  //__________________________________
  //
  bool doLoadBalancing = force || thresholdExceeded(patch_costs);

  double total_lb_time = lb_total_timer().seconds();
  //------------------------------< end load balance timing >------------------------------

  if (d_myworld->myrank() == 0) {
    dbg << " Time to LB: " << total_lb_time << std::endl;
  }
  doing << d_myworld->myrank() << "   APF END\n";

  return doLoadBalancing;
}

//______________________________________________________________________
//
bool
DynamicLoadBalancer::thresholdExceeded( const vector< vector<double> >& patch_costs )
{
  // add up the costs each processor for the current assignment
  // and for the temp assignment, then calculate the standard deviation
  // for both.  If (curStdDev / tmpStdDev) > threshold, return true,
  // and have possiblyDynamicallyRelocate change the load balancing

  int num_procs = d_myworld->size();
  int num_levels = patch_costs.size();

  vector<vector<double> > currentProcCosts(num_levels);
  vector<vector<double> > tempProcCosts(num_levels);

  int i = 0;
  for( int l = 0; l < num_levels; l++ ) {
    currentProcCosts[l].resize(num_procs);
    tempProcCosts[l].resize(num_procs);

    currentProcCosts[l].assign(num_procs,0);
    tempProcCosts[l].assign(num_procs,0);

    for(int p=0;p<(int)patch_costs[l].size();p++,i++)
    {
      currentProcCosts[l][d_processorAssignment[i]] += patch_costs[l][p];
      tempProcCosts[l][d_tempAssignment[i]] += patch_costs[l][p];
    }
  }

  double total_max_current=0, total_avg_current=0;
  double total_max_temp=0, total_avg_temp=0;

  if(d_levelIndependent) {
    double avg_current = 0;
    double max_current = 0;
    double avg_temp = 0;
    double max_temp = 0;
    for(int l=0;l<num_levels;l++) {
      avg_current = 0;
      max_current = 0;
      avg_temp = 0;
      max_temp = 0;
      for (int i = 0; i < d_myworld->size(); i++) {
        if (currentProcCosts[l][i] > max_current)
          max_current = currentProcCosts[l][i];
        if (tempProcCosts[l][i] > max_temp)
          max_temp = tempProcCosts[l][i];
        avg_current += currentProcCosts[l][i];
        avg_temp += tempProcCosts[l][i];
      }

      avg_current /= d_myworld->size();
      avg_temp /= d_myworld->size();

      total_max_current+=max_current;
      total_avg_current+=avg_current;
      total_max_temp+=max_temp;
      total_avg_temp+=avg_temp;
    }
  }
  else {
    for(int i=0;i<d_myworld->size();i++) {
      double current_cost=0, temp_cost=0;
      for(int l=0;l<num_levels;l++) {
        current_cost+=currentProcCosts[l][i];
        temp_cost+=currentProcCosts[l][i];
      }
      if(current_cost>total_max_current) {
        total_max_current=current_cost;
      }
      if(temp_cost>total_max_temp) {
          total_max_temp=temp_cost;
      }
      total_avg_current+=current_cost;
      total_avg_temp+=temp_cost;
    }
    total_avg_current/=d_myworld->size();
    total_avg_temp/=d_myworld->size();
  }

  if (d_myworld->myrank() == 0) {
    stats << "Total:"  << " maxCur:" << total_max_current << " maxTemp:"  << total_max_temp << " avgCur:" << total_avg_current << " avgTemp:" << total_avg_temp <<endl;
  }

  // if tmp - cur is positive, it is an improvement
  if( (total_max_current-total_max_temp)/total_max_current>d_lbThreshold) {
    return true;
  }
  else {
    return false;
  }
}

//______________________________________________________________________
//
bool
DynamicLoadBalancer::assignPatchesRandom( const GridP &, bool force )
{
  // this assigns patches in a random form - every time we re-load balance
  // We get a random seed on the first proc and send it out (so all procs
  // generate the same random numbers), and assign the patches accordingly
  // Not a good load balancer - useful for performance comparisons
  // because we should be able to come up with one better than this

  int seed;

  if (d_myworld->myrank() == 0) {
    seed = (int)Time::currentSeconds();
  }

  MPI::Bcast(&seed, 1, MPI_INT,0,d_myworld->getComm());

  srand( seed );

  int num_procs = d_myworld->size();
  int num_patches = (int)d_tempAssignment.size();

  vector<int> proc_record(num_procs,0);
  int max_ppp = num_patches / num_procs;

  for (int i = 0; i < num_patches; i++) {
    int proc = (int) (((float) rand()) / RAND_MAX * num_procs);
    int newproc = proc;

    // only allow so many patches per proc.  Linear probe if necessary,
    // but make sure every proc has some work to do
    while (proc_record[newproc] >= max_ppp) {
      newproc++;
      if (newproc >= num_procs) {
        newproc = 0;
      }
      if (proc == newproc) {
        // let each proc have more - we've been through all procs
        max_ppp++;
      }
    }
    proc_record[newproc]++;
    d_tempAssignment[i] = newproc;
  }
  return true;
}
//______________________________________________________________________
//
bool
DynamicLoadBalancer::assignPatchesCyclic(const GridP&, bool force)
{
  // This assigns patches in a cyclic form - every time we re-load balance
  // we move each patch up one proc - this obviously isn't a very good
  // lb technique, but it tests its capabilities pretty well.

  int num_procs = d_myworld->size();
  for (unsigned i = 0; i < d_tempAssignment.size(); i++) {
    d_tempAssignment[i] = (d_processorAssignment[i] + 1 ) % num_procs;
  }
  return true;
}

//______________________________________________________________________
//
bool
DynamicLoadBalancer::needRecompile(       double /*time*/,
                                          double /*delt*/,
                                    const GridP & grid )
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
  else if ((time == 0 && d_collectParticles == true) || d_checkAfterRestart) {
    // do AFTER initialization timestep too (no matter how much init regridding),
    // so we can compensate for new particles
    do_check = true;
    d_checkAfterRestart = false;
  }
#endif

  if (dbg.active() && d_myworld->myrank() == 0){
    dbg << d_myworld->myrank() << " DLB::NeedRecompile: do_check: " << do_check << ", timestep: " << timestep
        << ", LB:timestepInterval: " << d_lbTimestepInterval << ", time[s]: " << time << ", LB:Interval: " << d_lbInterval
        << ", Last LB timestep: " << d_lastLbTimestep << ", Last LB time[s]: " << d_lastLbTime << endl;
  }

  // if it determines we need to re-load-balance, recompile
  if (do_check && possiblyDynamicallyReallocate(grid, LoadBalancer::check)) {
    doing << d_myworld->myrank() << " DLB - scheduling recompile " <<endl;
    return true;
  }
  else {
    d_oldAssignment = d_processorAssignment;
    d_oldAssignmentBasePatch = d_assignmentBasePatch;
    return false;
  }
}
//______________________________________________________________________
//
// If it is not a regrid the patch information is stored in grid, if it is during a regrid the patch information is stored in patches.
void
DynamicLoadBalancer::getCosts( const Grid * grid, vector< vector<double> > & costs )
{
  costs.clear();

  vector<vector<int> > num_particles;

  DataWarehouse* olddw = d_scheduler->get_dw(0);
  bool on_regrid = olddw != 0 && grid != olddw->getGrid();

  if( on_regrid ) {
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
  else {
    collectParticles(grid, num_particles);
  }

  //check if the forecaster is ready, if it is use it
  if(d_costForecaster->hasData()) {
    //we have data so don't collect particles
    d_costForecaster->getWeights(grid,num_particles,costs);
  }
  else { //otherwise just use a simple cost model (this happens on the first timestep when profiling data doesn't exist)
    CostModeler(d_patchCost,d_cellCost,d_extraCellCost,d_particleCost).getWeights(grid,num_particles,costs);
  }

  //__________________________________
  //  Debugging output
  if (dbg.active() && d_myworld->myrank() == 0) {
    for (unsigned l = 0; l < costs.size(); l++) {
      for (unsigned p = 0; p < costs[l].size(); p++) {
        dbg << "  DLB:getCosts  L: "  << l << " P: " << p << " cost " << costs[l][p] << endl;
      }
    }
  }
}
//______________________________________________________________________
//
bool
DynamicLoadBalancer::possiblyDynamicallyReallocate( const GridP & grid, int state )
{
  MALLOC_TRACE_TAG_SCOPE("DynamicLoadBalancer::possiblyDynamicallyReallocate");

  if (d_myworld->myrank() == 0) {
    dbg << d_myworld->myrank() << " In DLB, state " << state << endl;
  }

  //------------------------------< begin load balance timing >------------------------------
  Timers::Simple lb_timer;

  bool changed = false;
  bool force = false;

  // don't do on a restart unless procs changed between runs.  For restarts, this is
  // called mainly to update the perProc Patch sets (at the bottom)
  if (state != LoadBalancer::restart) {
    if (state != LoadBalancer::check) {
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

    //__________________________________
    //  temp assignment can be set if the regridder has already called the load balancer
    if (d_tempAssignment.empty()) {
      int num_patches = 0;
      for (int l = 0; l < grid->numLevels(); l++) {
        const LevelP& level = grid->getLevel(l);
        num_patches += level->numPatches();
      }

      d_tempAssignment.resize(num_patches);
      switch (d_dynamicAlgorithm) {
        case patch_factor_lb :
          dynamicAllocate = assignPatchesFactor(grid, force);
          break;
        case cyclic_lb :
          dynamicAllocate = assignPatchesCyclic(grid, force);
          break;
        case random_lb :
          dynamicAllocate = assignPatchesRandom(grid, force);
          break;
      }
    }
    else  //regridder has called dynamic load balancer so we must dynamically Allocate
    {
      dynamicAllocate = true;
    }

    //__________________________________
    //
    if (dynamicAllocate || state != LoadBalancer::check) {
      //d_oldAssignment = d_processorAssignment;
      changed = true;
      d_processorAssignment = d_tempAssignment;
      d_assignmentBasePatch = (*grid->getLevel(0)->patchesBegin())->getID();

      if (state == LoadBalancer::init) {
        // set it up so the old and new are in same place
        d_oldAssignment = d_processorAssignment;
        d_oldAssignmentBasePatch = d_assignmentBasePatch;
      }

      //__________________________________
      //  Debugging output
      if (lb.active()) {
        if (d_myworld->myrank() == 0) {
          LevelP curLevel = grid->getLevel(0);
          Level::const_patchIterator iter = curLevel->patchesBegin();
          DOUT(true, "  Possibly changing the Load Balance");
          for (unsigned int i = 0; i < d_processorAssignment.size(); i++) {
            if (d_oldAssignment[i] != d_processorAssignment[i]) {
              DOUT(true, "Patch: " << i << " ID: " << (*iter)->getID()
                   << ", migration from proc: " << d_oldAssignment[i]
                   << " to proc: " << d_processorAssignment[i]
                   << " patch size: " << (*iter)->getNumExtraCells() << " cells");
              iter++;
              if (iter == curLevel->patchesEnd() && i + 1 < d_processorAssignment.size()) {
                curLevel = curLevel->getFinerLevel();
                iter = curLevel->patchesBegin();
              }
            }
          }
        } // rank 0 reporting
      } // lb.active

    }
  }  // != restart

  d_tempAssignment.resize(0);

  int flag = LoadBalancer::check;
  if ( changed || state == LoadBalancer::restart){
    flag = LoadBalancer::regrid;
  }

  // this must be called here (it creates the new per-proc patch sets) even if DLB does nothing.  Don't move or return earlier.
  LoadBalancerCommon::possiblyDynamicallyReallocate(grid, flag);

  double lb_total_time = lb_timer().seconds();
  d_sharedState->d_runTimeStats[SimulationState::LoadBalancerTime] += lb_total_time;
  //------------------------------< end load balancer timing >------------------------------

  return changed;
}
//______________________________________________________________________
//
void
DynamicLoadBalancer::finalizeContributions( const GridP & grid )
{
  d_costForecaster->finalizeContributions(grid);
}

//______________________________________________________________________
//
void
DynamicLoadBalancer::problemSetup( ProblemSpecP & pspec, GridP & grid,  SimulationStateP & state )
{
  LoadBalancerCommon::problemSetup( pspec, grid, state );

  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  string       dynamicAlgo;
  double       interval = 0;
  int          timestepInterval = 10;
  double       threshold = 0.0;
  bool         spaceCurve = false;

  if (p != 0) {
    // if we have DLB, we know the entry exists in the input file...
    if(!p->get("timestepInterval", timestepInterval)) {
      timestepInterval = 0;
    }
    if (timestepInterval != 0 && !p->get("interval", interval)) {
      interval = 0.0; // default
    }
    p->getWithDefault("dynamicAlgorithm", dynamicAlgo, "patchFactor");
    p->getWithDefault("cellCost",         d_cellCost, 1);
    p->getWithDefault("extraCellCost",    d_extraCellCost, 1);
    p->getWithDefault("particleCost",     d_particleCost, 1.25);
    p->getWithDefault("patchCost",        d_patchCost, 16);
    p->getWithDefault("gainThreshold",    threshold, 0.05);
    p->getWithDefault("doSpaceCurve",     spaceCurve, true);
    p->getWithDefault("hasParticles",     d_collectParticles, false);

    string costAlgo="ModelLS";
    p->get("costAlgorithm",costAlgo);
    if(costAlgo=="ModelLS") {
      d_costForecaster= new CostModelForecaster(d_myworld,this,d_patchCost,d_cellCost,d_extraCellCost,d_particleCost);
    }
    else if(costAlgo=="Kalman") {
      int timestepWindow;
      p->getWithDefault("profileTimestepWindow",timestepWindow,10);
      d_costForecaster=new CostProfiler(d_myworld,ProfileDriver::KALMAN,this);
      d_costForecaster->setTimestepWindow(timestepWindow);
      d_collectParticles=false;
    }
    else if(costAlgo=="Memory") {
      int timestepWindow;
      p->getWithDefault("profileTimestepWindow",timestepWindow,10);
      d_costForecaster=new CostProfiler(d_myworld,ProfileDriver::MEMORY,this);
      d_costForecaster->setTimestepWindow(timestepWindow);
      d_collectParticles=false;
    }
    else if(costAlgo=="Model") {
      d_costForecaster=new CostModeler(d_patchCost,d_cellCost,d_extraCellCost,d_particleCost);
    }
    else {
      throw InternalError("Invalid CostAlgorithm in Dynamic Load Balancer\n",__FILE__,__LINE__);
    }

    p->getWithDefault("levelIndependent",d_levelIndependent,true);
  }


  if(d_myworld->myrank()==0) {
    cout << "Dynamic Algorithm: " << dynamicAlgo << endl;
  }

  if (dynamicAlgo == "cyclic") {
    d_dynamicAlgorithm = cyclic_lb;
  }
  else if (dynamicAlgo == "random") {
    d_dynamicAlgorithm = random_lb;
  }
  else if (dynamicAlgo == "patchFactor") {
    d_dynamicAlgorithm = patch_factor_lb;
  }
  else if (dynamicAlgo == "patchFactorParticles" || dynamicAlgo == "particle3") {
    // these are for backward-compatibility
    d_dynamicAlgorithm = patch_factor_lb;
    d_collectParticles = true;
  }
  else {
    if (d_myworld->myrank() == 0) {
      cout << "Invalid Load Balancer Algorithm: " << dynamicAlgo
        << "\nPlease select 'cyclic', 'random', 'patchFactor' (default), or 'patchFactorParticles'\n"
        << "\nUsing 'patchFactor' load balancer\n";
    }
    d_dynamicAlgorithm = patch_factor_lb;
  }

  d_lbInterval = interval;
  d_lbTimestepInterval = timestepInterval;
  d_doSpaceCurve = spaceCurve;
  d_lbThreshold = threshold;

  ASSERT(d_sharedState->getNumDims()>0 || d_sharedState->getNumDims()<4);
  // Set curve parameters that do not change between timesteps
  d_sfc.SetNumDimensions(d_sharedState->getNumDims());
  d_sfc.SetMergeMode(1);
  d_sfc.SetCleanup(BATCHERS);
  d_sfc.SetMergeParameters(3000,500,2,.15);  //Should do this by profiling

  // Set costProfiler mps
  Regridder *regridder = dynamic_cast<Regridder*>(getPort("regridder"));
  if( regridder ) {
    d_costForecaster->setMinPatchSize(regridder->getMinPatchSize());
  }
  else {
    // Query mps from a patch
    const Patch *patch=grid->getLevel(0)->getPatch(0);

    vector<IntVector> mps;
    mps.push_back(patch->getCellHighIndex()-patch->getCellLowIndex());

    d_costForecaster->setMinPatchSize(mps);
  }
}
