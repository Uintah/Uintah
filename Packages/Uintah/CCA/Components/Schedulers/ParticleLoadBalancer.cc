

#include <Packages/Uintah/CCA/Components/Schedulers/ParticleLoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
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

#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLURL.hpp>

#include <iostream> // debug only

using namespace Uintah;
using namespace SCIRun;
using std::cerr;
static DebugStream dbg("ParticleLoadBalancer", false);

#define DAV_DEBUG 0

struct PatchInfo {
  PatchInfo(int i, int n, int a) {id = i; numParticles = n; assigned = a;}
  PatchInfo() {assigned = 0;}
  
  int id;
  int numParticles;
  int assigned;
};

class ParticleCompare {
public:
  inline bool operator()(const PatchInfo& p1, const PatchInfo& p2) const {
    return p1.numParticles < p2.numParticles || 
      ( p1.numParticles == p2.numParticles && p1.id < p2.id);
  }
};

class PatchCompare {
public:
  inline bool operator()(const PatchInfo& p1, const PatchInfo& p2) const {
    return p1.id < p2.id;
  }
};


ParticleLoadBalancer::ParticleLoadBalancer(const ProcessorGroup* myworld)
   : LoadBalancerCommon(myworld)
{
  //need to compensate for restarts
  d_lbInterval = 0.0;
  d_lastLbTime = 0.0;
  d_lbTimestepInterval = 0;
  d_lastLbTimestep = 0;

  d_dynamicAlgorithm = static_lb;  
  d_state = needLoadBalance;
  d_do_AMR = false;
  d_pspec = 0;
}

ParticleLoadBalancer::~ParticleLoadBalancer()
{
}

void ParticleLoadBalancer::assignPatchesParticle(const GridP& grid, 
                                                 const SchedulerP& sch)
{
  double time = Time::currentSeconds();

  int numProcs = d_myworld->size();
  int numPatches = 0;
  int myrank = d_myworld->myrank();
  int i;

  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    numPatches += level->numPatches();
  }

  // TODO - make sure (for AMR) that the patch ids are contiguous.
  //   make realid and myid where myids are guaranteed contiguous.
  //setup for get each patch's number of particles
  vector<PatchInfo> particleList;

  // get how many particles were each patch had at the end of the last timestep
  //   gather from each proc - based on the last location
  SchedulerCommon* sc = const_cast<SchedulerCommon*>(dynamic_cast<const SchedulerCommon*>(sch.get_rep()));

  OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>
    (sc->get_dw(0));

  // proc 0 - order patches by processor #
  vector<int> sorted_processorAssignment = d_processorAssignment;
  sort(sorted_processorAssignment.begin(), sorted_processorAssignment.end());

  vector<int> displs;
  vector<int> recvcounts(numProcs,0); // init the counts to 0

  int offsetProc = 0;
  for (i = 0; i < (int)d_processorAssignment.size(); i++) {
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
      int id = patch->getID();
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
            ParticleSubset* psubset = dw->getParticleSubset(matl, patch);
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
  
  // each proc - gather all particles together
  vector<PatchInfo> allParticles(numPatches);

  //construct a mpi datatype for a pair
  MPI_Datatype particletype;
  MPI_Type_contiguous(3, MPI_INT, &particletype);
  MPI_Type_commit(&particletype);

  MPI_Gatherv(&particleList[0],particleList.size(),particletype,
	      &allParticles[0], &recvcounts[0], &displs[0], particletype,
	      0, d_myworld->getComm());
  
  // proc 0 - associate patches to particles, load balance, 
  //   MPI_Bcast it to all procs
  
  d_processorAssignment.clear();
  d_processorAssignment.resize(numPatches);

  if (myrank == 0) {
    // sort the particle list.  We need the patch number in there as 
    // well in order to look up the patch number later
    int patchesLeft = numPatches;

    //cout << "Post gather\n";
    //for (i = 0; i < (int)allParticles.size(); i++)
    //cout << "Patch: " << allParticles[i].id
    //     << " -> " << allParticles[i].numParticles << endl;

    // find the total cost and the average cost per processor based on the
    //   factor passed in the ups file.

    vector<Patch*> patchset;
    vector<float> patch_costs;
    float avg_costPerProc = 0;
    float totalCost = 0;

    // make a list of Patch*'s and costs per patch

    sort(allParticles.begin(), allParticles.end(), PatchCompare());
    for(int l=0;l<grid->numLevels();l++){
      const LevelP& level = grid->getLevel(l);
      for (Level::const_patchIterator iter = level->patchesBegin(); 
           iter != level->patchesEnd(); iter++) {
        Patch* patch = *iter;
        int id = patch->getID();
        // are patch id's in order of level and then patch?
        ASSERT(id == allParticles[id].id);
        patchset.push_back(patch);
        IntVector range = patch->getHighIndex() - patch->getLowIndex();
        float cost = allParticles[id].numParticles + d_cellFactor * 
          range.x() * range.y() * range.z();
        patch_costs.push_back(cost);
        totalCost += cost;
      }
    }
      
    avg_costPerProc = totalCost / numProcs;
    

    if (d_particleAlgo == 1) {
      sort(allParticles.begin(), allParticles.end(), ParticleCompare());
    
      
      int minPatch = 0;
      int maxPatch = numPatches-1;
      
    
      // assignment algorithm: loop through the processors (repeatedly) until we
      //   have no more patches to assign.  If, there are twice as many (or more)
      //   patches than processors, than assign the patch with the most and the
      //   the least particles to the current processor.  Otherwise, assign the 
      //   the patch with the most particles to the current processor.
      while (patchesLeft > 0) {
        for (i = 0; i < numProcs; i++) {
          int highpatch = allParticles[maxPatch].id;
          int lowpatch = allParticles[minPatch].id;
          if (patchesLeft >= 2*(numProcs-i)) {
            // give it min and max
            d_processorAssignment[highpatch]=i;
            maxPatch--;
            d_processorAssignment[lowpatch]=i;
            minPatch++;
            patchesLeft -= 2;
            
          } else if (patchesLeft > 0) {
            //give it max
            d_processorAssignment[maxPatch]=i;
            maxPatch--;
            patchesLeft--;
          } else if (patchesLeft == 0) {
            break;
          }
        }
      }
    }
    else if (d_particleAlgo == 2) {
      cout << " LB - PPP " << avg_costPerProc << endl;
      int procnum = 0;
      for (int p = 0; p < numPatches; p++) {
        if (allParticles[p].assigned)
          continue;
        int patches_per_proc = patchesLeft / (numProcs-(procnum));
        int patches_to_assign = patches_per_proc >= 2 ? 2 : patches_per_proc;
        //int patches_to_assign = 1;
        if (patches_to_assign == 1) {
          allParticles[p].assigned = true;
          cout << " LB - assigning patch " <<  allParticles[p].id << " with " 
               << allParticles[p].numParticles << " to proc " << procnum << endl;
          d_processorAssignment[p] = procnum;
          procnum = (procnum+1 >= numProcs) ? 0 : procnum+1;
          patchesLeft--;
        }
        else {
          // find a neighbor where the sum of particles is closest to the avg_ppp
          // try to spread patches over all procs (particularly if we wrap)
          const Patch* patch = patchset[p]->getRealPatch();
          Patch::selectType n;
          IntVector lowIndex, highIndex;
          float target = avg_costPerProc *  patches_to_assign / (numPatches/numProcs);
          cout << " LB - target PPP: " << target << endl;
          float best = -100; // so that 0 will be better than this 
          int bestIndex = -1;
          int maxGhost = 2;
          patch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0),
                                        Ghost::AroundCells, maxGhost, n,
                                        lowIndex, highIndex);
          for(int i=0;i<(int)n.size();i++){
            const Patch* neighbor = n[i]->getRealPatch();
            if (allParticles[neighbor->getID()].assigned || neighbor == patch)
              continue;
            float myGuess = patch_costs[p] + patch_costs[neighbor->getID()];
            if (abs(myGuess-target) < abs(best-target)) {
              best = myGuess;
              bestIndex = neighbor->getID();
            }
          }
          cout << " LB - assigning patch " << allParticles[p].id << " with " 
               << allParticles[p].numParticles << " to proc " << procnum << endl;
          allParticles[p].assigned = true;
          d_processorAssignment[p] = procnum;
          patchesLeft--;          
          if (bestIndex != -1) { // had available neighbors
            cout << " LB - assigning patch " << allParticles[bestIndex].id << " with " 
                 << allParticles[bestIndex].numParticles << " to proc " << procnum << endl;
            allParticles[bestIndex].assigned = true;
            d_processorAssignment[bestIndex] = procnum;
            patchesLeft--;          
          }
          procnum = (procnum+1 >= numProcs) ? 0 : procnum+1;
        }
      }
    }
    else if (d_particleAlgo == 3) {
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

      for (int p = 0; p < numPatches; p++) {
        if (patch_costs[p] > .9 * avg_costPerProc) {
          d_processorAssignment[p] = currentProc;
          allParticles[p].assigned = true;
          totalCost -= patch_costs[p];
          cout << "Patch " << p << "-> proc " << currentProc << " Cost: " 
               << patch_costs[p] << endl;
          currentProc++;
        }
      }

      if (currentProc > 0) {
        int optimal_procs = totalCost/((origTotalCost-totalCost)/currentProc)
          + currentProc;
        cout << "This simulation would probably perform better on " 
             << optimal_procs << "-" << optimal_procs+1 << " processors\n";
        cout << "  or rather - origCost " << origTotalCost << ", costLeft " 
             << totalCost << " after " << currentProc << " procs " << endl;
      }
      avg_costPerProc = totalCost / (numProcs-currentProc);

      IntVector lastIndex = patchset[0]->getLowIndex();
      for (int p = 0; p < numPatches; p++, z++) {
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
        int index = x * (y_patches*z_patches) + 
          ((wrap_y) ? (y_patches-1-y)*z_patches : y*z_patches) +
          ((wrap_z) ? z_patches-1-z : z);
        
        if (allParticles[index].assigned)
          continue;

        // assign the patch to a processor.  When we advance procs,
        // re-update the cost, so we use all procs (and don't go over)
        float patchCost = patch_costs[index];
        if (currentProcCost > avg_costPerProc ||
            (currentProcCost + patchCost > avg_costPerProc *1.2&&
             currentProcCost >=  .7*avg_costPerProc)) {
          // move to next proc and add this patch
          currentProc++;
          d_processorAssignment[index] = currentProc;
          totalCost -= currentProcCost;
          avg_costPerProc = totalCost / (numProcs-currentProc);
          currentProcCost = patchCost;
          cout << "Patch " << index << "-> proc " << currentProc 
               << " PatchCost: " << patchCost << ", ProcCost: " 
               << currentProcCost 
               << ", idcheck: " << patchset[index]->getID() << endl;
        }
        else {
          // add patch to currentProc
          d_processorAssignment[index] = currentProc;
          currentProcCost += patchCost;
          cout << "Patch " << index << "-> proc " << currentProc 
               << " PatchCost: " << patchCost << ", ProcCost: " 
               << currentProcCost 
               << ", idcheck: " << patchset[index]->getID() << endl;
        }
      }
    }
  }
  MPI_Bcast(&d_processorAssignment[0], numPatches, MPI_INT,0,d_myworld->getComm());
  time = Time::currentSeconds() - time;
  cout << " Time to LB: " << time << endl;
}

void ParticleLoadBalancer::assignPatchesRandom(const GridP&, 
                                               const SchedulerP&)
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
  int numPatches = d_processorAssignment.size();

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
    d_processorAssignment[i] = newproc;
  }
}

void ParticleLoadBalancer::assignPatchesCyclic(const GridP&, 
                                               const SchedulerP&)
{
  // this assigns patches in a cyclic form - every time we re-load balance
  // we move each patch up one proc - this obviously isn't a very good
  // lb technique, but it tests its capabilities pretty well.

  int numProcs = d_myworld->size();
  for (unsigned i = 0; i < d_processorAssignment.size(); i++) {
    d_processorAssignment[i] = (d_processorAssignment[i] + 1 ) % numProcs;
  }
}


int
ParticleLoadBalancer::getPatchwiseProcessorAssignment(const Patch* patch)
{
  int proc = d_processorAssignment[patch->getRealPatch()->getID()];
  //cout << group->myrank() << " Requesting patch " << patch->getID()
  //   << " which is stored on processor " << proc << endl;
  //int proc = (patch->getLevelIndex()*numProcs)/patch->getLevel()->numPatches();
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
  int proc = d_oldAssignment[patch->getID()];
  //cout << d_myworld->myrank() << " Requesting patch " << patch->getID()
  //   << " which *used to be* stored on processor " << proc << endl;
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
  //return getPatchwiseProcessorAssignment(patch, d_myworld);
}

bool 
ParticleLoadBalancer::needRecompile(double /*time*/, double delt, 
				    const GridP& /*grid*/)
{
  if (d_dynamicAlgorithm == static_lb)
    return false;
  //need to compensate for restarts
  
  double time = d_sharedState->getElapsedTime();
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();

  //cout << d_myworld->myrank() << " PLB recompile: "
  // << time << d_lbTimestepInterval << ' ' << ' ' << d_lastLbTimestep
  //   << timestep << ' ' << d_lbInterval << ' ' << ' ' << d_lastLbTime << endl;

  if (d_lbTimestepInterval != 0 && 
      timestep >= d_lastLbTimestep + d_lbTimestepInterval) {
    d_lastLbTimestep = timestep;
    d_state = needLoadBalance;
  }
  else if (d_lbInterval != 0 && time >= d_lastLbTime + d_lbInterval) {
    d_lastLbTime = time;
    d_state = needLoadBalance;
  }

  //cout << d_myworld->myrank() << " PLB recompile: " << d_state << endl;
  return d_state != idle; // to recompile when need to load balance or post lb
} 

void
ParticleLoadBalancer::restartInitialize(ProblemSpecP& pspec, XMLURL tsurl)
{
  // here we need to grab the uda data to reassign patch data to the 
  // processor that will get the data
  for (unsigned i = 0; i < d_processorAssignment.size(); i++)
    d_processorAssignment[i]= -1;

  ASSERT(pspec != 0);
  ProblemSpecP datanode = pspec->findBlock("Data");
  if(datanode == 0)
    throw InternalError("Cannot find Data in timestep");
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
          throw InternalError("timestep href not found");
        
        XMLURL dataurl(tsurl, datafile.c_str());
        char* urltext = XMLString::transcode(dataurl.getURLText());

        // open the datafiles
        ProblemSpecReader psr(urltext);
        delete [] urltext;

        ProblemSpecP dataDoc = psr.readInputFile();
        if (!dataDoc)
          throw InternalError("Cannot open data file");
        for(ProblemSpecP r = dataDoc->getFirstChild(); r != 0; r=r->getNextSibling()){
          if(r->getNodeName() == "Variable") {
            int patchid;
            if(!r->get("patch", patchid) && !r->get("region", patchid))
              throw InternalError("Cannot get patch id");
            if (d_processorAssignment[patchid] == -1) {
              // assign the patch to the processor
              d_processorAssignment[patchid] = procnum % d_myworld->size();
            }
          }
        }            
        
      }
    }
  }
  for (unsigned i = 0; i < d_processorAssignment.size(); i++)
    ASSERT(d_processorAssignment[i] != -1);
}

void 
ParticleLoadBalancer::setDynamicAlgorithm(std::string algo, double interval,
                                          int timestepInterval, float factor)
{
  if (algo == "cyclic")
    d_dynamicAlgorithm = cyclic_lb;
  else if (algo == "random")
    d_dynamicAlgorithm = random_lb;
  else if (algo == "particle1") {
    d_dynamicAlgorithm = particle_lb;
    d_particleAlgo = 1;
  }
  else if (algo == "particle2") {
    d_dynamicAlgorithm = particle_lb;
    d_particleAlgo = 2;
  }
  else if (algo == "particle3") {
    d_dynamicAlgorithm = particle_lb;
    d_particleAlgo = 3;
  }
  else if (algo == "particle4") {
    d_dynamicAlgorithm = particle_lb;
    d_particleAlgo = 4;
  }
  else if (algo == "static")
    d_dynamicAlgorithm = static_lb;
  else {
    if (d_myworld->myrank() == 0)
      cout << "Invalid Load Balancer Algorithm: " << algo 
           << "\nPlease select 'cyclic', 'random', 'particle', or 'static' (default)\n"
           << "\nUsing 'static' load balancer\n";
    d_dynamicAlgorithm = particle_lb;
  }
  d_lbInterval = interval;
  d_lbTimestepInterval = timestepInterval;
  d_cellFactor = factor;
}


void ParticleLoadBalancer::dynamicReallocation(const GridP& grid, 
                                               const SchedulerP& sch)
{

  dbg << d_myworld->myrank() << " In PLB - State: " << d_state << endl;

  // if this is just during a recompile to compensate for 
  // loadbalancing on the last timestep
  if (d_state != needLoadBalance) {
    d_oldAssignment = d_processorAssignment;
    d_state = idle;
    return;
  }
  
  SchedulerCommon* sc = const_cast<SchedulerCommon*>(dynamic_cast<const SchedulerCommon*>(sch.get_rep()));

  int numProcs = d_myworld->size();
  int numPatches = 0;

  for(int l=0;l<grid->numLevels();l++){
    const LevelP& level = grid->getLevel(l);
    numPatches += level->numPatches();
  }

  int myrank = d_myworld->myrank();

  OnDemandDataWarehouse* dw = dynamic_cast<OnDemandDataWarehouse*>
    (sc->get_dw(0));


  if (dw == 0) {
    // on the first timestep, just assign the patches in a simple fashion
    // then on the next timestep do the real work
    d_processorAssignment.resize(numPatches);
    d_oldAssignment.resize(d_processorAssignment.size());

    for(int l=0;l<grid->numLevels();l++){
      const LevelP& level = grid->getLevel(l);

      for (Level::const_patchIterator iter = level->patchesBegin(); 
           iter != level->patchesEnd(); iter++) {
        Patch *patch = *iter;
        d_processorAssignment[patch->getID()] = patch->getID() % numProcs;
        ASSERTRANGE(patch->getID(),0,numPatches);
      }
    }

    d_oldAssignment = d_processorAssignment;
    d_state = needLoadBalance;  // l.b. on next timestep
  }
  else {
    d_oldAssignment = d_processorAssignment;
    switch (d_dynamicAlgorithm) {
    case particle_lb:  assignPatchesParticle(grid, sch); break;
    case cyclic_lb:    assignPatchesCyclic(grid, sch); break;
    case random_lb:    assignPatchesRandom(grid, sch); break;
    // static_lb does nothing
    }
    d_state = postLoadBalance;
  }
  
  if (dbg.active()) {
    for (int i = 0; i < numPatches; i++) {
      if (myrank == 0) 
        dbg << myrank << " patch " << i << " -> proc " << d_processorAssignment[i] << " (old " << d_oldAssignment[i] << ") - " << d_processorAssignment.size() << ' ' << d_oldAssignment.size() << "\n";
    }
  }
}
