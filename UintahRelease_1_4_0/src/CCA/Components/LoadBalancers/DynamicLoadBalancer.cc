/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/

//Allgatherv currently performs poorly on Kraken.  
//This hack changes the Allgatherv to an allgather 
//by padding the digits
//#define AG_HACK  

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/LoadBalancers/DynamicLoadBalancer.h>
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
static DebugStream doing("DynamicLoadBalancer_doing", false);
static DebugStream lb("DynamicLoadBalancer_lb", false);
static DebugStream dbg("DynamicLoadBalancer", false);
static DebugStream stats("LBStats",false);
static DebugStream times("LBTimes",false);
static DebugStream lbout("LBOut",false);
double lbtimes[5]={0};

//if defined the space-filling curve will be computed in parallel, this may not be a good idea because the time to compute 
//the space-filling curve is so small that it might not parallelize well.
//#define SFC_PARALLEL  

DynamicLoadBalancer::DynamicLoadBalancer(const ProcessorGroup* myworld)
  : LoadBalancerCommon(myworld), d_costForecaster(0), sfc(myworld)
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

#if defined( HAVE_ZOLTAN )
  float ver;
  Zoltan_Initialize( 0, NULL, &ver );
  zz = new Zoltan( d_myworld->getComm() );

  if( zz == NULL ){
    throw InternalError("Zoltan creation failed!", __FILE__, __LINE__);
  }
  //This parameter is to avoid using MPI_Comm_dup, MPI_Comm_split and MPI_Comm_free functions
  //These functions may not work well in some mpi implementations (mvapich 1.0 on TACC Ranger) 
  //and cause memory leek in Zoltan library. Maybe we can remove this line in the future.
  zz->Set_Param("TFLOPS_SPECIAL", "1"); 
#endif
}

DynamicLoadBalancer::~DynamicLoadBalancer()
{
  if(d_costForecaster)
  {
    delete d_costForecaster;
    d_costForecaster=0;
  }
#if defined( HAVE_ZOLTAN )
  delete zz;
#endif
}

void DynamicLoadBalancer::collectParticlesForRegrid(const Grid* oldGrid, const vector<vector<Region> >& newGridRegions, vector<vector<int> >& particles)
{
  // collect particles from the old grid's patches onto processor 0 and then distribute them
  // (it's either this or do 2 consecutive load balances).  For now, it's safe to assume that
  // if there is a new level or a new patch there are no particles there.

  int num_procs = d_myworld->size();
  int myrank = d_myworld->myrank();
  int num_patches = 0;

  particles.resize(newGridRegions.size());
  for (unsigned i = 0; i < newGridRegions.size(); i++)
  {
    particles[i].resize(newGridRegions[i].size());
    num_patches += newGridRegions[i].size();
  }
  
  if(!d_collectParticles)
    return;

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

void DynamicLoadBalancer::collectParticles(const Grid* grid, vector<vector<int> >& particles)
{
  particles.resize(grid->numLevels());
  for(int l=0;l<grid->numLevels();l++)
  {
    particles[l].resize(grid->getLevel(l)->numPatches());
    particles[l].assign(grid->getLevel(l)->numPatches(),0);
  }
  if (d_processorAssignment.size() == 0)
    return; // if we haven't been through the LB yet, don't try this.

  //if we are not supposed to collect particles just return
  if(!d_collectParticles || !d_scheduler->get_dw(0))
    return;

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

void DynamicLoadBalancer::useSFC(const LevelP& level, int* order)
{
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
  for(unsigned int i=0;i<indices.size();i++)
  {
    DistributedIndex di=indices[i];
    order[i]=originalPatchStart[di.p]+di.i;
  }
#else
  //write order array
  for(unsigned int i=0;i<indices.size();i++)
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

bool DynamicLoadBalancer::assignPatchesZoltanSFC(const GridP& grid, bool force)
{
  doing << d_myworld->myrank() << "   assignPatchesZoltanSFC\n";
  double time = Time::currentSeconds();

  // This will store a vector per level of costs for each patch:
  vector< vector<double> > patch_costs;

  int num_procs = d_myworld->size();

  getCosts(grid.get_rep(), patch_costs);

  int level_offset=0;
  
  int dim=d_sharedState->getNumDims();    //Number of dimensions
  int *dimensions=d_sharedState->getActiveDims(); //dimensions will store the active dimensions up to the number of dimensions

  for(int l=0;l<grid->numLevels();l++){

    const LevelP& level = grid->getLevel(l);
    int num_patches = level->numPatches();

    //create the positions vector
    vector<double> positions;
    vector<double> my_costs;
    vector<int> my_gids;
    for (Level::const_patchIterator iter = level->patchesBegin(); iter != level->patchesEnd(); iter++) 
    {
      const Patch* patch = *iter;

      //create positions vector
      //place in long longs to avoid overflows with large numbers of patches and processors
      long long pindex=patch->getLevelIndex();
      long long num_procs=d_myworld->size();
      long long proc = (pindex*num_procs)/(long long)level->numPatches();
    
      ASSERTRANGE(proc,0,d_myworld->size());
      
      if(d_myworld->myrank()==proc)
      {
        Vector point=(patch->getCellLowIndex()+patch->getCellHighIndex()).asVector()/2.0;
	my_costs.push_back(patch_costs[l][patch->getLevelIndex()]);
	my_gids.push_back(patch->getLevelIndex());
        for(int d=0;d<dim;d++)
        {
          positions.push_back(point[dimensions[d]]);
        }
      }
    }

    // costs[l][p] gives you the cost of patch p on level l
    // positions[p*3+d] gives you the location of patch p for the dimension d.
 //   std::cout << "rank " << d_myworld->myrank() << ": my_costs_size=" << my_costs.size() << ", costs_size=" << patch_costs[l].size() << "\n";

#if defined( HAVE_ZOLTAN )

    vector<void *> obj_data;
    obj_data.push_back((void *) &(my_costs));
    obj_data.push_back((void *) &(my_gids));
    
    /* General Zoltan parameters */
    zz->Set_Param("DEBUG_LEVEL", "0");     // amount of debug messages desired
    zz->Set_Param("LB_METHOD", d_zoltanAlgorithm.c_str());     // zoltan load balance algorithm
    zz->Set_Param("IMBALANCE_TOL", d_zoltanIMBTol.c_str());    // imbalance result toleration
    zz->Set_Param("NUM_GID_ENTRIES", "1"); // number of integers in a global ID
    zz->Set_Param("NUM_LID_ENTRIES", "1"); // number of integers in a local ID
    zz->Set_Param("OBJ_WEIGHT_DIM", "1");  // dimension of a vertex weight
    zz->Set_Param("RETURN_LISTS", "ALL");  // return all lists in LB_Partition
    
    /* Balance Method parameters */
    zz->Set_Param("KEEP_CUTS", "0");

    /* Set Zoltan Functions */
    zz->Set_Num_Obj_Fn(ZoltanFuncs::zoltan_get_number_of_objects, & (my_costs) );
    zz->Set_Obj_List_Fn(ZoltanFuncs::zoltan_get_object_list, &(obj_data));
    zz->Set_Num_Geom_Fn(ZoltanFuncs::zoltan_get_number_of_geometry, &dim);
    zz->Set_Geom_Multi_Fn(ZoltanFuncs::zoltan_get_geometry_list, &(positions));


    /* Perform Partition */
    int changes;
    int numGidEntries;
    int numLidEntries;
    int numImport;
    ZOLTAN_ID_PTR importGlobalIds;
    ZOLTAN_ID_PTR importLocalIds;
    int *importProcs;
    int *importToPart;
    int numExport;
    ZOLTAN_ID_PTR exportGlobalIds;
    ZOLTAN_ID_PTR exportLocalIds;
    int *exportProcs;
    int *exportToPart;

    int rc = zz->LB_Partition(changes, numGidEntries, numLidEntries,
      numImport, importGlobalIds, importLocalIds, importProcs, importToPart,
      numExport, exportGlobalIds, exportLocalIds, exportProcs, exportToPart);

    if (rc != ZOLTAN_OK){
      throw InternalError("Zoltan partition failed!", __FILE__, __LINE__);
      return false;
    }

    //set assignment result array 
    int nMyGids = my_gids.size();
    int nGids   = num_patches;
    int *gid_list = new int[nMyGids];
    int *lid_list = new int[nMyGids];
    
    ZoltanFuncs::zoltan_get_object_list(&obj_data, nMyGids, nMyGids,
      (ZOLTAN_ID_PTR)gid_list, (ZOLTAN_ID_PTR)lid_list, 0, NULL, &rc);
    
    int *gid_flags = new int[nGids];
    int *gid_results = new int[nGids];
    memset(gid_flags, 0, sizeof(int) * nGids);
    for (int i=0; i<nMyGids; i++){
      gid_flags[gid_list[i]] = d_myworld->myrank();    // my original vertices
    }
    for (int i=0; i<numImport; i++){
       gid_flags[importGlobalIds[i]] = d_myworld->myrank();  // my imports
    }
    for (int i=0; i<numExport; i++){
       gid_flags[exportGlobalIds[i]] = 0;  // my exports
    }
    

    MPI_Allreduce(gid_flags, gid_results, nGids, MPI_INT, MPI_SUM, d_myworld->getComm());

    // Code should set d_tempAssignment[level_offset+p] to be equal to the owner of the patch p...
    for (int i=0; i<nGids; i++){
       d_tempAssignment[level_offset+i] = gid_results[i];
      // std::cout << "Proc "<< d_myworld->myrank() << ": Gid " << i << " assigned to proc " << gid_results[i] << "\n";
    }

    delete [] gid_results;
    delete [] gid_flags;
    delete [] gid_list;
    delete [] lid_list;
  
    //mpi_communicator is d_myworld->getComm()
    //mpi_rank is d_myworld->myrank()
    //mpi_size is d_myworld->size()
#endif

    if(stats.active() && d_myworld->myrank()==0)
    {
      //calculate lb stats:
      double totalCost=0;
      vector<double> procCosts(num_procs,0);
      for(int p=0;p<num_patches;p++)
      {
        totalCost+=patch_costs[l][p];
        procCosts[d_tempAssignment[level_offset+p]]+=patch_costs[l][p];
      }

      double meanCost=totalCost/num_procs;
      double minCost=procCosts[0];
      double maxCost=procCosts[0];
      //if(d_myworld->myrank()==0)
      //  stats << "Level:" << l << " ProcCosts:";

      for(int p=0;p<num_procs;p++)
      {
        if(minCost>procCosts[p])
          minCost=procCosts[p];
        else if(maxCost<procCosts[p])
          maxCost=procCosts[p];

       // if(d_myworld->myrank()==0)
       //   stats << p << ":" << procCosts[p] << " ";
      }
      //if(d_myworld->myrank()==0)
      //  stats << endl;

      stats << "LoadBalance Stats level(" << l << "):"  << " Mean:" << meanCost << " Min:" << minCost << " Max:" << maxCost << " Imb:" << 1-meanCost/maxCost <<  endl;
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
  

bool DynamicLoadBalancer::assignPatchesFactor(const GridP& grid, bool force)
{
  doing << d_myworld->myrank() << "   APF\n";
  vector<vector<double> > patch_costs;
  double time = Time::currentSeconds();
  for(int i=0;i<5;i++)
    lbtimes[i]=0;
      

  double start=Time::currentSeconds();
  
  lbtimes[0]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();

  static int lbiter=-1; //counter to identify which regrid
  lbiter++;

  int num_procs = d_myworld->size();

  getCosts(grid.get_rep(),patch_costs);

  lbtimes[1]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();

  int level_offset=0;

  vector<double> totalProcCosts(num_procs,0);
  vector<double> procCosts(num_procs,0);
  vector<double> previousProcCosts(num_procs,0);
  double previous_total_cost=0;
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
    
    lbtimes[2]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();

    //hard maximum cost for assigning a patch to a processor
    double avgCost = (total_cost+previous_total_cost) / num_procs;
    double hardMaxCost = total_cost;    
    double myMaxCost =hardMaxCost-(hardMaxCost-avgCost)/d_myworld->size()*(double)d_myworld->myrank();
    double myStoredMax=DBL_MAX;
    int minProcLoc = -1;

    if(!force)  //use initial load balance to start the iteration
    {
      //compute costs of current load balance
      vector<double> currentProcCosts(num_procs);

      for(int p=0;p<(int)patch_costs[l].size();p++)
      {
        //copy assignment from current load balance
        d_tempAssignment[level_offset+p]=d_processorAssignment[level_offset+p];
        //add assignment to current costs
        currentProcCosts[d_processorAssignment[level_offset+p]] += patch_costs[l][p];
      }

      //compute maximum of current load balance
      hardMaxCost=currentProcCosts[0];
      for(int i=1;i<num_procs;i++)
      {
        if(currentProcCosts[i]+previousProcCosts[i]>hardMaxCost)
          hardMaxCost=currentProcCosts[i]+previousProcCosts[i];
      }
      double range=hardMaxCost-avgCost;
      myStoredMax=hardMaxCost;
      myMaxCost=hardMaxCost-range/d_myworld->size()*(double)d_myworld->myrank();
    }

    //temperary vector to assign the load balance in
    vector<int> temp_assignment(d_tempAssignment);
    vector<int> maxList(num_procs);

    int iter=0;
    double improvement=1;
    //iterate the load balancing algorithm until the max can no longer be lowered
    while(improvement>0)
    {

      double remainingCost=total_cost+previous_total_cost;
      double avgCostPerProc = remainingCost / num_procs;

      int currentProc = 0;
      vector<double> currentProcCosts(num_procs,0);
      double currentMaxCost = 0;

      for (int p = 0; p < num_patches; p++) {
        int index;
        if (d_doSpaceCurve) {
          index = order[p];
        }
        else {
          // not attempting space-filling curve
          index = p;
        }

        // assign the patch to a processor.  When we advance procs,
        // re-update the cost, so we use all procs (and don't go over)
        double patchCost = patch_costs[l][index];
        double notakeimb=fabs(previousProcCosts[currentProc]+currentProcCosts[currentProc]-avgCostPerProc);
        double takeimb=fabs(previousProcCosts[currentProc]+currentProcCosts[currentProc]+patchCost-avgCostPerProc);

        if ( previousProcCosts[currentProc]+currentProcCosts[currentProc]+patchCost<myMaxCost && takeimb<=notakeimb) {
          // add patch to currentProc
          temp_assignment[level_offset+index] = currentProc;
          currentProcCosts[currentProc] += patchCost;
        }
        else {
          if(previousProcCosts[currentProc]+currentProcCosts[currentProc]>currentMaxCost)
          {
            currentMaxCost=previousProcCosts[currentProc]+currentProcCosts[currentProc];
          }


          //subtract currentProc's cost from remaining cost
          remainingCost -= (currentProcCosts[currentProc]+previousProcCosts[currentProc]);

          // move to next proc 
          currentProc++;

          //if currentProc to large then load balance is invalid so break out
          if(currentProc>=num_procs)
            break;

          //assign patch to currentProc
          temp_assignment[level_offset+index] = currentProc;

          //update average (this ensures we don't over/under fill to much)
          avgCostPerProc = remainingCost / (num_procs-currentProc);
          currentProcCosts[currentProc] = patchCost;

        }
      }

      //check if last proc is the max
      if(currentProc<num_procs && previousProcCosts[currentProc]+currentProcCosts[currentProc]>currentMaxCost)
        currentMaxCost=previousProcCosts[currentProc]+currentProcCosts[currentProc];

      //if the max was lowered and the assignments are valid
      if(currentMaxCost<myStoredMax && currentProc<num_procs)
      {

#if 1
        //take this assignment
        for(int p=0;p<num_patches;p++)
        {
          d_tempAssignment[level_offset+p]=temp_assignment[level_offset+p];
        }
#else
        d_tempAssignment.swap(temp_assignment);
#endif
        //update myMaxCost
        myStoredMax=currentMaxCost;
      }

      double_int maxInfo(myStoredMax,d_myworld->myrank());
      double_int min;

      //gather the maxes
      //change to all reduce with loc
      if(num_procs>1)
        MPI_Allreduce(&maxInfo,&min,1,MPI_DOUBLE_INT,MPI_MINLOC,d_myworld->getComm());    
      else
        min=maxInfo;

      //set improvement
      improvement=hardMaxCost-min.val;

      if(min.val<hardMaxCost)
      {
        //set hardMax
        hardMaxCost=min.val;
        //set minloc
        minProcLoc=min.loc;
      }

      //compute average cost per proc
      double average=(total_cost+previous_total_cost)/num_procs;
      //set new myMax by having each processor search at even intervals in the range
      double range=hardMaxCost-average;
      myMaxCost=hardMaxCost-range/d_myworld->size()*(double)d_myworld->myrank();
      iter++;
    }

    lbtimes[3]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();

    if(minProcLoc!=-1 && num_procs>1)
    {
      //broadcast load balance
      MPI_Bcast(&d_tempAssignment[0],d_tempAssignment.size(),MPI_INT,minProcLoc,d_myworld->getComm());
    }

    if(!d_levelIndependent)
    {
      //update previousProcCost

      //loop through the assignments for this level and add costs to previousProcCosts
      for(int p=0;p<num_patches;p++)
      {
        previousProcCosts[d_tempAssignment[level_offset+p]]+=patch_costs[l][p];
      }
      previous_total_cost+=total_cost;
    }

    if(stats.active() && d_myworld->myrank()==0)
    {
      //calculate lb stats:
      double totalCost=0;
      vector<double> procCosts(num_procs,0);
      vector<int> patchCounts(num_procs,0);
      for(int p=0;p<num_patches;p++)
      {
        totalCost+=patch_costs[l][p];
        procCosts[d_tempAssignment[level_offset+p]]+=patch_costs[l][p];
        totalProcCosts[d_tempAssignment[level_offset+p]]+=patch_costs[l][p];
        patchCounts[d_tempAssignment[level_offset+p]]++;
      }

      double meanCost=totalCost/num_procs;
      double minCost=procCosts[0];
      double maxCost=procCosts[0];
      int maxProc=0;

      //if(d_myworld->myrank()==0)
      //  stats << "Level:" << l << " ProcCosts:";

      for(int p=0;p<num_procs;p++)
      {
        if(minCost>procCosts[p])
          minCost=procCosts[p];
        else if(maxCost<procCosts[p])
        {
          maxCost=procCosts[p];
          maxProc=p;
        }
       // if(d_myworld->myrank()==0)
       //   stats << p << ":" << procCosts[p] << " ";
      }
      //if(d_myworld->myrank()==0)
      //  stats << endl;
      stats << "LoadBalance Stats level(" << l << "):"  << " Mean:" << meanCost << " Min:" << minCost << " Max:" << maxCost << " Imb:" << 1-meanCost/maxCost << " max on:" << maxProc << endl;
    }  

    if(lbout.active() && d_myworld->myrank()==0)
    {
      for(int p=0;p<num_patches;p++)
      {
        int index; //compute order index
        if (d_doSpaceCurve) {
          index = order[p];
        }
        else {
          // not attempting space-filling curve
          index = p;
        }

        IntVector sum=(level->getPatch(index)->getCellLowIndex()+level->getPatch(index)->getCellHighIndex());

        Vector loc(sum.x()/2.0,sum.y()/2.0,sum.z()/2.0);
        //output load balance information
        lbout << lbiter << " " << l << " " << index << " " << d_tempAssignment[level_offset+index] << " " <<  patch_costs[l][index] << " " << loc.x() << " " << loc.y() << " " << loc.z() << endl;
      }
    }
    
    level_offset+=num_patches;
    lbtimes[4]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();
  }

  if(stats.active() && d_myworld->myrank()==0)
  {
      double meanCost=0;
      double minCost=totalProcCosts[0];
      double maxCost=totalProcCosts[0];
      int maxProc=0;

      for(int p=0;p<num_procs;p++)
      {
        meanCost+=totalProcCosts[p];
        
        if(minCost>totalProcCosts[p])
          minCost=totalProcCosts[p];
        else if(maxCost<totalProcCosts[p])
        {
          maxCost=totalProcCosts[p];
          maxProc=p;
        }
      }
      meanCost/=num_procs;

      stats << "LoadBalance Stats total:"  << " Mean:" << meanCost << " Min:" << minCost << " Max:" << maxCost << " Imb:" << 1-meanCost/maxCost << " max on:" << maxProc << endl;

  }
  if(times.active())
  {
    double avg[5]={0};
    MPI_Reduce(&lbtimes,&avg,5,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    if(d_myworld->myrank()==0) {
      cout << "LoadBalance Avg Times: "; 
      for(int i=0;i<5;i++)
      {
        avg[i]/=d_myworld->size();
        cout << avg[i] << " ";
      }
      cout << endl;
    }
    double max[5]={0};
    MPI_Reduce(&lbtimes,&max,5,MPI_DOUBLE,MPI_MAX,0,d_myworld->getComm());
    if(d_myworld->myrank()==0) {
      cout << "LoadBalance Max Times: "; 
      for(int i=0;i<5;i++)
      {
        cout << max[i] << " ";
      }
      cout << endl;
    }
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
  int num_levels = patch_costs.size();
  
  vector<vector<double> > currentProcCosts(num_levels);
  vector<vector<double> > tempProcCosts(num_levels);

  int i=0;
  for(int l=0;l<num_levels;l++)
  {
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
  
  if(d_levelIndependent)
  {
    double avg_current = 0;
    double max_current = 0;
    double avg_temp = 0;
    double max_temp = 0;
    for(int l=0;l<num_levels;l++)
    {
      avg_current = 0;
      max_current = 0;
      avg_temp = 0;
      max_temp = 0;
      for (int i = 0; i < d_myworld->size(); i++) 
      {
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
  else
  {
      for(int i=0;i<d_myworld->size();i++)
      {
        double current_cost=0, temp_cost=0;
        for(int l=0;l<num_levels;l++)
        {
          current_cost+=currentProcCosts[l][i];
          temp_cost+=currentProcCosts[l][i];
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
  }
    
  
  if (d_myworld->myrank() == 0)
  {
    stats << "Total:"  << " maxCur:" << total_max_current << " maxTemp:"  << total_max_temp << " avgCur:" << total_avg_current << " avgTemp:" << total_avg_temp <<endl;
  }

  // if tmp - cur is positive, it is an improvement
  if( (total_max_current-total_max_temp)/total_max_current>d_lbThreshold)
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
  // if on a copy-data timestep and we ask about an old patch, that could cause problems
  if (d_sharedState->isCopyDataTimestep() && patch->getRealPatch()->getID() < d_assignmentBasePatch)
    return -patch->getID();
 
  ASSERTRANGE(patch->getRealPatch()->getID(), d_assignmentBasePatch, d_assignmentBasePatch + (int) d_processorAssignment.size());
  int proc = d_processorAssignment[patch->getRealPatch()->getGridIndex()];

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
  if ((int)patch->getRealPatch()->getID() < d_oldAssignmentBasePatch || patch->getRealPatch()->getID() >= d_oldAssignmentBasePatch + (int)d_oldAssignment.size())
    return -9999;
  
  if (patch->getGridIndex() >= (int) d_oldAssignment.size())
    return -999;

  int proc = d_oldAssignment[patch->getRealPatch()->getGridIndex()];
  ASSERTRANGE(proc, 0, d_myworld->size());
  return proc;
}

bool 
DynamicLoadBalancer::needRecompile(double /*time*/, double /*delt*/, 
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
  else if ((time == 0 && d_collectParticles == true) || d_checkAfterRestart) {
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
DynamicLoadBalancer::restartInitialize( DataArchive* archive, int time_index, ProblemSpecP& pspec,
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
void DynamicLoadBalancer::getCosts(const Grid* grid, vector<vector<double> >&costs)
{
  costs.clear();
    
  vector<vector<int> > num_particles;

  DataWarehouse* olddw = d_scheduler->get_dw(0);
  bool on_regrid = olddw != 0 && grid != olddw->getGrid();

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

  //check if the forecaster is ready, if it is use it
  if(d_costForecaster->hasData())
  {
    //we have data so don't collect particles
    d_costForecaster->getWeights(grid,num_particles,costs);
  }
  else //otherwise just use a simple cost model (this happens on the first timestep when profiling data doesn't exist)
  {
    CostModeler(d_patchCost,d_cellCost,d_extraCellCost,d_particleCost).getWeights(grid,num_particles,costs);
  }

  if (dbg.active() && d_myworld->myrank() == 0) {
    for (unsigned l = 0; l < costs.size(); l++)
      for (unsigned p = 0; p < costs[l].size(); p++)
        dbg << "L"  << l << " P " << p << " cost " << costs[l][p] << endl;
  }
}

bool DynamicLoadBalancer::possiblyDynamicallyReallocate(const GridP& grid, int state)
{
  MALLOC_TRACE_TAG_SCOPE("DynamicLoadBalancer::possiblyDynamicallyReallocate");
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
      switch (d_dynamicAlgorithm) {
        case patch_factor_lb:  dynamicAllocate = assignPatchesFactor(grid, force); break;
        case cyclic_lb:        dynamicAllocate = assignPatchesCyclic(grid, force); break;
        case random_lb:        dynamicAllocate = assignPatchesRandom(grid, force); break;
        case zoltan_sfc_lb:    dynamicAllocate = assignPatchesZoltanSFC(grid, force); break;
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
        // int num_procs = (int)d_myworld->size();
        int myrank = d_myworld->myrank();
        if (myrank == 0) {
          LevelP curLevel = grid->getLevel(0);
          Level::const_patchIterator iter = curLevel->patchesBegin();
          lb << "  Changing the Load Balance\n";
          for (unsigned int i = 0; i < d_processorAssignment.size(); i++) {
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

void DynamicLoadBalancer::finalizeContributions(const GridP grid) 
{
    d_costForecaster->finalizeContributions(grid);
}


void
DynamicLoadBalancer::problemSetup(ProblemSpecP& pspec, GridP& grid,  SimulationStateP& state)
{
  LoadBalancerCommon::problemSetup(pspec, grid, state);

  ProblemSpecP p = pspec->findBlock("LoadBalancer");
  string dynamicAlgo;
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
    p->getWithDefault("dynamicAlgorithm", dynamicAlgo, "patchFactor");
    p->getWithDefault("cellCost", d_cellCost, 1);
    p->getWithDefault("extraCellCost", d_extraCellCost, 1);
    p->getWithDefault("particleCost", d_particleCost, 1.25);
    p->getWithDefault("patchCost", d_patchCost, 16);
    p->getWithDefault("gainThreshold", threshold, 0.05);
    p->getWithDefault("doSpaceCurve", spaceCurve, true);

    p->getWithDefault("hasParticles", d_collectParticles, false);
    
    string costAlgo="ModelLS";
    p->get("costAlgorithm",costAlgo);
    if(costAlgo=="ModelLS")
    {
      d_costForecaster= scinew CostModelForecaster(d_myworld,this,d_patchCost,d_cellCost,d_extraCellCost,d_particleCost);
    }
    else if(costAlgo=="Kalman")
    {
      int timestepWindow;
      p->getWithDefault("profileTimestepWindow",timestepWindow,10);
      d_costForecaster=scinew CostProfiler(d_myworld,ProfileDriver::KALMAN,this);
      d_costForecaster->setTimestepWindow(timestepWindow);
      d_collectParticles=false;
    }
    else if(costAlgo=="Memory")
    {
      int timestepWindow;
      p->getWithDefault("profileTimestepWindow",timestepWindow,10);
      d_costForecaster=scinew CostProfiler(d_myworld,ProfileDriver::MEMORY,this);
      d_costForecaster->setTimestepWindow(timestepWindow);
      d_collectParticles=false;
    }
    else if(costAlgo=="Model")
    {
      d_costForecaster=scinew CostModeler(d_patchCost,d_cellCost,d_extraCellCost,d_particleCost);
    }
    else
    {
      throw InternalError("Invalid CostAlgorithm in Dynamic Load Balancer\n",__FILE__,__LINE__);
    }
   
    p->getWithDefault("levelIndependent",d_levelIndependent,true);
  }


  if(d_myworld->myrank()==0)
    cout << "Dynamic Algorithm: " << dynamicAlgo << endl;

  if (dynamicAlgo == "cyclic")
    d_dynamicAlgorithm = cyclic_lb;
  else if (dynamicAlgo == "random")
    d_dynamicAlgorithm = random_lb;
  else if (dynamicAlgo == "patchFactor") {
    d_dynamicAlgorithm = patch_factor_lb;
  }
  else if (dynamicAlgo == "patchFactorParticles" || dynamicAlgo == "particle3") {
    // these are for backward-compatibility
    d_dynamicAlgorithm = patch_factor_lb;
    d_collectParticles = true;
  }
#if defined( HAVE_ZOLTAN )
  else if (dynamicAlgo == "Zoltan")
  {
    d_dynamicAlgorithm=zoltan_sfc_lb;
    p->getWithDefault("zoltanAlgorithm",d_zoltanAlgorithm,"HSFC");
    p->getWithDefault("zoltanIMBTol",d_zoltanIMBTol,"1.1");
  }
#endif
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

  //set costProfiler mps
  Regridder *regridder = dynamic_cast<Regridder*>(getPort("regridder"));
  if(regridder)
  {
    d_costForecaster->setMinPatchSize(regridder->getMinPatchSize());
  }
  else
  {
    //query mps from a patch
    const Patch *patch=grid->getLevel(0)->getPatch(0);

    vector<IntVector> mps;
    mps.push_back(patch->getCellHighIndex()-patch->getCellLowIndex());

    d_costForecaster->setMinPatchSize(mps);
  }
}

#if defined( HAVE_ZOLTAN )

void
ZoltanFuncs::zoltan_get_object_list( void *data, int sizeGID, int sizeLID,
                                     ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                                     int wgt_dim, float *obj_wgts, int *ierr )
{
  vector<void *> * obj_data = (vector<void *> *) data;
  vector<double> * obj_costs =  (vector<double> *) ((*obj_data)[0]);
  vector<int> * obj_gids =  (vector<int> *)((*obj_data)[1]);
  for (unsigned int i=0; i < obj_costs->size(); i++) {
    globalID[i] = (*obj_gids)[i];
    localID[i] = i;
    if (wgt_dim) obj_wgts[i] = (float) (*obj_costs)[i];
  }
  *ierr = ZOLTAN_OK;
}

int
ZoltanFuncs::zoltan_get_number_of_objects( void *data, int *ierr )
{
    vector<double> * obj_costs =  (vector<double> *)data;
    *ierr = ZOLTAN_OK;
    return obj_costs->size();
}

int
ZoltanFuncs::zoltan_get_number_of_geometry( void *data, int *ierr )
{
    *ierr = ZOLTAN_OK;
    return *static_cast<int*>(data);
}

void
ZoltanFuncs::zoltan_get_geometry_list( void *data, int sizeGID, int sizeLID, int num_obj, 
                                       ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID, 
                                       int num_dim, double *geom_vec, int *ierr )
{
  vector<double> * obj_pos =  (vector<double> *)data;
  for( int i=0; i < (num_obj * num_dim); i++ ) {
    geom_vec[i] = (*obj_pos)[i];
  }
  *ierr = ZOLTAN_OK;
}

#endif
