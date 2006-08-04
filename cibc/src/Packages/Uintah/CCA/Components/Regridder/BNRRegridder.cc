#include <Packages/Uintah/CCA/Components/Regridder/BNRRegridder.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/PatchRangeTree.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
using namespace Uintah;

#include <vector>
#include <set>
#include <algorithm>
#include <iomanip>
using namespace std;

static DebugStream dbgpatches("BNRPatches",false);
static DebugStream dbgstats("BNRStats",false);
static DebugStream dbgtimes("BNRTimes",false);

BNRRegridder::BNRRegridder(const ProcessorGroup* pg) : RegridderCommon(pg), task_count_(0),tola_(1),tolb_(1), patchfixer_(pg)
{
  int numprocs=d_myworld->size();
  int rank=d_myworld->myrank();
  
  int *tag_ub, maxtag_ ,flag;
  int tag_start, tag_end;

  //generate tag lists for processors
  if(numprocs>1)
  {  
    MPI_Attr_get(d_myworld->getComm(),MPI_TAG_UB,&tag_ub,&flag);
    if(flag)
      maxtag_=(*tag_ub>>1);
    else
      maxtag_=(32767>>1);

    maxtag_++;
    int div=maxtag_/numprocs;
    int rem=maxtag_%numprocs;
  
    tag_start=div*rank;
  
    if(rank<rem)
      tag_start+=rank;
    else
      tag_start+=rem;
  
    if(rank<rem)
      tag_end=tag_start+div+1;
    else
      tag_end=tag_start+div;

    //don't have zero in the tag list  
    if(rank==0)
      tag_start++;

    for(int i=tag_start;i<tag_end;i++)
      tags_.push(i<<1);
  }
  
  if(dbgpatches.active() && rank==0)
  {
     fout.open("patches.bin");
  }
}

BNRRegridder::~BNRRegridder()
{
  if(dbgpatches.active() && d_myworld->myrank()==0)
  {
    fout.close();
  }
}

Grid* BNRRegridder::regrid(Grid* oldGrid, SchedulerP& sched, const ProblemSpecP& ups)
{
  double start;
  double brtotal=0,futotal=0,sltotal=0,pfutotal=0,crtotal=0, ctotal=0, ftotal=0, atotal=0;
  LoadBalancer *lb=sched->getLoadBalancer();
  
  Grid* newGrid = scinew Grid();
  ProblemSpecP grid_ps = ups->findBlock("Grid");

  vector<set<IntVector> > coarse_flag_sets(oldGrid->numLevels());
  vector<vector <IntVector> > flag_sets;

  if(dbgpatches.active())
  {
    flag_sets.resize(oldGrid->numLevels());
  }

  vector< vector<PseudoPatch> > patch_sets(oldGrid->numLevels());
  //create flags sets
  
  //For each level Fine to Coarse
  for(int l=oldGrid->numLevels()-1; l >= 0;l--)
  {
    if(l>=d_maxLevels-1)
      continue;

    start=MPI_Wtime();
    const LevelP level=oldGrid->getLevel(l);
   
    //create coarse flag set 
    const PatchSubset *ps=lb->getPerProcessorPatchSet(level)->getSubset(d_myworld->myrank());
    for(int p=0;p<ps->size();p++)
    {
      const Patch *patch=ps->get(p);
      DataWarehouse *dw=sched->getLastDW();
      constCCVariable<int> flags;
      dw->get(flags, d_dilatedCellsCreationLabel, 0, patch, Ghost::None, 0);
      for (CellIterator ci(patch->getInteriorCellLowIndex(), patch->getInteriorCellHighIndex()); !ci.done(); ci++)
      {
        if (flags[*ci])
        {
         coarse_flag_sets[l].insert(*ci*d_cellRefinementRatio[l]/d_minPatchSize);
         if(dbgpatches.active())
         {
            flag_sets[l].push_back(*ci);
         }
        }
      }
    }
    //create flags vector
    vector<IntVector> coarse_flag_vector(coarse_flag_sets[l].size());
    coarse_flag_vector.assign(coarse_flag_sets[l].begin(),coarse_flag_sets[l].end());
    ctotal+=MPI_Wtime()-start;
    start=MPI_Wtime(); 
    //Parallel BR over coarse flags
    RunBR(coarse_flag_vector,patch_sets[l]);  
    brtotal+=MPI_Wtime()-start;

    if(patch_sets[l].empty()) //no patches goto next level
       continue;

    //add saftey layers for finer level
    if(l>0)
    {
      start=MPI_Wtime();
      AddSafetyLayer(patch_sets[l], coarse_flag_sets[l-1], lb->getPerProcessorPatchSet(oldGrid->getLevel(l))->getSubset(d_myworld->myrank())->getVector(), l);
      sltotal+=MPI_Wtime()-start;
    }
    //Fixup patchlist:  this forces neighbor constraints
    start=MPI_Wtime();
    patchfixer_.FixUp(patch_sets[l]);
    futotal+=MPI_Wtime()-start;
    //Post fixup patchlist: this creates more patches as specified in the input file
    start=MPI_Wtime();
    PostFixup(patch_sets[l],d_minPatchSize);
    pfutotal+=MPI_Wtime()-start;
  }

  //Create the grid
  start=MPI_Wtime(); 
  // create level 0
  Point anchor = oldGrid->getLevel(0)->getAnchor();
  Vector spacing = oldGrid->getLevel(0)->dCell();
  IntVector extraCells = oldGrid->getLevel(0)->getExtraCells();
  IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
  LevelP level0 = newGrid->addLevel(anchor, spacing);
  level0->setExtraCells(extraCells);
  for (Level::const_patchIterator iter = oldGrid->getLevel(0)->patchesBegin(); iter != oldGrid->getLevel(0)->patchesEnd(); iter++)
  {
    const Patch* p = *iter;
    IntVector inlow = p->getInteriorCellLowIndex();
    IntVector low = p->getCellLowIndex();
    IntVector inhigh = p->getInteriorCellHighIndex();
    IntVector high = p->getCellHighIndex();
    level0->addPatch(low, high, inlow, inhigh);
  }
  crtotal+=MPI_Wtime()-start;
  start=MPI_Wtime();
  level0->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
  ftotal+=MPI_Wtime()-start;
  start=MPI_Wtime();
  level0->assignBCS(grid_ps);
  atotal+=MPI_Wtime()-start;
  start=MPI_Wtime();

  //For each level Coarse -> Fine
  for(int l=0; l < oldGrid->numLevels() && l < d_maxLevels-1;l++)
  {
    // if there are no patches on this level, don't create any more levels
    if (patch_sets[l].size() == 0)
      break;

    // parameters based on next-fine level.
    spacing = spacing / d_cellRefinementRatio[l];
    
    LevelP newLevel = newGrid->addLevel(anchor, spacing);
    newLevel->setExtraCells(extraCells);

    //for each patch
    for(unsigned int p=0;p<patch_sets[l].size();p++)
    {
      //uncoarsen
      IntVector low = patch_sets[l][p].low = patch_sets[l][p].low*d_minPatchSize;
      IntVector high = patch_sets[l][p].high = patch_sets[l][p].high*d_minPatchSize;
      //create patch
      newLevel->addPatch(low, high, low, high);
    }
    
    crtotal+=MPI_Wtime()-start;
    start=MPI_Wtime();
    newLevel->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    ftotal+=MPI_Wtime()-start;
    start=MPI_Wtime();
    newLevel->assignBCS(grid_ps);
    atotal+=MPI_Wtime()-start;
    start=MPI_Wtime();
  }
  crtotal+=MPI_Wtime()-start;
  if(dbgtimes.active() && d_myworld->myrank()==0)
          cout << "BRTime:" << brtotal << " SLTime:" << sltotal << " FUTime:" << futotal << " PFUTime:" << pfutotal << " CoarsenTime: " << ctotal << " CRTime:" << crtotal << " FTime: " << ftotal << " ATime: " << atotal << endl; 
  if (*newGrid == *oldGrid) {
    delete newGrid;
    return oldGrid;
  }

  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();

  if (dbgpatches.active())
  {
     if(d_myworld->myrank()==0)
     { 
         //gather flags
        //for each processors
        for(int p=1;p<d_myworld->size();p++)
        {
           //for each level
           for(unsigned int l=0;l<flag_sets.size();l++)
           {
              int numFlags;
              MPI_Status status;
              //recieve the number of flags they have
              MPI_Recv(&numFlags,1,MPI_INT,p,0,MPI_COMM_WORLD,&status);
              int size=flag_sets[l].size();
              //resize vector
              flag_sets[l].resize(size+numFlags);
              //recieve the flags
              MPI_Recv(&flag_sets[l][size],sizeof(IntVector)*numFlags,MPI_BYTE,p,1,MPI_COMM_WORLD,&status);
           }
         }
        //cout << *newGrid;
        writeGrid(newGrid,flag_sets);
     }
     else
     {
      
        //for each level
        for(unsigned int l=0;l<flag_sets.size();l++)
        {
          int size=flag_sets[l].size();

          //send the number of flags i have
          MPI_Send(&size,1,MPI_INT,0,0,MPI_COMM_WORLD);
          //send the flags
          MPI_Send(&flag_sets[l][0],sizeof(IntVector)*size,MPI_BYTE,0,1,MPI_COMM_WORLD);
        }
     }
  }
  if (dbgstats.active() && d_myworld->myrank() == 0) 
  {
    dbgstats << " Grid Statistics:\n";
    for (int l = 0; l < newGrid->numLevels()-1; l++) 
    {
      int total_vol=0;
      int sum_of_vol_squared=0;
      int n = patch_sets[l].size();
      //calculate total volume and volume squared
      int vol_mult=d_minPatchSize[0]*d_minPatchSize[1]*d_minPatchSize[2];
      for(int p=0;p<n;p++)
      {
        int vol=patch_sets[l][p].volume*vol_mult;
        total_vol+=vol;
        sum_of_vol_squared+=vol*vol;
      }
      //calculate mean
      double mean = total_vol /(double) n;
      double stdv = sqrt((sum_of_vol_squared-total_vol*total_vol/(double)n)/(double)n);
      
      dbgstats << left << "  L" << setw(8) << l+1 << ": Patches: " << setw(8) << n << " Volume: " << setw(8) << total_vol<< " Mean Volume: " << setw(8) << mean << " stdv: " << setw(8) << stdv << " relative stdv: " << setw(8) << stdv/mean << endl;
    }
  }

  newGrid->performConsistencyCheck();
  return newGrid;
}

void BNRRegridder::RunBR( vector<IntVector> &flags, vector<PseudoPatch> &patches)
{
  int rank=d_myworld->myrank();
  int numprocs=d_myworld->size();
  
  tasks_.clear();
  vector<int> procs(numprocs);
  for(int p=0;p<numprocs;p++)
    procs[p]=p;
  
  //bound local flags
  PseudoPatch patch;
  if(flags.size()>0)
  {
    patch.low=patch.high=flags[0];
    for(unsigned int f=1;f<flags.size();f++)
    {
      for(int d=0;d<3;d++)
      {
        if(flags[f][d]<patch.low[d])
          patch.low[d]=flags[f][d];
        if(flags[f][d]>patch.high[d])
          patch.high[d]=flags[f][d];
      }
    }
    //make high bounds non-inclusive
    patch.high[0]++;
    patch.high[1]++;
    patch.high[2]++;
  }
  else
  {
    //use INT_MAX to signal no patch;
    patch.low[0]=INT_MAX;
  }
  
  //Calculate global bounds
  if(numprocs>1)
  {
    vector<PseudoPatch> bounds(numprocs);
    MPI_Allgather(&patch,sizeof(PseudoPatch),MPI_BYTE,&bounds[0],sizeof(PseudoPatch),MPI_BYTE,d_myworld->getComm());

    //search for first processor that has flags 
    int p=0;
    while(bounds[p].low[0]==INT_MAX && p<numprocs )
    {
      p++;
    }

    if(p==numprocs)
    {
      //no flags exit
      return;
    }
 
    //find the bounds
    patch.low=bounds[p].low;
    patch.high=bounds[p].high;
    for(p++;p<numprocs;p++)
    {
      for(int d=0;d<3;d++)
      {
        if(bounds[p].low[0]!=INT_MAX)
        { 
          if(bounds[p].low[d]<patch.low[d])
            patch.low[d]=bounds[p].low[d];
          if(bounds[p].high[d]>patch.high[d])
            patch.high[d]=bounds[p].high[d];
        }
      }
    }
  }
  else if (flags.size()==0)
  {
    //no flags on this level
    return;
  }

  //create initial task
  BNRTask::controller_=this;
  FlagsList flagslist;
  flagslist.locs=&flags[0];
  flagslist.size=flags.size();
  tasks_.push_back(BNRTask(patch,flagslist,procs,rank,0,0));
  BNRTask *root=&tasks_.back();
 
  //place on immediate_q_
  immediate_q_.push(root);                  
  //control loop
  while(true)
  {
    BNRTask *task;
    //check tag_q for processors waiting for tags
    if(!tag_q_.empty() && tags_.size()>1)
    {
        //2 tags are available continue the task
        task=tag_q_.front();
        tag_q_.pop();
        task->continueTask();
    }
    else if(!immediate_q_.empty())  //check for tasks that are able to make progress
    {
      task=immediate_q_.front();
      immediate_q_.pop();
      //runable task found, continue task
      //cout << "rank:" << rank << ": starting from immediate_q_ with status: " << task->status_ << endl;
      if(task->p_group_.size()==1)
        task->continueTaskSerial();
      else
        task->continueTask();
    }
    else if(free_requests_.size()<requests_.size())  //no tasks can make progress finish communication
    {
      int count;
      //wait on requests
      MPI_Waitsome(requests_.size(),&requests_[0],&count,&indicies_[0],MPI_STATUSES_IGNORE);
      //handle each request
      for(int c=0;c<count;c++)
      {
        BNRTask *task=request_to_task_[indicies_[c]];
        free_requests_.push(indicies_[c]);
        if(--task->remaining_requests_==0)  //task has completed communication
        {
          if(task->status_!=TERMINATED)     //if task needs more work
          {
            immediate_q_.push(task);        //place it on the immediate_q 
          }
        }
      }
    }
    else  //no tasks remaining, no communication waiting, algorithm is done
    {
      break; 
    }
  }
 
  //check for controlling processors 
  if(rank==tasks_.front().p_group_[0])
  {
    //assign the patches to my list
    patches.assign(tasks_.front().my_patches_.begin(),tasks_.front().my_patches_.end());
  }
  if(numprocs>1)
  {
    //broad cast size out
    unsigned int size=patches.size();
    MPI_Bcast(&size,1,MPI_INT,tasks_.front().p_group_[0],d_myworld->getComm());
    //resize patchlist
    patches.resize(size);
    //broadcast patches
    MPI_Bcast(&patches[0],size*sizeof(PseudoPatch),MPI_BYTE,tasks_.front().p_group_[0],d_myworld->getComm());
  }
}

void BNRRegridder::problemSetup(const ProblemSpecP& params, 
                                const GridP& oldGrid,
                                const SimulationStateP& state)
{
  RegridderCommon::problemSetup(params, oldGrid, state);
  d_sharedState = state;

  ProblemSpecP amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");
  
  if (!regrid_spec) {
    return; // already warned about it in RC::problemSetup
  }
  // get min patch size
  regrid_spec->require("min_patch_size", d_minPatchSize);

  regrid_spec->get("patch_split_tolerance", tola_);
  regrid_spec->get("patch_combine_tolerance", tolb_);

  //bound tolerances
  if (tola_ < 0) {
    if (d_myworld->myrank() == 0)
      cout << "  Bounding Regridder's patch_split_tolerance to [0,1]\n";
    tola_ = 0;
  }
  if (tola_ > 1) {
    if (d_myworld->myrank() == 0)
      cout << "  Bounding Regridder's patch_split_tolerance to [0,1]\n";
    tola_ = 1;
  }
  if (tolb_ < 0) {
    if (d_myworld->myrank() == 0)
      cout << "  Bounding Regridder's patch_combine_tolerance to [0,1]\n";
    tolb_ = 0;
  }
  if (tolb_ > 1) {
    if (d_myworld->myrank() == 0)
      cout << "  Bounding Regridder's patch_combine_tolerance to [0,1]\n";
    tolb_ = 1;
  }
 
  //set target patches
  if(d_myworld->size()==1)
  {
    //if there is only 1 processor attempt for minimum number of patches
    target_patches_=1;
  }
  else
  {
    int patches_per_proc=1;
    regrid_spec->get("patches_per_level_per_proc",patches_per_proc);
    if (patches_per_proc<1)
    {
      if (d_myworld->myrank() == 0)
        cout << "  Bounding patches_per_level_per_proc to [1,infinity]\n";
      patches_per_proc=1;
    }
    target_patches_=patches_per_proc*d_myworld->size();
  }


  for (int k = 0; k < d_maxLevels; k++) {
    if (k < (d_maxLevels-1)) {
      problemSetup_BulletProofing(k);
    }
  }
}

//_________________________________________________________________
void BNRRegridder::problemSetup_BulletProofing(const int k)
{
  RegridderCommon::problemSetup_BulletProofing(k);

  // For 2D problems the lattice refinement ratio 
  // and the cell refinement ratio must be 1 in that plane
  for(int dir = 0; dir <3; dir++){
    if(d_cellNum[k][dir] == 1 && d_minPatchSize[dir] != 1) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is <3D. \n"
          << " The min Patch Size must be 1 in the other dimensions. \n"
          << "Grid Size: " << d_cellNum[k] 
          << " min patch size: " << d_minPatchSize << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      
    }

    if(d_cellNum[k][dir] != 1 && d_minPatchSize[dir] < 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be at least 4 cells in each dimension \n"
          << "except for 1-cell-wide dimensions.\n"
          << "  Patch size on level " << k << ": " << d_minPatchSize << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }

    if(d_cellNum[k][dir] != 1 && d_minPatchSize[dir] % d_cellRefinementRatio[k][dir] != 0) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be divisible by the cell refinement ratio\n"
          << "  Patch size on level " << k << ": " << d_minPatchSize 
          << ", refinement ratio on level " << k << ": " << d_cellRefinementRatio[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
  }
  if ( Mod( d_cellNum[k], d_minPatchSize ) != IntVector(0,0,0) ) {
    ostringstream msg;
    msg << "Problem Setup: Regridder: The overall number of cells on level " << k << "(" << d_cellNum[k] << ") is not divisible by the minimum patch size (" <<  d_minPatchSize << ")\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }
    
}

/***************************************************
 * PostFixup takes the patchset and attempts to subdivide 
 * the largest patches until target_patches_ patches exist
 * this should help the load balancer.  This function
 * does not split patches up to be smaller than the minimum
 * patch size in each dimension.
 * *************************************************/
//I am doing this in serial for now, it should be very quick and more expensive 
//to do in parallel since the number of patches is typically very small
void BNRRegridder::PostFixup(vector<PseudoPatch> &patches, IntVector min_patch_size)
{
  //calculate total volume
  int volume=0;
  for(unsigned int p=0;p<patches.size();p++)
  {
    volume+=patches[p].volume;
  }

  int volume_threshold=volume/d_myworld->size();

  //build a max heap
  make_heap(patches.begin(),patches.end());
 
  unsigned int i=patches.size()-1; 
  //split max and place children on heap until i have enough patches and patches are not to big
  while(patches.size()<target_patches_ || patches[0].volume>volume_threshold)
  {
    if(patches[0].volume==1)  //check if patch is to small to split
       return;
    
    pop_heap(patches.begin(),patches.end());

    //find max dimension
    IntVector size=patches[i].high-patches[i].low;
    
    int max_d=0;
    
    //find max
    for(int d=1;d<3;d++)
    {
      if(size[d]>size[max_d] && size[d]>1)
        max_d=d;
    }

    //calculate split point
    int index=(patches[i].high[max_d]+patches[i].low[max_d])/2;

    PseudoPatch right=patches[i];
    //adjust patches by split
    patches[i].high[max_d]=right.low[max_d]=index;

    //recalculate volumes
    size=right.high-right.low;
    right.volume=size[0]*size[1]*size[2];
    patches[i].volume-=right.volume;
    //heapify
    push_heap(patches.begin(),patches.end());
    patches.push_back(right);
    push_heap(patches.begin(),patches.end());

    i++;
  }
}

void BNRRegridder::AddSafetyLayer(const vector<PseudoPatch> patches, set<IntVector> &coarse_flags, 
                                  const vector<const Patch*>& coarse_patches, int level)
{
  
  if (coarse_patches.size() == 0)
    return;
  //create a range tree out of my patches
  PatchRangeTree prt(coarse_patches);
  //for each patch (padded with 1 cell) 
  for(unsigned p=0;p<patches.size();p++)
  {
    // convert from coarse coordinates to real coordinates on the coarser level, with a safety layer
    IntVector low = patches[p].low*d_minPatchSize/d_cellRefinementRatio[level] - d_minBoundaryCells;
    IntVector high = patches[p].high*d_minPatchSize/d_cellRefinementRatio[level] + d_minBoundaryCells;
    
    Level::selectType intersecting_patches;
    //intersect range tree
    prt.query(low, high, intersecting_patches);
    
    //for each intersecting patch
    for (int i = 0; i < intersecting_patches.size(); i++)
    {
      const Patch* patch = intersecting_patches[i];
      IntVector int_low = Max(patch->getCellLowIndex(), low);
      IntVector int_high = Min(patch->getCellHighIndex(), high);
      
      //round low coordinate down
      int_low=int_low/d_minPatchSize;
      //round high coordinate up
      int_high[0]=(int)ceil(int_high[0]/(float)d_minPatchSize[0]);
      int_high[1]=(int)ceil(int_high[1]/(float)d_minPatchSize[1]);
      int_high[2]=(int)ceil(int_high[2]/(float)d_minPatchSize[2]);
      
      //for overlapping cells
      for (CellIterator iter(int_low, int_high); !iter.done(); iter++)
      {
        //add to coarse flag list, in coarse coarse-level coordinates
        coarse_flags.insert(*iter);
      }
    }
  }
}

void BNRRegridder::writeGrid(Grid* grid,vector<vector<IntVector> > flag_sets)
{
  int levels=grid->numLevels();
  flag_sets.resize(levels);
  //write #of levels
  fout.write((char*)&levels,sizeof(int));

  //for each level
  for(int l=0;l<levels;l++)
  {
    const LevelP level=grid->getLevel(l);     
    int patches=level->numPatches();
    //write # of patches
    fout.write((char*)&patches,sizeof(int));

    //write patches
    for(Level::const_patchIterator iter=level->patchesBegin();iter!=level->patchesEnd();iter++)
    {
      const Patch *patch=*iter;
      IntVector low=patch->getInteriorCellLowIndex();
      IntVector high=patch->getInteriorCellHighIndex();
      fout.write((char*)&low,sizeof(IntVector));
      fout.write((char*)&high,sizeof(IntVector));

    }
    int numFlags=flag_sets[l].size();
    //write number of flags
    fout.write((char*)&numFlags,sizeof(int));
    //write flags
    if(numFlags>0)
      fout.write((char*)&flag_sets[l][0],sizeof(IntVector)*numFlags);
  }
  fout.flush();
}
