#include <Packages/Uintah/CCA/Components/Regridder/BNRRegridder.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/PatchRangeTree.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/CCA/Ports/SFC.h>
using namespace Uintah;

#include <vector>
#include <set>
#include <algorithm>
#include <iomanip>
using namespace std;

static DebugStream dbgstats("BNRStats",false);

bool BNRRegridder::getTags(int &tag1, int &tag2)
{

  int free_tags=free_tag_end_-free_tag_start_;
  
  //check if queue has tags
  if(tags_.size()>1)
  {
    tag1=tags_.top(); 
    tags_.pop();
    tag2=tags_.top();
    tags_.pop(); 
    return true;  
  }
  //check if tags can be allocated 
  else if(free_tags>1)
  {
    tag1=free_tag_start_<<1; free_tag_start_++;
    tag2=free_tag_start_<<1; free_tag_start_++;
    return true;
  }
  //check if 1 tag is on the queue and 1 avialable at the end
  else if(tags_.size()==1 && free_tags==1)
  {
    tag1=tags_.top();
    tags_.pop();
    tag2=free_tag_start_<<1; free_tag_start_++;
    return true;
  }
  //no more tags available
  else
  {
    return false;
  }
}
BNRRegridder::BNRRegridder(const ProcessorGroup* pg) : RegridderCommon(pg), task_count_(0),tola_(1),tolb_(1), patchfixer_(pg)
{
  int numprocs=d_myworld->size();
  int rank=d_myworld->myrank();
  
  int *tag_ub, maxtag_ ,flag;

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
  
    free_tag_start_=div*rank;
  
    if(rank<rem)
      free_tag_start_+=rank;
    else
      free_tag_start_+=rem;
  
    if(rank<rem)
      free_tag_end_=free_tag_start_+div+1;
    else
      free_tag_end_=free_tag_start_+div;

    //don't have zero in the tag list  
    if(rank==0)
      free_tag_start_++;
  }
}

BNRRegridder::~BNRRegridder()
{
}

template<int DIM> 
void BNRRegridder::LoadBalance(vector<PseudoPatch> &patches)
{
    vector<double> positions;
    vector<DistributedIndex> indices;

    int *dimensions=d_sharedState->getActiveDims();

    IntVector low=patches[0].low,high=patches[0].high;
    
    //bound patches and compute positions
    for(unsigned int p=0;p<patches.size();p++)
    {
      PseudoPatch &patch=patches[p];
      low=min(low,patch.low);
      high=max(high,patch.high);
      Vector point=(patch.high+patch.low).asVector()/2.0;

      int proc = (p*d_myworld->size())/patches.size();
      if(d_myworld->myrank()==proc)
      {
        for(int d=0;d<DIM;d++)
        {
          positions.push_back(point[dimensions[d]]);
        }
      }
    }
    Vector range=(high-low).asVector();
    Vector center=(high+low).asVector()/2.0;

    //make parameters for SFC curve
    double r[DIM], c[DIM], delta[DIM];
    for(int d=0;d<DIM;d++)
    {
      r[d]=range[dimensions[d]];
      c[d]=center[dimensions[d]];
      delta[d]=d_minPatchSize[dimensions[d]];
    }

    //move to load balancer
    SFC<double> sfc(DIM,d_myworld);
    //end move to load balacner


    sfc.SetLocalSize(positions.size()/DIM);
    sfc.SetDimensions(r);
    sfc.SetRefinementsByDelta(delta);
    sfc.SetCenter(c);
    sfc.SetLocations(&positions);
    sfc.SetOutputVector(&indices);
    
    //move to load balancer
    sfc.SetMergeMode(1);
    sfc.SetCleanup(BATCHERS);
    sfc.SetMergeParameters(3000,500,2,.15);  //add profiling every X timesteps?
    //end move to load balancer

    sfc.GenerateCurve();

    //get costs

    //balance curve
       //split where necessary
          //split by calculating optimal split or in half along longest dimension?
}


Grid* BNRRegridder::regrid(Grid* oldGrid)
{

  vector<set<IntVector> > coarse_flag_sets(oldGrid->numLevels());
  vector< vector<PseudoPatch> > patch_sets(oldGrid->numLevels());

  vector<int> processor_assignments;

//  //create coarse flag sets
  CreateCoarseFlagSets(oldGrid,coarse_flag_sets);

  //For each level Fine to Coarse
  for(int l=oldGrid->numLevels()-1; l >= 0;l--)
  {
    if(l>=d_maxLevels-1)
      continue;

    //create coarse flag vector
    vector<IntVector> coarse_flag_vector(coarse_flag_sets[l].size());
    coarse_flag_vector.assign(coarse_flag_sets[l].begin(),coarse_flag_sets[l].end());
  
    //Parallel BR over coarse flags
    RunBR(coarse_flag_vector,patch_sets[l]);  

    if(patch_sets[l].empty()) //no patches goto next level
       continue;

    //add saftey layers for finer level
    if(l>0)
    {
      AddSafetyLayer(patch_sets[l], coarse_flag_sets[l-1], lb_->getPerProcessorPatchSet(oldGrid->getLevel(l-1))->getSubset(d_myworld->myrank())->getVector(), l);
    }
    //Fixup patchlist:  this forces neighbor constraints
    patchfixer_.FixUp(patch_sets[l]);
     
    if(!d_loadBalance)
    {
      //Post fixup patchlist: this creates more patches as specified in the input file in order to improve load balance
      PostFixup(patch_sets[l]);
    }
    //uncoarsen
    for(unsigned int p=0;p<patch_sets[l].size();p++)
    {
      patch_sets[l][p].low=patch_sets[l][p].low*d_minPatchSize;
      patch_sets[l][p].high=patch_sets[l][p].high*d_minPatchSize;
    }

    //if lb
    if(d_loadBalance)
    {
      switch(d_sharedState->getNumDims())
      {
        case 1:
          LoadBalance<1>(patch_sets[l]);
          break;
        case 2:
          LoadBalance<2>(patch_sets[l]);
          break;
        case 3:
          LoadBalance<3>(patch_sets[l]);
          break;
        default:
          throw InternalError("Invalid Number of Dimensions",__FILE__,__LINE__);
      }
    }
  }

  //Create the grid
  Grid *newGrid = CreateGrid(oldGrid,patch_sets);

  if (*newGrid == *oldGrid) 
  {
    delete newGrid;
    return oldGrid;
  }

  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();
  
  OutputGridStats(patch_sets);

  newGrid->performConsistencyCheck();
  return newGrid;
}
Grid* BNRRegridder::CreateGrid(Grid* oldGrid, vector< vector<PseudoPatch> > &patch_sets )
{
  Grid* newGrid = scinew Grid();
  
  Vector spacing = oldGrid->getLevel(0)->dCell();
  Point anchor = oldGrid->getLevel(0)->getAnchor();
  IntVector extraCells = oldGrid->getLevel(0)->getExtraCells();
  IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
  
    
  //Create Level 0
  LevelP level = newGrid->addLevel(anchor, spacing);
  level->setExtraCells(extraCells);

  for (Level::const_patchIterator iter = oldGrid->getLevel(0)->patchesBegin(); iter != oldGrid->getLevel(0)->patchesEnd(); iter++)
  {
    const Patch* p = *iter;
    IntVector inlow = p->getInteriorCellLowIndex();
    IntVector low = p->getCellLowIndex();
    IntVector inhigh = p->getInteriorCellHighIndex();
    IntVector high = p->getCellHighIndex();
    level->addPatch(low, high, inlow, inhigh);
  }
  level->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
  level->assignBCS(grid_ps_);

  //For each level Coarse -> Fine
  for(int l=1; l < oldGrid->numLevels()+1 && l < d_maxLevels;l++)
  {
    // if level is not needed, don't create any more levels
    if(patch_sets[l-1].size()==0)
       break;

    // parameters based on next-fine level.
    spacing = spacing / d_cellRefinementRatio[l];
  
    LevelP level = newGrid->addLevel(anchor, spacing);
    level->setExtraCells(extraCells);

    //cout << "New level " << l << " num patches " << patch_sets[l-1].size() << endl;
    //for each patch
    for(unsigned int p=0;p<patch_sets[l-1].size();p++)
    {
      IntVector low = patch_sets[l-1][p].low;
      IntVector high = patch_sets[l-1][p].high;
      //create patch
      level->addPatch(low, high, low, high);
    }

    level->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    level->assignBCS(grid_ps_);
  }
  return newGrid;
}
void BNRRegridder::CreateCoarseFlagSets(Grid *oldGrid, vector<set<IntVector> > &coarse_flag_sets)
{
  for(int l=oldGrid->numLevels()-1; l >= 0;l--)
  {
    if(l>=d_maxLevels-1)
      continue;

    const LevelP level=oldGrid->getLevel(l);
    //create coarse flag set
    const PatchSubset *ps=lb_->getPerProcessorPatchSet(level)->getSubset(d_myworld->myrank());
    for(int p=0;p<ps->size();p++)
    {
      const Patch *patch=ps->get(p);
      DataWarehouse *dw=sched_->getLastDW();
      constCCVariable<int> flags;
      dw->get(flags, d_dilatedCellsCreationLabel, 0, patch, Ghost::None, 0);
      for (CellIterator ci(patch->getInteriorCellLowIndex(), patch->getInteriorCellHighIndex()); !ci.done(); ci++)
      {
        if (flags[*ci])
        {
         coarse_flag_sets[l].insert(*ci*d_cellRefinementRatio[l]/d_minPatchSize);
        }
      }
    }
  }
}


void BNRRegridder::OutputGridStats(vector< vector<PseudoPatch> > &patch_sets)
{
  if (dbgstats.active() && d_myworld->myrank() == 0) 
  {
    dbgstats << " Grid Statistics:\n";
    for (unsigned int l = 0; l < patch_sets.size(); l++) 
    {
      if(patch_sets[l].empty())
        break;

      double total_vol=0;
      double sum_of_vol_squared=0;
      int n = patch_sets[l].size();
      //calculate total volume and volume squared
      double vol_mult=double(d_minPatchSize[0]*d_minPatchSize[1]*d_minPatchSize[2]);
      for(int p=0;p<n;p++)
      {
        double vol=double(patch_sets[l][p].volume*vol_mult);
        total_vol+=vol;
        sum_of_vol_squared+=vol*vol;
      }
      //calculate mean
      double mean = total_vol /(double) n;
      double stdv = sqrt((sum_of_vol_squared-total_vol*total_vol/(double)n)/(double)n);
      dbgstats << left << "  L" << setw(8) << l+1 << ": Patches: " << setw(8) << n << " Volume: " << setw(8) << total_vol<< " Mean Volume: " << setw(8) << mean << " stdv: " << setw(8) << stdv << " relative stdv: " << setw(8) << stdv/mean << endl;
    }
  }
}

void BNRRegridder::RunBR( vector<IntVector> &flags, vector<PseudoPatch> &patches)
{
  int rank=d_myworld->myrank();
  int numprocs=d_myworld->size();
  
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

    if(!tag_q_.empty() && tags_.size() + free_tag_end_ - free_tag_start_>1 )
    {
        //2 tags are available continue the task
        task=tag_q_.front();
        tag_q_.pop();
        task->continueTask();
    }
    else if(!immediate_q_.empty())  //check for tasks that are able to make progress
    {
      task=immediate_q_.top();
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
    else if(tag_q_.empty())  //no tasks remaining, no communication waiting, algorithm is done
    {
      break; 
    }
    else
    {
      //no tasks on the immediate_q, tasks are on the taq_q
      if(tags_.size() + free_tag_end_ - free_tag_start_ < 2) //this if might not be needed 
      {
        throw InternalError("Not enough tags",__FILE__,__LINE__);
      }
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
 
  tasks_.clear();

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

  regrid_spec->getWithDefault("do_loadBalancing",d_loadBalance,false);
  regrid_spec->get("patch_split_tolerance", tola_);
  regrid_spec->get("patch_combine_tolerance", tolb_);
  regrid_spec->getWithDefault("patch_ratio_to_target",d_patchRatioToTarget,.25);
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
void BNRRegridder::PostFixup(vector<PseudoPatch> &patches)
{
  //calculate total volume
  int volume=0;
  for(unsigned int p=0;p<patches.size();p++)
  {
    volume+=patches[p].volume;
  }

  double volume_threshold=volume/(d_myworld->size()*d_patchRatioToTarget);

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
  //for each patch (padded with saftey layer) 
  
  for(unsigned p=0;p<patches.size();p++)
  {
    //add saftey layer and convert from coarse coordinates to real coordinates on the coarser level
    IntVector low = (patches[p].low*d_minPatchSize-d_minBoundaryCells)/d_cellRefinementRatio[level]/d_cellRefinementRatio[level-1];
    IntVector high;
    high[0] = (int)ceil((patches[p].high[0]*d_minPatchSize[0]+d_minBoundaryCells[0])/(float)d_cellRefinementRatio[level][0]/d_cellRefinementRatio[level-1][0]);
    high[1] = (int)ceil((patches[p].high[1]*d_minPatchSize[1]+d_minBoundaryCells[1])/(float)d_cellRefinementRatio[level][1]/d_cellRefinementRatio[level-1][1]);
    high[2] = (int)ceil((patches[p].high[2]*d_minPatchSize[2]+d_minBoundaryCells[2])/(float)d_cellRefinementRatio[level][2]/d_cellRefinementRatio[level-1][2]);
     
    //clamp low and high points to domain boundaries 
    for(int d=0;d<3;d++)
    {
      if(low[d]<0)
      {
         low[d]=0;
      }
      if(high[d]>d_cellNum[level-1][d])
      {
          high[d]=d_cellNum[level-1][d];
      }
    }
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
      int_low=int_low*d_cellRefinementRatio[level-1]/d_minPatchSize;
      //round high coordinate up
      int_high[0]=(int)ceil(int_high[0]*d_cellRefinementRatio[level-1][0]/(float)d_minPatchSize[0]);
      int_high[1]=(int)ceil(int_high[1]*d_cellRefinementRatio[level-1][1]/(float)d_minPatchSize[1]);
      int_high[2]=(int)ceil(int_high[2]*d_cellRefinementRatio[level-1][2]/(float)d_minPatchSize[2]);

      //for overlapping cells
      for (CellIterator iter(int_low, int_high); !iter.done(); iter++)
      {
        //add to coarse flag list, in coarse coarse-level coordinates
        coarse_flags.insert(*iter);
      }
    }
  }
}

