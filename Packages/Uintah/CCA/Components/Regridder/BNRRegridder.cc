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
    tag1=tags_.front(); 
    tags_.pop();
    tag2=tags_.front();
    tags_.pop(); 
    return true;  
  }
  //check if tags can be allocated 
  else if(free_tags>1)
  {
    tag1=free_tag_start_; free_tag_start_++;
    tag2=free_tag_start_; free_tag_start_++;
    return true;
  }
  //check if 1 tag is on the queue and 1 avialable at the end
  else if(tags_.size()==1 && free_tags==1)
  {
    tag1=tags_.front();
    tags_.pop();
    tag2=free_tag_start_; free_tag_start_++;
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
      maxtag_=*tag_ub;
    else
      maxtag_=32767;

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

Grid* BNRRegridder::regrid(Grid* oldGrid)
{
  vector<set<IntVector> > coarse_flag_sets(oldGrid->numLevels());
  vector< vector<Region> > patch_sets(min(oldGrid->numLevels()+1,d_maxLevels));

  vector<int> processor_assignments;

  //create coarse flag sets
  CreateCoarseFlagSets(oldGrid,coarse_flag_sets);

  //add old level 0 to patch sets
  for (Level::const_patchIterator p = oldGrid->getLevel(0)->patchesBegin(); p != oldGrid->getLevel(0)->patchesEnd(); p++)
  {
    patch_sets[0].push_back(Region((*p)->getInteriorCellLowIndex(),(*p)->getInteriorCellHighIndex()));
  }

  //For each level Fine to Coarse
  for(int l=min(oldGrid->numLevels()-1,d_maxLevels-2); l >= 0;l--)
  {
    //create coarse flag vector
    vector<IntVector> coarse_flag_vector(coarse_flag_sets[l].size());
    coarse_flag_vector.assign(coarse_flag_sets[l].begin(),coarse_flag_sets[l].end());

    //Parallel BR over coarse flags
      //flags on level l are used to create patches on level l+1
   
    RunBR(coarse_flag_vector,patch_sets[l+1]);  
    if(patch_sets[l+1].empty()) //no patches goto next level
      continue;

    //Fixup patchlist:  this forces neighbor constraints
    patchfixer_.FixUp(patch_sets[l+1]);

    if(!d_loadBalance)
    {
      //Post fixup patchlist: this creates more patches as specified in the input file in order to improve load balance
      PostFixup(patch_sets[l+1]);
    }
    //uncoarsen
    for(unsigned int p=0;p<patch_sets[l+1].size();p++)
    {
      patch_sets[l+1][p].low()=patch_sets[l+1][p].getLow()*d_minPatchSize[l+1];
      patch_sets[l+1][p].high()=patch_sets[l+1][p].getHigh()*d_minPatchSize[l+1];
    }
    
    //add saftey layers for finer level
    if(l>0)
    {
      //patches on level l must be at least one cell more than on level l+1
        //so use the patches on level l+1 to add extended flags to level l-1
        //(thus extending patches on level l)
      AddSafetyLayer(patch_sets[l+1], coarse_flag_sets[l-1], lb_->getPerProcessorPatchSet(oldGrid->getLevel(l-1))->getSubset(d_myworld->myrank())->getVector(), l);
    }
  }
  //if lb
  if(d_loadBalance)
  {
    lb_->dynamicallyLoadBalanceAndSplit(oldGrid,d_minPatchSize,patch_sets,true);
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
  
  OutputGridStats(patch_sets, newGrid);

  newGrid->performConsistencyCheck();
  return newGrid;
}
Grid* BNRRegridder::CreateGrid(Grid* oldGrid, vector<vector<Region> > &patch_sets )
{

  Grid* newGrid = scinew Grid();
  
  Vector spacing = oldGrid->getLevel(0)->dCell();
  Point anchor = oldGrid->getLevel(0)->getAnchor();
  IntVector extraCells = oldGrid->getLevel(0)->getExtraCells();
  IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
  
  //For each level Coarse -> Fine
  for(int l=0; l < oldGrid->numLevels()+1 && l < d_maxLevels;l++)
  {
    // if level is not needed, don't create any more levels
    if(patch_sets[l].size()==0)
       break;
  
    LevelP level = newGrid->addLevel(anchor, spacing);
    level->setExtraCells(extraCells);

    //cout << "New level " << l << " num patches " << patch_sets[l-1].size() << endl;
    //for each patch
    for(unsigned int p=0;p<patch_sets[l].size();p++)
    {
      IntVector low = patch_sets[l][p].getLow();
      IntVector high = patch_sets[l][p].getHigh();
      //create patch
      level->addPatch(low, high, low, high);
    }

    level->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    level->assignBCS(grid_ps_);
    
    // parameters based on next-fine level.
    spacing = spacing / d_cellRefinementRatio[l];
  }
  return newGrid;
}
void BNRRegridder::CreateCoarseFlagSets(Grid *oldGrid, vector<set<IntVector> > &coarse_flag_sets)
{
  DataWarehouse *dw=sched_->getLastDW();

  int toplevel=min(oldGrid->numLevels(),d_maxLevels-1);
  for(int l=0; l<toplevel ;l++)
  {
    const LevelP level=oldGrid->getLevel(l);
    //create coarse flag set
    const PatchSubset *ps=lb_->getPerProcessorPatchSet(level)->getSubset(d_myworld->myrank());
    for(int p=0;p<ps->size();p++)
    {
      const Patch *patch=ps->get(p);
      constCCVariable<int> flags;
      dw->get(flags, d_dilatedCellsRegridLabel, 0, patch, Ghost::None, 0);
      for (CellIterator ci(patch->getInteriorCellLowIndex(), patch->getInteriorCellHighIndex()); !ci.done(); ci++)
      {
        //cout << "Checking flag:" << *ci << " value is:" << flags[*ci] << endl;
        if (flags[*ci])
        {
         //to coarsen the flags on level l
            //multiply by the cell refinement ratio on level l to convert to level l+1 coordinates
            //divide by the minimum patch size on level l+1 to coarsen
         coarse_flag_sets[l].insert(*ci*d_cellRefinementRatio[l]/d_minPatchSize[l+1]);
        }
      }
    }
  }
}


void BNRRegridder::OutputGridStats(vector< vector<Region> > &patch_sets, Grid* newGrid)
{
  if (dbgstats.active() && d_myworld->myrank() == 0) 
  {
    dbgstats << " Grid Statistics:\n";
    for (unsigned int l = 0; l < patch_sets.size(); l++) 
    {
      if(patch_sets[l].empty())
        break;

      double total_cells=0;
      double sum_of_cells_squared=0;
      int n = patch_sets[l].size();
      //calculate total cells and cells squared
      for(int p=0;p<n;p++)
      {
        double cells=double(patch_sets[l][p].getVolume());
        total_cells+=cells;
        sum_of_cells_squared+=cells*cells;
      }
      //calculate conversion factor into simulation coordinates
      double factor=1;
      for(int d=0;d<3;d++)
      {
          factor*=newGrid->getLevel(l)->dCell()[d];
      }
      //calculate mean
      double mean = total_cells /(double) n;
      double stdv = sqrt((sum_of_cells_squared-total_cells*total_cells/(double)n)/(double)n);
      dbgstats << left << "  L" << setw(8) << l+1 << ": Patches: " << setw(8) << n << " Total Cells: " << setw(8) << total_cells << " Mean Cells: " << setw(8) << mean << " stdv: " << setw(8) << stdv << " relative stdv: " << setw(8) << stdv/mean << " Volume: " << setw(8) << total_cells*factor << endl;
    }
  }
}

void BNRRegridder::RunBR( vector<IntVector> &flags, vector<Region> &patches)
{
  int rank=d_myworld->myrank();
  int numprocs=d_myworld->size();
  
  vector<int> procs(numprocs);
  for(int p=0;p<numprocs;p++)
    procs[p]=p;
  
  //bound local flags
  Region patch;
  if(flags.size()>0)
  {
    patch.low()=patch.high()=flags[0];
    for(unsigned int f=1;f<flags.size();f++)
    {
      patch.low()=Min(patch.getLow(),flags[f]);
      patch.high()=Max(patch.getHigh(),flags[f]);
    }
    //make high bounds non-inclusive
    patch.high()=patch.getHigh()+IntVector(1,1,1);
  }
  else
  {
    //use INT_MAX to signal no patch;
    patch.low()[0]=INT_MAX;
  }
  //Calculate global bounds
  if(numprocs>1)
  {
    vector<Region> bounds(numprocs);
    MPI_Allgather(&patch,sizeof(Region),MPI_BYTE,&bounds[0],sizeof(Region),MPI_BYTE,d_myworld->getComm());

    //search for first processor that has flags 
    int p=0;
    while(bounds[p].getLow()[0]==INT_MAX && p<numprocs )
    {
      p++;
    }

    if(p==numprocs)
    {
      //no flags on any processors so exit
      return;   
    }
 
    //find the bounds
    patch=bounds[p];
    for(p++;p<numprocs;p++)
    {
      if(bounds[p].getLow()[0]!=INT_MAX)
      { 
          patch.low()=Min(patch.getLow(),bounds[p].getLow());
          patch.high()=Max(patch.getHigh(),bounds[p].getHigh());
      }
    }
  }

  //create initial task
  BNRTask::controller_=this;
  FlagsList flagslist;
  if(flags.size()>0)
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
        if(--(task->remaining_requests_)==0)  //task has completed communication
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
    MPI_Bcast(&patches[0],size*sizeof(Region),MPI_BYTE,tasks_.front().p_group_[0],d_myworld->getComm());
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

  int size=d_minPatchSize.size();

  //it is not required to specifiy the minimum patch size on each level
  //if every level is not specified reuse the lowest level minimum patch size
  IntVector lastSize = d_minPatchSize[size - 1];
  if (size < d_maxLevels) {
    d_minPatchSize.reserve(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++)
      d_minPatchSize.push_back(lastSize);
  }
  
  //calculate the minimum patch size on level 0
  IntVector min_size(INT_MAX,INT_MAX,INT_MAX);

  LevelP level=oldGrid->getLevel(0);

  for(Level::patchIterator patch=level->patchesBegin();patch<level->patchesEnd();patch++)
  {
    IntVector size=(*patch)->getInteriorCellHighIndex()-(*patch)->getInteriorCellLowIndex();
    min_size=Min(min_size,size);
  }
  d_minPatchSize.insert(d_minPatchSize.begin(),min_size);


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
    if(k!=0 && d_cellNum[k][dir] == 1 && d_minPatchSize[k][dir] != 1) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is <3D. \n"
          << " The min Patch Size must be 1 in the other dimensions. \n"
          << "Grid Size: " << d_cellNum[k] 
          << " min patch size: " << d_minPatchSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      
    }

    if(k!=0 && d_cellNum[k][dir] != 1 && d_minPatchSize[k][dir] < 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be at least 4 cells in each dimension \n"
          << "except for 1-cell-wide dimensions.\n"
          << "  Patch size on level " << k << ": " << d_minPatchSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }

    if(k!=0 && d_cellNum[k][dir] != 1 && d_minPatchSize[k][dir] % d_cellRefinementRatio[k][dir] != 0) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be divisible by the cell refinement ratio\n"
          << "  Patch size on level " << k << ": " << d_minPatchSize[k] 
          << ", refinement ratio on level " << k << ": " << d_cellRefinementRatio[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
  }
  if (k!=0 && Mod( d_cellNum[k], d_minPatchSize[k] ) != IntVector(0,0,0) ) {
    ostringstream msg;
    msg << "Problem Setup: Regridder: The overall number of cells on level " << k << "(" << d_cellNum[k] << ") is not divisible by the minimum patch size (" <<  d_minPatchSize[k] << ")\n";
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
void BNRRegridder::PostFixup(vector<Region> &patches)
{
  //calculate total volume
  int volume=0;
  for(unsigned int p=0;p<patches.size();p++)
  {
    volume+=patches[p].getVolume();
  }

  double volume_threshold=volume/d_myworld->size()*d_patchRatioToTarget;
  
  //build a max heap
  make_heap(patches.begin(),patches.end(),Region::VolumeCompare());
 
  unsigned int i=patches.size()-1; 
  //split max and place children on heap until i have enough patches and patches are not to big
  while(patches.size()<target_patches_ || patches[0].getVolume()>volume_threshold)
  {
    if(patches[0].getVolume()==1)  //check if patch is to small to split
       return;
    
    pop_heap(patches.begin(),patches.end(),Region::VolumeCompare());

    //find max dimension
    IntVector size=patches[i].getHigh()-patches[i].getLow();
    
    int max_d=0;
    
    //find max
    for(int d=1;d<3;d++)
    {
      if(size[d]>size[max_d] && size[d]>1)
        max_d=d;
    }

    //calculate split point
    int index=(patches[i].getHigh()[max_d]+patches[i].getLow()[max_d])/2;

    Region right=patches[i];
    //adjust patches by split
    patches[i].high()[max_d]=right.low()[max_d]=index;

    //heapify
    push_heap(patches.begin(),patches.end(),Region::VolumeCompare());
    patches.push_back(right);
    push_heap(patches.begin(),patches.end(),Region::VolumeCompare());

    i++;
  }
}

//patches are on level l+1
//coarse flags are for level l-1
//coarse patches are for level l-1
void BNRRegridder::AddSafetyLayer(const vector<Region> patches, set<IntVector> &coarse_flags, 
                                  const vector<const Patch*>& coarse_patches, int l)
{
  if (coarse_patches.size() == 0)
    return;
  //create a range tree out of my patches
  PatchRangeTree prt(coarse_patches);
  
  //for each patch
  for(unsigned p=0;p<patches.size();p++)
  {
    //add a saftey layer and convert coordinates to a coarser level by dividing by the refinement ratios.  
    IntVector low = (patches[p].getLow()-d_minBoundaryCells)/d_cellRefinementRatio[l]/d_cellRefinementRatio[l-1];
    IntVector high= Ceil( (patches[p].getHigh()+d_minBoundaryCells).asVector()/d_cellRefinementRatio[l].asVector()/d_cellRefinementRatio[l-1].asVector());
    
    //clamp low and high points to domain boundaries 
    for(int d=0;d<3;d++)
    {
      if(low[d]<0)
      {
         low[d]=0;
      }
      if(high[d]>d_cellNum[l-1][d])
      {
          high[d]=d_cellNum[l-1][d];
      }
    }
    Level::selectType intersecting_patches;
    //intersect range tree
    prt.query(low, high, intersecting_patches);
     
    //for each intersecting patch
    for (int i = 0; i < intersecting_patches.size(); i++)
    {
      const Patch* patch = intersecting_patches[i];

      IntVector int_low = Max(patch->getInteriorCellLowIndex(), low);
      IntVector int_high = Min(patch->getInteriorCellHighIndex(), high);
      
      //convert to coarsened coordinates
        //multiply by refinement ratio to convert back to fine levels cell coordinates
        //divide by minimum patch size to coarsen
      int_low=int_low*d_cellRefinementRatio[l-1]/d_minPatchSize[l];
      int_high=Ceil(int_high*d_cellRefinementRatio[l-1].asVector()/d_minPatchSize[l].asVector());
      //for overlapping cells
      for (CellIterator iter(int_low, int_high); !iter.done(); iter++)
      {
        //add to coarse flag list, in coarse coarse-level coordinates
        coarse_flags.insert(*iter);
      }
    }
  }
}

