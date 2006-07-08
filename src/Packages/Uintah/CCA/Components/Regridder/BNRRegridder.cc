#include <Packages/Uintah/CCA/Components/Regridder/BNRRegridder.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
using namespace Uintah;

#include <vector>
#include <set>
#include <algorithm>
using namespace std;

BNRRegridder::BNRRegridder(const ProcessorGroup* pg) : RegridderCommon(pg), task_count_(0),tola_(1),tolb_(1), patchfixer_(pg)
{
  int numprocs=d_myworld->size();
  int rank=d_myworld->myrank();
  
  int *tag_ub, maxtag_ ,flag;
  int tag_start, tag_end;

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

}

BNRRegridder::~BNRRegridder()
{
}

Grid* BNRRegridder::regrid(Grid* oldGrid, SchedulerP& sched, 
                     const ProblemSpecP& ups)
{
  LoadBalancer *lb=sched->getLoadBalancer();
  
  Grid* newGrid = scinew Grid();
  ProblemSpecP grid_ps = ups->findBlock("Grid");
  
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
  level0->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
  level0->assignBCS(grid_ps);
  
  //For each level Fine -> Coarse
  for(int l=0; l < oldGrid->numLevels() && l < d_maxLevels-1;l++)
  {
    const LevelP level=oldGrid->getLevel(l);
    
    //Create Flag List
    vector<IntVector> flaglist;
    
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
         flaglist.push_back(*ci);
        }
      }
    }
    
    //if flags are tighltly bound by old grid level...
      //keep old grid level
      //continue
     
		//saftey layers? how?
		

    //Coarsen Flags
    set<IntVector> coarse_flag_set;
    for(unsigned int f=0;f<flaglist.size();f++)
    {
      coarse_flag_set.insert( flaglist[f]*d_cellRefinementRatio[l]/d_minPatchSize_[l] );
    }
    
    //create flags vector
    vector<IntVector> coarse_flag_vector(coarse_flag_set.size());
    coarse_flag_vector.assign(coarse_flag_set.begin(),coarse_flag_set.end());
		
    vector<PseudoPatch> patches;
    //Parallel BR over coarse flags
    RunBR(coarse_flag_vector,patches);  
    
    // if there are no patches on this level, don't create any more levels
    if (patches.size() == 0)
      break;
    //Fixup patchlist
    patchfixer_.FixUp(patches);
   
    PostFixup(patches,d_minPatchSize_[l]);

    //Uncoarsen
    for(unsigned int p=0;p<patches.size();p++)
    {
      patches[p].low=patches[p].low*d_minPatchSize_[l];
      patches[p].high=patches[p].high*d_minPatchSize_[l];
    }

    //create level and set up parameters
    Point anchor;
    Vector spacing;
    
    // parameters based on next-fine level.
    if (l+1 < oldGrid->numLevels()) {
      anchor = oldGrid->getLevel(l+1)->getAnchor();
      spacing = oldGrid->getLevel(l+1)->dCell();
    } else {
      anchor = newGrid->getLevel(l)->getAnchor();
      spacing = newGrid->getLevel(l)->dCell() / d_cellRefinementRatio[l];
    }
    
    LevelP newLevel = newGrid->addLevel(anchor, spacing);
    newLevel->setExtraCells(extraCells);
    
    //for each patch
    for(unsigned int p=0;p<patches.size();p++)
    {
      IntVector low = patches[p].low, high = patches[p].high;
      //create patch
      newLevel->addPatch(low, high, low, high);
    }
    
    newLevel->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    newLevel->assignBCS(grid_ps);
  }
  if (*newGrid == *oldGrid) {
    delete newGrid;
    return oldGrid;
  }


  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();


  //cout << *newGrid;
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
  
  //bound flags
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
    patch.high[0]++;
    patch.high[1]++;
    patch.high[2]++;
  }
  else
  {
    //use INT_MAX to signal no patch;
    patch.low[0]=INT_MAX;
  }
  
  
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
      if(rank==0)
        cout << "No flags on this level\n";
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
  else if (patch.low==patch.high)
  {
    cout << "No flags on this level\n";
    return;
  }

  /*
    cout << "rank: " << rank << " initial patch: {" 
    << patch.low[0] << "-" << patch.high[0] << ", "
    << patch.low[1] << "-" << patch.high[1] << ", "
    << patch.low[2] << "-" << patch.high[2] << "}\n";
  */
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
    //		cout << "rank:" << rank << ": control loop: immediate_q_.size():" << immediate_q_.size() << " delay_q_.size():" << delay_q_.size() << endl;
    BNRTask *task;
    if(!tag_q_.empty() && tags_.size()>1)
    {
        task=tag_q_.front();
        tag_q_.pop();
        task->continueTask();
    }
    else if(!immediate_q_.empty())
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
    else if(!delay_q_.empty())
    {
      //search through delay_q_ checking for finished MPI and runnable tasks
      
      //cout << rank << ": queue size b: " << delay_q_.size() << endl;			
      task=delay_q_.front();
      delay_q_.pop();
      MPI_Status status;
      //			cout << "rank:" << rank << ": finding a ready task on delay_q_\n";
      while(!task->mpi_requests_.empty())
      {
        int completed;
        //cout << "rank:" << rank << ": pid:" << task->tag_ << ": testing request: status:" << task->status << endl;
        MPI_Test(&task->mpi_requests_.front(),&completed,&status);
        if(completed)
        {
          //					cout << "rank:" << rank << ": pid:" << task->tag_ << " completed non-blocking request" << endl;
          if(status.MPI_ERROR!=0)
          {
            cout << "rank:" << rank << ": pid:" << task->tag_ << " non-blocking request returned error code: " << status.MPI_ERROR << endl;
            exit(0);
          }
          //pop request
          task->mpi_requests_.pop();
        }
        else
        {
          /*
            MPI_Status status;
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&flag,&status);
            if(flag)
            {
            cout << "rank:" << rank << ": pid:" << task->tag_ << " non-blocking request cannot be completed but message waiting: "
            << " source:" << status.MPI_SOURCE << " tag_:" << status.MPI_TAG << endl;
            
            }
          */
          //place at the end of the queue and continue searching
          delay_q_.push(task);
          //get a new task
          task=delay_q_.front();
          delay_q_.pop();
        }
      }
      //cout << rank << ": queue size a: " << delay_q_.size() << endl;			
      //runable task found, continue task
      //			cout << "rank:" << rank << ": starting from delay_q_ with status: " << task->status << endl;
      task->continueTask();
    }
    else if(tag_q_.empty())
    {
      
      //			cout << "rank:" << rank << ": all queues empty, terminating\n";
      
      break;
    }
    //cout << "rank:" << rank << ": pid:" << task->tag_ << ": task returned with status: " << task->status << endl;
    
  }
  
  if(rank==tasks_.front().p_group_[0])
  {
    patches.assign(tasks_.front().my_patches_.begin(),tasks_.front().my_patches_.end());
  }
  if(numprocs>1)
  {
    unsigned int size=tasks_.front().my_patches_.size();
    //cout << "rank:" << rank << ": pid:" << tasks_.front().tag_ << ": broadcasting size root is:" << tasks_.front().p_group_[0] << endl;
    MPI_Bcast(&size,1,MPI_INT,tasks_.front().p_group_[0],d_myworld->getComm());
    //cout << "rank:" << rank << ": pid:" << tasks_.front().tag_ << ": size is:" << size << endl;
    patches.resize(size);
  
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
  // get lattice refinement ratio, expand it to max levels
  regrid_spec->require("min_patch_size", d_minPatchSize_);

  int size = (int) d_minPatchSize_.size();
  IntVector lastRatio = d_minPatchSize_[size - 1];
  if (size < d_maxLevels-1) {
    d_minPatchSize_.resize(d_maxLevels-1);
    for (int i = size; i < d_maxLevels-1; i++)
      d_minPatchSize_[i] = lastRatio;
  }

  regrid_spec->get("patch_split_tolerance", tola_);
  regrid_spec->get("patch_combine_tolerance", tolb_);

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
 
  if(d_myworld->size()==1)
  {
    target_patches_=1;
  }
  else
  {
    //NOTE TO BRYAN: 
    //Put in code to read target patches here.  User should be able to specify a fixed number or a number per processor
    //if there is only 1 proc then target_patches_ should always be 1 (at least that makes the most sense)
    //i am not if any bullet proofing is needed.
    target_patches_=4*d_myworld->size();
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
    if(d_cellNum[k][dir] == 1 && d_minPatchSize_[k][dir] != 1) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is <3D. \n"
          << " The min Patch Size must be 1 in the other dimensions. \n"
          << "Grid Size: " << d_cellNum[k] 
          << " min patch size: " << d_minPatchSize_[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      
    }

    if(d_cellNum[k][dir] != 1 && d_minPatchSize_[k][dir] < 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be at least 4 cells in each dimension \n"
          << "except for 1-cell-wide dimensions.\n"
          << "  Patch size on level " << k << ": " << d_minPatchSize_[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
  }
  if ( Mod( d_cellNum[k], d_minPatchSize_[k] ) != IntVector(0,0,0) ) {
    ostringstream msg;
    msg << "Problem Setup: Regridder: The overall number of cells on level " << k << "(" << d_cellNum[k] << ") is not divisible by the minimum patch size (" <<  d_minPatchSize_[k] << ")\n";
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
  if(patches.size()>=target_patches_) //enough patches do not do a post fixup
  {
    return;
  }
  //build a max heap
  make_heap(patches.begin(),patches.end());
  //split max and place children on heap until i have enough patches
  unsigned int i=patches.size()-1;
  while(patches.size()<target_patches_)
  {
    IntVector size;
    pop_heap(patches.begin(),patches.end());
    //find max dimension
    size=patches[i].high-patches[i].low;
    
    //skip dimensions that splitting would make them to small
    int max_d=0;
    while(size[max_d]<2*min_patch_size[max_d] && max_d<3)
       max_d++;

    if(max_d==3)  //no dimensions can be split
    {
        break;
    }
    //find max
    for(int d=max_d+1;d<3;d++)
    {
      if(size[d]>size[max_d] && size[d]>=2*min_patch_size[d])
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

