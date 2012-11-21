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

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Regridder/BNRRegridder.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/PatchBVH/PatchBVH.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <CCA/Ports/SFC.h>
using namespace Uintah;

#include <vector>
#include <set>
#include <algorithm>
#include <iomanip>
using namespace std;


static DebugStream times("BNRTimes",false);
static DebugStream grid_dbg("GridDBG",false);
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
  MALLOC_TRACE_TAG_SCOPE("BNRRegridder::regrid");
  double t[6]={0};
  double avg[6];
  
  double start=Time::currentSeconds();
  
  TAU_PROFILE("BNRRegridder::regrid", " ", TAU_USER);
  vector<set<IntVector> > coarse_flag_sets(oldGrid->numLevels());
  vector< vector<Region> > patch_sets(min(oldGrid->numLevels()+1,d_maxLevels));

  vector<int> processor_assignments;

  //create coarse flag sets
  CreateCoarseFlagSets(oldGrid,coarse_flag_sets);
 
  //add old level 0 to patch sets
  for (Level::const_patchIterator p = oldGrid->getLevel(0)->patchesBegin(); p != oldGrid->getLevel(0)->patchesEnd(); p++)
  {
    patch_sets[0].push_back(Region((*p)->getCellLowIndex(),(*p)->getCellHighIndex()));
  }
    
  int rank=d_myworld->myrank();
  int procs=d_myworld->size();
  MPI_Comm comm=d_myworld->getComm();
  
  t[0]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();

  //For each level Fine to Coarse
  for(int l=min(oldGrid->numLevels()-1,d_maxLevels-2); l >= 0;l--)
  {
    t[1]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();
    //create coarse flag vector
    vector<IntVector> coarse_flag_vector(coarse_flag_sets[l].size());
    coarse_flag_vector.assign(coarse_flag_sets[l].begin(),coarse_flag_sets[l].end());
   
    //Calcualte coarsening factor
    int coarsen_factor=d_minPatchSize[l+1][0]*d_minPatchSize[l+1][1]*d_minPatchSize[l+1][2]/d_cellRefinementRatio[l][0]/d_cellRefinementRatio[l][1]/d_cellRefinementRatio[l][2];
      
    TAU_PROFILE_TIMER(combinetimer, "BNRRegridder::consolidate flags", "", TAU_USER);
    TAU_PROFILE_START(combinetimer);
    
    //Calculate the number of stages to reduce
    //this is a guess based on the coarsening factor and the number of processors
    int stages=static_cast<int>(log((float)coarsen_factor)/log(2.0) + log((float)procs)/log(2.0)/4);
    int stride=1;
    MPI_Status status;
    //consoldate flags along a hypercube sending the shortest distance first
      //this is important for keeping flags clustered(ordered according to LB)
    for(int i=0;i<stages;i++)
    {
     if(rank%(stride*2)==0)
     {
      if(rank+stride<procs)
      {
        //recieve from rank+stride
        int size=coarse_flag_vector.size();
        int numReceive;
        //recieve number of flags
        MPI_Recv(&numReceive,1,MPI_INT,rank+stride,0,comm,&status);
        coarse_flag_vector.resize(size+numReceive);
        //recieve new flags
        MPI_Recv(&coarse_flag_vector[size],numReceive*sizeof(IntVector),MPI_BYTE,rank+stride,0,comm,&status);
      }
     }
     else
     {
       //send to rank-stride
       int numSend=coarse_flag_vector.size();
       //send number of flags
       MPI_Send(&numSend,1,MPI_INT,rank-stride,0,comm);
       //send flags
       MPI_Send(&coarse_flag_vector[0],numSend*sizeof(IntVector),MPI_BYTE,rank-stride,0,comm);
       coarse_flag_vector.clear();
       break;
     }
     stride*=2;
    }
    //send flags to the begining processors
      //this is important for being able to exploit on-node communication
    stride=1<<stages; 
    if(rank%stride==0)
    {
        int to=rank/stride;
        if(to!=rank)
        {
          int numSend=coarse_flag_vector.size();
          //send number of flags
          MPI_Send(&numSend,1,MPI_INT,to,0,comm);
          //send flags
          MPI_Send(&coarse_flag_vector[0],numSend*sizeof(IntVector),MPI_BYTE,to,0,comm);
          coarse_flag_vector.clear();
        }
    }
   
    if(rank<ceil(procs/(float)stride))
    {
      int from=rank*stride;
      if(from!=rank)
      {
        //recieve from rank+stride
        int numReceive;
        //recieve number of flags
        MPI_Recv(&numReceive,1,MPI_INT,from,0,comm,&status);
        coarse_flag_vector.resize(numReceive);
        //recieve new flags
        MPI_Recv(&coarse_flag_vector[0],numReceive*sizeof(IntVector),MPI_BYTE,from,0,comm,&status);
      }
    }
    TAU_PROFILE_STOP(combinetimer);
    //Parallel BR over coarse flags
      //flags on level l are used to create patches on level l+1
   
    RunBR(coarse_flag_vector,patch_sets[l+1]);  
    if(patch_sets[l+1].empty()) //no patches goto next level
      continue;

    //Fixup patchlist:  this forces neighbor constraints
    patchfixer_.FixUp(patch_sets[l+1]);

    //Post fixup patchlist: this creates more patches as specified in the input file in order to improve load balance
    PostFixup(patch_sets[l+1]);
    
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
  t[2]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
 
  //Create the grid
  Grid *newGrid = CreateGrid(oldGrid,patch_sets);
  if (newGrid->isSimilar(*oldGrid)) 
  {
    delete newGrid;
    return oldGrid;
  }

  //finalize the grid
  TAU_PROFILE_TIMER(finalizetimer, "BNRRegridder::finalize grid", "", TAU_USER);
  TAU_PROFILE_START(finalizetimer);
  IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
  
  t[3]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
  for(int l=0;l<newGrid->numLevels();l++)
  {
    LevelP level= newGrid->getLevel(l);
    level->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    //level->assignBCS(grid_ps_,0);
  }
  t[4]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
  TAU_PROFILE_STOP(finalizetimer);
  
  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();
  
  OutputGridStats(patch_sets, newGrid);

  //initialize the weights on new patches
  lb_->initializeWeights(oldGrid,newGrid);
  t[5]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
 
  if(times.active())
  {
    MPI_Reduce(&t,&avg,6,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    if(d_myworld->myrank()==0)
    {
      times << "BNRTimes: ";
      for(int i=0;i<6;i++)
        times << avg[i]/d_myworld->size() << " ";
      times << endl;
    }
  }

#if SCI_ASSERTION_LEVEL > 0
  if(!verifyGrid(newGrid))
  {
    throw InternalError("Grid is not consistent across processes",__FILE__,__LINE__);
  }
#endif 

  return newGrid;
}
Grid* BNRRegridder::CreateGrid(Grid* oldGrid, vector<vector<Region> > &patch_sets )
{
  MALLOC_TRACE_TAG_SCOPE("BNRRegridd::CreateGrid");
  TAU_PROFILE("BNRRegridder::CreateGrid()", " ", TAU_USER);

  Grid* newGrid = scinew Grid();
  
  Vector spacing = oldGrid->getLevel(0)->dCell();
  Point anchor = oldGrid->getLevel(0)->getAnchor();
  IntVector extraCells = oldGrid->getLevel(0)->getExtraCells();
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
      level->addPatch(low, high, low, high,newGrid);
    }
    // parameters based on next-fine level.
    spacing = spacing / d_cellRefinementRatio[l];
  }
  return newGrid;
}
void BNRRegridder::CreateCoarseFlagSets(Grid *oldGrid, vector<set<IntVector> > &coarse_flag_sets)
{
  MALLOC_TRACE_TAG_SCOPE("BNRRegridder::CreateCoarseFlagSets");
  TAU_PROFILE("BNRRegridder::CreateCoarseFlagSets()", " ", TAU_USER);
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
      for (CellIterator ci(patch->getCellLowIndex(), patch->getCellHighIndex()); !ci.done(); ci++)
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
  if (d_myworld->myrank() == 0) 
  {
    cout << " Grid Statistics:\n";
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
      cout << left << "  L" << setw(8) << l+1 << ": Patches: " << setw(8) << n << " Total Cells: " << setw(8) << total_cells << " Mean Cells: " << setw(8) << mean << " stdv: " << setw(8) << stdv << " relative stdv: " << setw(8) << stdv/mean << " Volume: " << setw(8) << total_cells*factor << endl;
    }
  }
}

void BNRRegridder::RunBR( vector<IntVector> &flags, vector<Region> &patches)
{
  MALLOC_TRACE_TAG_SCOPE("BNRRegridder::RunBR");
  TAU_PROFILE("BNRRegridder::RunBR()", " ", TAU_USER);
  int rank=d_myworld->myrank();
  int numprocs=d_myworld->size();
 
  vector<int> procs(numprocs);
  BNRTask *root=0;  
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
  int prank=-1;
  //Calculate global bounds
  if(numprocs>1)
  {
    vector<Region> bounds(numprocs);
    MPI_Allgather(&patch,sizeof(Region),MPI_BYTE,&bounds[0],sizeof(Region),MPI_BYTE,d_myworld->getComm());

    //calculate participating processor set
    int count=0;
    for(int p=0;p<numprocs;p++)
    {
      if(bounds[p].getLow()[0]!=INT_MAX)
      {
        if(p==rank)
        {
          prank=count;
        }
        procs[count++]=p;   
      }
    }
  
    if(count==0) 
    {
      //no flags on any processors so exit
      return;   
    }
            
    procs.resize(count);
    
    //find the bounds
    patch=bounds[procs[0]];
    for(int p=1;p<count;p++)
    {
      patch.low()=Min(patch.getLow(),bounds[procs[p]].getLow());
      patch.high()=Max(patch.getHigh(),bounds[procs[p]].getHigh());
    }
  }
  
  if(flags.size()>0)
  {
    //create initial task
    BNRTask::controller_=this;
    FlagsList flagslist;
 
    flagslist.locs=&flags[0];
    flagslist.size=flags.size();
    tasks_.push_back(BNRTask(patch,flagslist,procs,prank,0,0));
    root=&tasks_.back();
 
    //place on immediate_q_
    immediate_q_.push(root);                  
    //control loop
    //MPI_Errhandler_set(d_myworld->getComm(), MPI_ERRORS_RETURN);
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
        if(task->p_group_.size()==1)
          task->continueTaskSerial();
        else
          task->continueTask();
      }
      else if(free_requests_.size()<requests_.size())  //no tasks can make progress finish communication
      {
        int count;
        //wait on requests
        //MPI_STATUSES_IGNORE
        if(MPI_Waitsome(requests_.size(),&requests_[0],&count,&indicies_[0],&statuses_[0])==MPI_ERR_IN_STATUS)
        {
                BNRTask *task;
                cerr << "rank:" << rank << " error in MPI_Waitsome status\n";
                for(int c=0;c<count;c++)
                {
                  if(statuses_[c].MPI_ERROR!=MPI_SUCCESS)
                  {
                    char message[MPI_MAX_ERROR_STRING];
                    int length;
                    
                    MPI_Error_string(statuses_[c].MPI_ERROR,message,&length);
                    cerr << "Error message" << ": '" << message << "'\n";
                  
                    task=request_to_task_[indicies_[c]];
                    cerr << "Task status:" << task->status_ << " patch:" << task->patch_ << endl;
                  }
                }
                cerr << "Entering infinite loop so debugger can be attached\n";
                while(1); //hang so debugger can be attached
        }
      
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
    if(rank==root->p_group_[0])
    {
      //assign the patches to my list
      patches.assign(root->my_patches_.begin(),root->my_patches_.end());
    }
  }
  
  if(numprocs>1)
  {
    //communicate the patchset to rank 0 for broadcasting
    if(root!=0 && rank==root->p_group_[0] && rank!=0) //if I am the root and not rank 0
    {
      int size=patches.size();
      //send to rank 0
      MPI_Send(&size,1,MPI_INT,0,0,d_myworld->getComm());
      MPI_Send(&patches[0],size*sizeof(Region),MPI_BYTE,0,0,d_myworld->getComm());
    }
    else if(rank==0 && (root==0 || rank!=root->p_group_[0])) //if I am rank 0 and not the root
    {
      MPI_Status status;
      int size;
      //receive from any rank
      MPI_Recv(&size,1,MPI_INT,MPI_ANY_SOURCE,0,d_myworld->getComm(),&status);
      patches.resize(size);
      MPI_Recv(&patches[0],size*sizeof(Region),MPI_BYTE,MPI_ANY_SOURCE,0,d_myworld->getComm(),&status);
    }
          
    //broadcast size out
    unsigned int size=patches.size();
    MPI_Bcast(&size,1,MPI_INT,0,d_myworld->getComm());
    //resize patchlist
    patches.resize(size);
    //broadcast patches
    MPI_Bcast(&patches[0],size*sizeof(Region),MPI_BYTE,0,d_myworld->getComm());
  }
 
  tasks_.clear();
}

void BNRRegridder::problemSetup(const ProblemSpecP& params, 
                                const GridP& oldGrid,
                                const SimulationStateP& state)
{
  if(d_myworld->myrank()==0)
    cout << " WARNING: The BNR regridder has performance issues and has been superseeded by the tiled regridder\n";

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
    for (int i = size; i < d_maxLevels-1; i++)
      d_minPatchSize.push_back(lastSize);
  }
   

  LevelP level=oldGrid->getLevel(0);
  
  //calculate the patch size on level 0
  IntVector patch_size(0,0,0);

  for(Level::patchIterator patch=level->patchesBegin();patch<level->patchesEnd();patch++)
  {
    IntVector size=(*patch)->getCellHighIndex()-(*patch)->getCellLowIndex();
    if(patch_size==IntVector(0,0,0))
      patch_size=size;
    if(size!=patch_size)
    {
      ostringstream msg;
      msg << "Problem Setup Error: The patch size on level 0 is not constant. \n"
          << "The BNR Regridder requires all patches on level 0 to be constant. \n"
          << "Please update your input file so that the number of patches divides into the resolution evenly. \n";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }
  }
  d_minPatchSize.insert(d_minPatchSize.begin(),patch_size);


  regrid_spec->get("patch_split_tolerance", tola_);
  regrid_spec->get("patch_combine_tolerance", tolb_);
  regrid_spec->getWithDefault("patch_ratio_to_target",d_patchRatioToTarget,.125);
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
    if (k < (d_maxLevels)) {
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
      msg << "Problem Setup: Regridder: Min Patch Size needs to be greater than 4 cells in each dimension \n"
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
  MALLOC_TRACE_TAG_SCOPE("BNRRegridder::PostFixup");
  TAU_PROFILE("BNRRegridder::PostFixup()", " ", TAU_USER);
  //calculate total volume
  int volume=0;
  for(unsigned int p=0;p<patches.size();p++)
  {
    volume+=patches[p].getVolume();
  }

  double volume_threshold=volume/(float)d_myworld->size()*d_patchRatioToTarget;
  
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
  MALLOC_TRACE_TAG_SCOPE("BNRRegridder::AddSafetyLayer");
  TAU_PROFILE("BNRRegridder::AddSafetyLayer()", " ", TAU_USER);
  if (coarse_patches.size() == 0)
    return;
  //create a range tree out of my patches
  PatchBVH pbvh(coarse_patches);
  
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
    pbvh.query(low, high, intersecting_patches);
     
    //for each intersecting patch
    for (int i = 0; i < intersecting_patches.size(); i++)
    {
      const Patch* patch = intersecting_patches[i];

      IntVector int_low = Max(patch->getCellLowIndex(), low);
      IntVector int_high = Min(patch->getCellHighIndex(), high);
      
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

bool BNRRegridder::verifyGrid(Grid *grid)
{
  //if we are running in serial there is no reason to verify that each processor has the same grid.
  if(d_myworld->size()==1)
    return true;

  vector<int> checksums;
  vector<int> their_checksums;
  vector<string> labels;

  int num_levels=grid->numLevels();
  grid_dbg << d_myworld->myrank() << " Grid number of levels:" << num_levels << endl;
  their_checksums.resize(d_myworld->size());
  MPI_Gather(&num_levels,1,MPI_INT,&their_checksums[0],1,MPI_INT,0,d_myworld->getComm());

  if(d_myworld->myrank()==0)
  {
    for(int i=0;i<d_myworld->size();i++)
    {
      if(num_levels!=their_checksums[i])
      {
        cout << d_myworld->myrank() << " Error number of levels does not match on rank " << i << " my levels:" << num_levels << " their levels:" << their_checksums[i] << endl;
        return false;
      }
    }
  }
  for(int i=0;i<num_levels;i++)
  {
    LevelP level=grid->getLevel(i);
    checksums.push_back(level->numPatches());
    char label[100];
    sprintf(label,"Patchset on level %d",i);
    labels.push_back(label);

    IntVector Sum;
    IntVector Diff;
    int sum=0;
    int diff=0;
    for(int p=0;p<level->numPatches();p++)
    {
      const Patch* patch = level->getPatch(p); 
      grid_dbg << d_myworld->myrank() << "    Level: " << i << " Patch " << p << ": " << *patch << endl;
      Sum=Abs(patch->getCellHighIndex())+Abs(patch->getCellLowIndex());
      Diff=Abs(patch->getCellHighIndex())-Abs(patch->getCellLowIndex());
      
      sum+=Sum[0]*Sum[1]*Sum[2]*(p+1);
      diff+=Diff[0]*Diff[1]*Diff[2]*(p+1000000);
      //cout << d_myworld->myrank() << " patch:" << *patch << " sum:" << Sum[0]*Sum[1]*Sum[2]*(p+1) << " diff:" << Diff[0]*Diff[1]*Diff[2]*(p+1) << endl;
    }
    checksums[i]+=(sum+diff);
  }

  their_checksums.resize(checksums.size()*d_myworld->size());
  MPI_Gather(&checksums[0],checksums.size(),MPI_INT,&their_checksums[0],checksums.size(),MPI_INT,0,d_myworld->getComm());
 
  if(d_myworld->myrank()==0)
  {
    for(int p=0;p<d_myworld->size();p++)
    {
      for(unsigned int i=0;i<checksums.size();i++)
      {
        if(checksums[i]!=their_checksums[p*checksums.size()+i])
        {
          cout << d_myworld->myrank() << " Error grid inconsistency: " << labels[i] << " does not match on rank:" << p << endl;
          return false;
        }
      }
    }
  }
  //if(d_myworld->myrank()==0)
  //  cout << " GRIDS ARE CONSISTENT\n";
  return true;
}
