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
using namespace std;

BNRRegridder::BNRRegridder(const ProcessorGroup* pg) : RegridderCommon(pg), task_count(0),tola(1),tolb(1), patchfixer(pg)
{
  comm=d_myworld->getComm();
  int numprocs=d_myworld->size();
  int rank=d_myworld->myrank();
  
  int *tagub, maxtag ,flag;
  int tag_start, tag_end;
  
  MPI_Attr_get(comm,MPI_TAG_UB,&tagub,&flag);
  if(flag)
    maxtag=(*tagub>>1);
  else
    maxtag=(32767>>1);

  maxtag++;
  int div=maxtag/numprocs;
  int rem=maxtag%numprocs;
  
  tag_start=div*rank;
  
  if(rank<rem)
    tag_start+=rank;
  else
    tag_start+=rem;
  
  if(rank<rem)
    tag_end=tag_start+div+1;
  else
    tag_end=tag_start+div;
  
  
  for(int i=tag_start;i<tag_end;i++)
    tags.push(i<<1);
  //	cout << "rank:" << rank << ": tag_start:" << tag_start << " tags_end:" << tag_end << endl;
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
      coarse_flag_set.insert(flaglist[f]/d_minPatchSize[l]);
    }
    
    //create flags vector
    vector<IntVector> coarse_flag_vector(coarse_flag_set.size());
    coarse_flag_vector.assign(coarse_flag_set.begin(),coarse_flag_set.end());
		
    vector<PseudoPatch> patches;
    //Parallel BR over coarse flags
    RunBR(coarse_flag_vector,patches);  
    //Fixup patchlist
    patchfixer.FixUp(patches);
    
    //Uncoarsen
    IntVector mult=d_minPatchSize[l]*d_cellRefinementRatio[l];
    for(unsigned int p=0;p<patches.size();p++)
    {
      patches[p].low=patches[p].low*mult;
      patches[p].high=patches[p].high*mult;
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


  cout << *newGrid;
  newGrid->performConsistencyCheck();
  return newGrid;
}


void BNRRegridder::RunBR( vector<IntVector> &flags, vector<PseudoPatch> &patches)
{
  int rank=d_myworld->myrank();
  int numprocs=d_myworld->size();
  
  tasks.clear();
  
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
  
  
  vector<PseudoPatch> bounds(numprocs);
  MPI_Allgather(&patch,sizeof(PseudoPatch),MPI_BYTE,&bounds[0],sizeof(PseudoPatch),MPI_BYTE,comm);
  
  patch.low=bounds[0].low;
  patch.high=bounds[0].high;
  for(int p=1;p<numprocs;p++)
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
  /*
    cout << "rank: " << rank << " initial patch: {" 
    << patch.low[0] << "-" << patch.high[0] << ", "
    << patch.low[1] << "-" << patch.high[1] << ", "
    << patch.low[2] << "-" << patch.high[2] << "}\n";
  */
  //create initial task
  BNRTask::controller=this;
  FlagsList flagslist;
  flagslist.locs=&flags[0];
  flagslist.size=flags.size();
  tasks.push_back(BNRTask(patch,flagslist,procs,rank,0,0));
  BNRTask *root=&tasks.back();
  //increment task count to insure tag uniqueness
  if(rank==0)
  {
    if(tags.front()!=0)
      cout << "ERROR wrong tag at front\n";
    tags.pop(); 
    task_count++;
  }
  
  //place on immediate_q
  immediate_q.push(root);									
  //control loop
  while(true)
  {
    //		cout << "rank:" << rank << ": control loop: immediate_q.size():" << immediate_q.size() << " delay_q.size():" << delay_q.size() << endl;
    BNRTask *task;
    if(!tag_q.empty() && tags.size()>1)
    {
        task=tag_q.front();
        tag_q.pop();
        task->continueTask();
    }
    else if(!immediate_q.empty())
    {
      task=immediate_q.front();
      immediate_q.pop();
      //runable task found, continue task
      //			cout << "rank:" << rank << ": starting from immediate_q with status: " << task->status << endl;
      task->continueTask();
    }
    else if(!delay_q.empty())
    {
      //search through delay_q checking for finished MPI and runnable tasks
      
      //cout << rank << ": queue size b: " << delay_q.size() << endl;			
      task=delay_q.front();
      delay_q.pop();
      MPI_Status status;
      //			cout << "rank:" << rank << ": finding a ready task on delay_q\n";
      while(!task->mpi_requests.empty())
      {
        int completed;
        //cout << "rank:" << rank << ": pid:" << task->tag << ": testing request: status:" << task->status << endl;
        MPI_Test(&task->mpi_requests.front(),&completed,&status);
        if(completed)
        {
          //					cout << "rank:" << rank << ": pid:" << task->tag << " completed non-blocking request" << endl;
          if(status.MPI_ERROR!=0)
          {
            cout << "rank:" << rank << ": pid:" << task->tag << " non-blocking request returned error code: " << status.MPI_ERROR << endl;
            exit(0);
          }
          //pop request
          task->mpi_requests.pop();
        }
        else
        {
          /*
            MPI_Status status;
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&flag,&status);
            if(flag)
            {
            cout << "rank:" << rank << ": pid:" << task->tag << " non-blocking request cannot be completed but message waiting: "
            << " source:" << status.MPI_SOURCE << " tag:" << status.MPI_TAG << endl;
            
            }
          */
          //place at the end of the queue and continue searching
          delay_q.push(task);
          //get a new task
          task=delay_q.front();
          delay_q.pop();
        }
      }
      //cout << rank << ": queue size a: " << delay_q.size() << endl;			
      //runable task found, continue task
      //			cout << "rank:" << rank << ": starting from delay_q with status: " << task->status << endl;
      task->continueTask();
    }
    else if(tag_q.empty())
    {
      
      //			cout << "rank:" << rank << ": all queues empty, terminating\n";
      
      break;
    }
    //cout << "rank:" << rank << ": pid:" << task->tag << ": task returned with status: " << task->status << endl;
    
  }
  
  /*	
    int p_rank=tasks.front().p_rank;
    
    if(rank==0 && rank==p_rank)
    {
      //root processor already has patch list, copy it over
      patches.assign(tasks.front().my_patches.begin(),tasks.front().my_patches.end());
    }
    else 
    {
      if(rank==0)	//recieve patch list
      {
        MPI_Status status;
        int size;
        MPI_Recv(&size,1,MPI_INT,MPI_ANY_SOURCE,0,comm,&status);
        cout << "size is : " << size << endl;
        patches.resize(size);
        MPI_Recv(&patches[0],size*sizeof(PseudoPatch),MPI_BYTE,MPI_ANY_SOURCE,0,comm,&status);
      }
      else if(p_rank==0)	//send patch list
      {
        unsigned int size=tasks.front().my_patches.size();
        MPI_Send(&size,1,MPI_INT,0,0,comm);
        MPI_Send(&tasks.front().my_patches[0],size*sizeof(PseudoPatch),MPI_BYTE,0,0,comm);
      }	
    }
  */
  //cout << "rank:" << rank << ": remaing tags on this proc:" << tags.size() << endl;;
  if(rank==tasks.front().p_group[0])
  {
    patches.assign(tasks.front().my_patches.begin(),tasks.front().my_patches.end());
  }
  unsigned int size=tasks.front().my_patches.size();
  //cout << "rank:" << rank << ": pid:" << tasks.front().tag << ": broadcasting size root is:" << tasks.front().p_group[0] << endl;
  MPI_Bcast(&size,1,MPI_INT,tasks.front().p_group[0],comm);
  //cout << "rank:" << rank << ": pid:" << tasks.front().tag << ": size is:" << size << endl;
  patches.resize(size);
  
  MPI_Bcast(&patches[0],size*sizeof(PseudoPatch),MPI_BYTE,tasks.front().p_group[0],comm);
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
  regrid_spec->require("min_patch_size", d_minPatchSize);

  int size = (int) d_minPatchSize.size();
  IntVector lastRatio = d_minPatchSize[size - 1];
  if (size < d_maxLevels-1) {
    d_minPatchSize.resize(d_maxLevels-1);
    for (int i = size; i < d_maxLevels-1; i++)
      d_minPatchSize[i] = lastRatio;
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
    if(d_cellNum[k][dir] == 1 && d_minPatchSize[k][dir] != 1) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is <3D. \n"
          << " The min Patch Size must be 1 in the other dimensions. \n"
          << "Grid Size: " << d_cellNum[k] 
          << " min patch size: " << d_minPatchSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      
    }

    if(d_cellNum[k][dir] != 1 && d_minPatchSize[k][dir] < 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be at least 4 cells in each dimension \n"
          << "except for 1-cell-wide dimensions.\n"
          << "  Patch size on level " << k << ": " << d_minPatchSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
  }
}

