
#include <Packages/Uintah/CCA/Components/SimulationController/AMRSimulationController.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/Array3.h>
#include <Core/Thread/Time.h>
#include <Core/OS/Dir.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <TauProfilerForSCIRun.h>
#include <iostream>
#include <iomanip>
#include <values.h>

#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
#include <list>
#include <fstream>
#include <math.h>
#endif

#include <Core/Malloc/Allocator.h> // for memory leak tests...

using std::cerr;
using std::cout;

using namespace SCIRun;
using namespace Uintah;

static DebugStream amrout("AMR", false);

AMRSimulationController::AMRSimulationController(const ProcessorGroup* myworld) :
  SimulationController(myworld)
{
   d_restarting = false;
}

AMRSimulationController::~AMRSimulationController()
{
}

void AMRSimulationController::doRestart(std::string restartFromDir, int timestep,
				     bool fromScratch, bool removeOldDir)
{
   d_restarting = true;
   d_restartFromDir = restartFromDir;
   d_restartTimestep = timestep;
   d_restartFromScratch = fromScratch;
   d_restartRemoveOldDir = removeOldDir;
}

// currently defined in SimpleSimulationController.cc
double stdDeviation(double sum_of_x, double sum_of_x_squares, int n);

void AMRSimulationController::run()
{
   UintahParallelPort* pp = getPort("problem spec");
   ProblemSpecInterface* psi = dynamic_cast<ProblemSpecInterface*>(pp);
   
   // Get the problem specification
   ProblemSpecP ups = psi->readInputFile();
   ups->writeMessages(d_myworld->myrank() == 0);
   if(!ups)
      throw ProblemSetupException("Cannot read problem specification");
   
   releasePort("problem spec");
   
   if(ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification");

   bool log_dw_mem=false;
#ifndef DISABLE_SCI_MALLOC
   ProblemSpecP debug = ups->findBlock("debug");
   if(debug){
     ProblemSpecP log_mem = debug->findBlock("logmemory");
     if(log_mem)
       log_dw_mem=true;
   }
#endif

   Output* output = dynamic_cast<Output*>(getPort("output"));
   output->problemSetup(ups);
   
   // Setup the initial grid
   GridP grid=scinew Grid();

   problemSetup(ups, grid);
   
   if(grid->numLevels() == 0){
      cerr << "No problem specified.  Exiting AMRSimulationController.\n";
      return;
   }
   
   // Check the grid
   grid->performConsistencyCheck();
   // Print out meta data
   if (d_myworld->myrank() == 0)
     grid->printStatistics();

   SimulationStateP sharedState = scinew SimulationState(ups);
   
   // Initialize the CFD and/or MPM components
   SimulationInterface* sim = dynamic_cast<SimulationInterface*>(getPort("sim"));
   if(!sim)
     throw InternalError("No simulation component");
   sim->problemSetup(ups, grid, sharedState);

   // Finalize the shared state/materials
   sharedState->finalizeMaterials();

   Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
   sched->problemSetup(ups);
   SchedulerP scheduler(sched);

   LoadBalancer* lb = dynamic_cast<LoadBalancer*>
     (dynamic_cast<SchedulerCommon*>(sched)->getPort("load balancer"));
   lb->problemSetup(ups);

   output->initializeOutput(ups);

   if(d_myworld->myrank() == 0){
     cout << "Compiling initialization taskgraph...\n";
   }
   double start = Time::currentSeconds();
   
   scheduler->initialize(1, 1);
   scheduler->advanceDataWarehouse(grid);

   double t;

   // Parse time struct
   SimulationTime timeinfo(ups);

   // Thsi has to change for AMR restarting, to support the readin
   // of grid hierarchies (and with that a scheduler hierarchie)
   if(d_restarting){
     // create a temporary DataArchive for reading in the checkpoints
     // archive for restarting.
     Dir restartFromDir(d_restartFromDir);
     Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
     DataArchive archive(checkpointRestartDir.getName(),
			 d_myworld->myrank(), d_myworld->size());
     
     double delt = 0;
     archive.restartInitialize(d_restartTimestep, grid,
			       scheduler->get_dw(1), &t, &delt);
     
     output->restartSetup(restartFromDir, 0, d_restartTimestep, t,
			  d_restartFromScratch, d_restartRemoveOldDir);
     sharedState->setCurrentTopLevelTimeStep( output->getCurrentTimestep() );
     // Tell the scheduler the generation of the re-started simulation.
     // (Add +1 because the scheduler will be starting on the next
     // timestep.)
     scheduler->setGeneration( output->getCurrentTimestep()+1 );
     
     scheduler->get_dw(1)->setID( output->getCurrentTimestep() );
     scheduler->get_dw(1)->finalize();
     sim->restartInitialize();
   } else {
     sharedState->setCurrentTopLevelTimeStep( 0 );
     // Initialize the CFD and/or MPM data
     for(int i=0;i<grid->numLevels();i++)
       sim->scheduleInitialize(grid->getLevel(i), scheduler);
   }
   
   double start_time = Time::currentSeconds();

   if (!d_restarting){
     t = timeinfo.initTime;
     for(int i=0;i<grid->numLevels();i++)
       sim->scheduleComputeStableTimestep(grid->getLevel(i),scheduler);
   }

   if(output)
      output->finalizeTimestep(t, 0, grid, scheduler);

   amrout << "Compiling initial schedule\n";
   scheduler->compile(d_myworld);
   
   double dt=Time::currentSeconds()-start;
   if(d_myworld->myrank() == 0)
     cout << "done taskgraph compile (" << dt << " seconds)\n";
   // No scrubbing for initial step
   scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
   scheduler->execute(d_myworld);

   int n = 0;
   double prevWallTime = Time::currentSeconds();
   double sum_of_walltime = 0; // sum of all walltimes
   double sum_of_walltime_squares = 0; // sum all squares of walltimes

   std::vector<int> levelids;

   ////////////////////////////////////////////////////////////////////////////
   // The main time loop; here the specified problem is actually getting solved
   
   int  iterations = 0;
   double prev_delt = 0;
   while( ( t < timeinfo.maxTime ) && 
	  ( iterations < timeinfo.num_time_steps ) ) {

     // After one step (either timestep or initialization) and correction
     // the delta we can finally, finalize our old timestep, eg. 
     // finalize and advance the Datawarehouse
     scheduler->advanceDataWarehouse(grid);

     // Compute number of dataWarehouses
     int totalFine=1;
     for(int i=1;i<grid->numLevels();i++)
       totalFine *= grid->getLevel(i)->timeRefinementRatio();
     
     iterations ++;
     double wallTime = Time::currentSeconds() - start_time;
 
     delt_vartype delt_var;
     DataWarehouse* oldDW = scheduler->get_dw(0);
     oldDW->get(delt_var, sharedState->get_delt_label());

     double delt = delt_var;
     delt *= timeinfo.delt_factor;
      
     if(delt < timeinfo.delt_min){
       if(d_myworld->myrank() == 0)
	 cerr << "WARNING: raising delt from " << delt
	      << " to minimum: " << timeinfo.delt_min << '\n';
       delt = timeinfo.delt_min;
     }
     if(iterations > 1 && timeinfo.max_delt_increase < 1.e90
	&& delt > (1+timeinfo.max_delt_increase)*prev_delt){
       if(d_myworld->myrank() == 0)
	 cerr << "WARNING: lowering delt from " << delt 
	      << " to maxmimum: " << (1+timeinfo.max_delt_increase)*prev_delt
	      << " (maximum increase of " << timeinfo.max_delt_increase
	      << ")\n";
       delt = (1+timeinfo.max_delt_increase)*prev_delt;
     }
     if(t <= timeinfo.initial_delt_range && delt > timeinfo.max_initial_delt){
       if(d_myworld->myrank() == 0)
	 cerr << "WARNING: lowering delt from " << delt 
	      << " to maximum: " << timeinfo.max_initial_delt
	      << " (for initial timesteps)\n";
       delt = timeinfo.max_initial_delt;
     }
     if(delt > timeinfo.delt_max){
       if(d_myworld->myrank() == 0)
	 cerr << "WARNING: lowering delt from " << delt 
	      << " to maximum: " << timeinfo.delt_max << '\n';
       delt = timeinfo.delt_max;
     }
     prev_delt=delt;

#ifndef DISABLE_SCI_MALLOC
     size_t nalloc,  sizealloc, nfree,  sizefree, nfillbin,
       nmmap, sizemmap, nmunmap, sizemunmap, highwater_alloc,  
       highwater_mmap;
      
     GetGlobalStats(DefaultAllocator(),
		    nalloc, sizealloc, nfree, sizefree,
		    nfillbin, nmmap, sizemmap, nmunmap,
		    sizemunmap, highwater_alloc, highwater_mmap);
     unsigned long memuse = sizealloc - sizefree;
     unsigned long highwater = highwater_mmap;
#else
     unsigned long memuse = (char*)sbrk(0)-start_addr;
     unsigned long highwater = 0;
#endif

     unsigned long avg_memuse = memuse;
     unsigned long max_memuse = memuse;
     unsigned long avg_highwater = highwater;
     unsigned long max_highwater = highwater;
     if (d_myworld->size() > 1) {
       MPI_Reduce(&memuse, &avg_memuse, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
		  d_myworld->getComm());
       if(highwater){
	 MPI_Reduce(&highwater, &avg_highwater, 1, MPI_UNSIGNED_LONG,
		    MPI_SUM, 0, d_myworld->getComm());
       }
       avg_memuse /= d_myworld->size(); // only to be used by processor 0
       avg_highwater /= d_myworld->size();
       MPI_Reduce(&memuse, &max_memuse, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0,
		  d_myworld->getComm());
       if(highwater){
	 MPI_Reduce(&highwater, &max_highwater, 1, MPI_UNSIGNED_LONG,
		    MPI_MAX, 0, d_myworld->getComm());
       }
     }
     
     if(log_dw_mem){
       scheduler->logMemoryUse();
       ostringstream fn;
       fn << "alloc." << setw(5) << setfill('0') << d_myworld->myrank() << ".out";
       string filename(fn.str());
       DumpAllocator(DefaultAllocator(), filename.c_str());
     }
     
     //output timestep statistics
     if(d_myworld->myrank() == 0){
       cout << "Time=" << t << ", delT=" << delt 
	    << ", elap T = " << wallTime 
	    << ", DW: " << oldDW->getID() << ", Mem Use = ";
       if (avg_memuse == max_memuse && avg_highwater == max_highwater){
	 cout << avg_memuse;
	 if(avg_highwater)
	   cout << "/" << avg_highwater;
       } else {
	 cout << avg_memuse;
	 if(avg_highwater)
	   cout << "/" << avg_highwater;
	 cout << " (avg), " << max_memuse;
	 if(max_highwater)
	   cout << "/" << max_highwater;
	 cout << " (max)";
       }
       cout << endl;

       // calculate mean/std dev
       if (n > 2) // ignore times 0,1,2
       {
	 //wallTimes.push_back(wallTime - prevWallTime);
	 sum_of_walltime += (wallTime - prevWallTime);
	 sum_of_walltime_squares += pow(wallTime - prevWallTime,2);
       }
       if (n > 3) {
	 double stdDev, mean;
	 
	 // divide by n-2 and not n, because we wait till n>2 to keep track
	 // of our stats
	 stdDev = stdDeviation(sum_of_walltime, sum_of_walltime_squares, n-2);
	 mean = sum_of_walltime / (n-2);
	 //	  ofstream timefile("avg_elapsed_walltime.txt");
	 //	  timefile << mean << " +- " << stdDev << endl;
	 cout << "Timestep mean: " << mean << " +- " << stdDev << endl;
       }
       prevWallTime = wallTime;
       n++;
     }

     // Put the current time into the shared state so other components
     // can access it.  Also increment (by one) the current time step
     // number so components can tell what timestep they are on.
     sharedState->setElapsedTime(t);
     sharedState->incrementCurrentTopLevelTimeStep();

     if(needRecompile(t, delt, grid, sim, output, lb, levelids)){
       if(d_myworld->myrank() == 0)
	 cout << "Compiling taskgraph...\n";
       double start = Time::currentSeconds();
       
       scheduler->initialize(1, totalFine);
       scheduler->fillDataWarehouses(grid);

       // Set up new DWs, DW mappings.
       scheduler->clearMappings();
       scheduler->mapDataWarehouse(Task::OldDW, 0);
       scheduler->mapDataWarehouse(Task::NewDW, totalFine);
       
       sim->scheduleTimeAdvance(grid->getLevel(0), scheduler, 0, 1);
       
       if(grid->numLevels() > 1)
	 subCycle(grid, scheduler, sharedState, 0, totalFine, 1, sim);

       scheduler->clearMappings();
       scheduler->mapDataWarehouse(Task::OldDW, 0);
       scheduler->mapDataWarehouse(Task::NewDW, totalFine);
       for(int i=0;i<grid->numLevels();i++){
	 Task* task = scinew Task("initializeErrorEstimate", this,
				  &AMRSimulationController::initializeErrorEstimate,
				  sharedState);
	 task->computes(sharedState->get_refineFlag_label());
	 sched->addTask(task, grid->getLevel(i)->eachPatch(),
			sharedState->allMaterials());
	 sim->scheduleErrorEstimate(grid->getLevel(i), scheduler);
	 sim->scheduleComputeStableTimestep(grid->getLevel(i), scheduler);
       }
       if(output)
	 output->finalizeTimestep(t, delt, grid, scheduler);

       scheduler->compile(d_myworld);

       double dt=Time::currentSeconds()-start;
       if(d_myworld->myrank() == 0)
	 cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
       levelids.resize(grid->numLevels());
       for(int i=0;i<grid->numLevels();i++)
	 levelids[i]=grid->getLevel(i)->getID();
     }

     oldDW->override(delt_vartype(delt), sharedState->get_delt_label());
     double delt_fine = delt;
     int skip=totalFine;
     for(int i=0;i<grid->numLevels();i++){
       const Level* level = grid->getLevel(i).get_rep();
       if(i != 0){
	 delt_fine /= level->timeRefinementRatio();
	 skip /= level->timeRefinementRatio();
       }
       for(int idw=0;idw<totalFine;idw+=skip){
	 DataWarehouse* dw = scheduler->get_dw(idw);
	 dw->override(delt_vartype(delt_fine), sharedState->get_delt_label(),
		      level);
       }
     }

     // Execute the current timestep
     for(int i=0;i<totalFine;i++)
       scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubComplete);
     scheduler->get_dw(totalFine)->setScrubbing(DataWarehouse::ScrubNonPermanent);
     scheduler->execute(d_myworld);
     if(output)
       output->executedTimestep();
     // remesh here!

     t += delt;
     TAU_DB_DUMP();
   }

   ups->releaseDocument();
   amrout << "ALL DONE\n";
}


void AMRSimulationController::subCycle(GridP& grid, SchedulerP& scheduler,
				       SimulationStateP& sharedState,
				       int startDW, int dwStride, int numLevel,
				       SimulationInterface* sim)
{
  amrout << "Start AMRSimulationController::subCycle, level=" << numLevel << '\n';
  // We are on (the fine) level numLevel
  LevelP fineLevel = grid->getLevel(numLevel);
  LevelP coarseLevel = grid->getLevel(numLevel-1);

  int numSteps = 2; // Make this configurable - Steve
  int newDWStride = dwStride/numSteps;

  ASSERT((newDWStride > 0 && numLevel+1 < grid->numLevels()) || (newDWStride == 0 || numLevel+1 == grid->numLevels()));
  int curDW = startDW;
  for(int step=0;step < numSteps;step++){
    scheduler->clearMappings();
    scheduler->mapDataWarehouse(Task::OldDW, curDW);
    scheduler->mapDataWarehouse(Task::NewDW, curDW+newDWStride);
    scheduler->mapDataWarehouse(Task::CoarseOldDW, startDW);
    scheduler->mapDataWarehouse(Task::CoarseNewDW, startDW+dwStride);

    sim->scheduleRefineInterface(fineLevel, scheduler, step, numSteps);
    sim->scheduleTimeAdvance(fineLevel, scheduler, step, numSteps);
    if(numLevel+1 < grid->numLevels()){
      ASSERT(newDWStride > 0);
      subCycle(grid, scheduler, sharedState, curDW, newDWStride,
	       numLevel+1, sim);
    }
    curDW += newDWStride;
  }
  // Coarsening always happens at the final timestep, completely within the
  // last DW
  scheduler->clearMappings();
  scheduler->mapDataWarehouse(Task::OldDW, 0);
  scheduler->mapDataWarehouse(Task::NewDW, curDW);
  sim->scheduleCoarsen(coarseLevel, scheduler);
}


void AMRSimulationController::problemSetup(const ProblemSpecP& params,
					   GridP& grid)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if(!grid_ps)
      return;
   
   Point anchor(MAXDOUBLE, MAXDOUBLE, MAXDOUBLE);
   int levelIndex=0;
   for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
       level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
      // Make two passes through the patches.  The first time, we
      // want to find the spacing and the lower left corner of the
      // problem domain.  Spacing can be specified with a dx,dy,dz
      // on the level, or with a resolution on the patch.  If a
      // resolution is used on a problem with more than one patch,
      // the resulting grid spacing must be consistent.
      Vector spacing;
      bool have_levelspacing=false;
      if(level_ps->get("spacing", spacing))
	 have_levelspacing=true;
      bool have_patchspacing=false;
	 
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
	  box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
	 Point lower;
	 box_ps->require("lower", lower);
	 Point upper;
	 box_ps->require("upper", upper);
	 if(levelIndex == 0)
	   anchor=Min(lower, anchor);

	 IntVector resolution;
	 if(box_ps->get("resolution", resolution)){
	    if(have_levelspacing){
	       throw ProblemSetupException("Cannot specify level spacing and patch resolution");
	    } else {
	       Vector newspacing = (upper - lower)/resolution;
	       if(have_patchspacing){
		  Vector diff = spacing - newspacing;
		  if(diff.length() > 1.e-6)
		     throw ProblemSetupException("Using patch resolution, and the patch spacings are inconsistent");
	       } else {
		  spacing = newspacing;
	       }
	       have_patchspacing=true;
	    }
	 }
      }
	 
      if(!have_levelspacing && !have_patchspacing)
	 throw ProblemSetupException("Box resolution is not specified");

      LevelP level = grid->addLevel(anchor, spacing);
      
      for(ProblemSpecP box_ps = level_ps->findBlock("Box");
	  box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
	Point lower;
	box_ps->require("lower", lower);
	Point upper;
	box_ps->require("upper", upper);
	
	IntVector lowCell = level->getCellIndex(lower);
	IntVector highCell = level->getCellIndex(upper+Vector(1.e-6,1.e-6,1.e-6));
	Point lower2 = level->getNodePosition(lowCell);
	Point upper2 = level->getNodePosition(highCell);
	double diff_lower = (lower2-lower).length();
	if(diff_lower > 1.e-6)
	  throw ProblemSetupException("Box lower corner does not coincide with grid");
	double diff_upper = (upper2-upper).length();
	if(diff_upper > 1.e-6){
	  cerr << "upper=" << upper << '\n';
	  cerr << "lowCell =" << lowCell << '\n';
	  cerr << "highCell =" << highCell << '\n';
	  cerr << "upper2=" << upper2 << '\n';
	  cerr << "diff=" << diff_upper << '\n';
	  throw ProblemSetupException("Box upper corner does not coincide with grid");
	}
	// Determine the interior cell limits.  For no extraCells, the limits
	// will be the same.  For extraCells, the interior cells will have
	// different limits so that we can develop a CellIterator that will
	// use only the interior cells instead of including the extraCell
	// limits.
	IntVector extraCells(0,0,0);
	if(box_ps->get("extraCells", extraCells)){
	  lowCell -= extraCells;
	  highCell += extraCells;
	}
	level->setExtraCells(extraCells);
	
	IntVector resolution(highCell-lowCell);
	if(resolution.x() < 1 || resolution.y() < 1 || resolution.z() < 1)
	  throw ProblemSetupException("Degenerate patch");
	
	IntVector patches;
	if(box_ps->get("patches", patches)){
	  level->setPatchDistributionHint(patches);
#if 0
	  if (d_myworld->size() > 1 &&
	      (patches.x() * patches.y() * patches.z() < d_myworld->size()))
	    throw ProblemSetupException("Number of patches must >= the number of processes in an mpi run");
#endif
	  for(int i=0;i<patches.x();i++){
	    for(int j=0;j<patches.y();j++){
	      for(int k=0;k<patches.z();k++){
		IntVector startcell = resolution*IntVector(i,j,k)/patches+lowCell;
		IntVector endcell = resolution*IntVector(i+1,j+1,k+1)/patches+lowCell;
		IntVector inStartCell = startcell;
		IntVector inEndCell = endcell;
		if(i==0)
		  inStartCell+=IntVector(extraCells.x(), 0, 0);
		if(i==patches.x()-1)
		  inEndCell-=IntVector(extraCells.x(), 0, 0);
		if(j==0)
		  inStartCell+=IntVector(0, extraCells.y(), 0);
		if(j==patches.y()-1)
		  inEndCell-=IntVector(0, extraCells.y(), 0);
		if(k==0)
		  inStartCell+=IntVector(0, 0, extraCells.z());
		if(k==patches.z()-1)
		  inEndCell-=IntVector(0, 0, extraCells.z());
		Patch* p = level->addPatch(startcell, endcell,
					   inStartCell, inEndCell);
		p->setLayoutHint(IntVector(i,j,k));
	      }
	    }
	  }
	} else {
	  level->addPatch(lowCell, highCell, lowCell+extraCells, highCell-extraCells);
	}
      }

      IntVector periodicBoundaries;
      if(level_ps->get("periodic", periodicBoundaries)){
	level->finalizeLevel(periodicBoundaries.x() != 0,
			     periodicBoundaries.y() != 0,
			     periodicBoundaries.z() != 0);
      }
      else {
	level->finalizeLevel();
      }
      level->assignBCS(grid_ps);
      levelIndex++;
   }
}

bool
AMRSimulationController::needRecompile(double time, double delt,
				       const GridP& grid,
				       SimulationInterface* sim,
				       Output* output,
				       LoadBalancer* lb,
				       vector<int>& levelids)
{
  // Currently, output and sim can request a recompile.  --bryan
  if(output && output->needRecompile(time, delt, grid))
    return true;
  if (sim && sim->needRecompile(time, delt, grid))
    return true;
  if (lb && lb->needRecompile(time, delt, grid))
    return true;
  if(static_cast<int>(levelids.size()) != grid->numLevels())
    return true;
  for(int i=0;i<grid->numLevels();i++){
    if(grid->getLevel(i)->getID() != levelids[i])
      return true;
  }
  return false;
}

void
AMRSimulationController::initializeErrorEstimate(const ProcessorGroup*,
						 const PatchSubset* patches,
						 const MaterialSubset* matls,
						 DataWarehouse*,
						 DataWarehouse* new_dw,
						 SimulationStateP sharedState)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m = 0;m<matls->size();m++){
      int matl = matls->get(m);

      CCVariable<int> refineFlag;
      new_dw->allocateAndPut(refineFlag, sharedState->get_refineFlag_label(),
			     matl, patch);
      refineFlag.initialize(false);
    }
  }
}
