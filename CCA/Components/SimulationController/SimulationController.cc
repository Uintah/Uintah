
#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>
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
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Ports/CFDInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>
#include <Packages/Uintah/CCA/Ports/MPMCFDInterface.h>
#include <Packages/Uintah/CCA/Ports/MDInterface.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/Analyze.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

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

// for calculating memory usage when sci-malloc is disabled.
char* SimulationController::start_addr = NULL;

SimulationController::SimulationController(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
   d_restarting = false;
}

SimulationController::~SimulationController()
{
}

void SimulationController::doRestart(std::string restartFromDir, int timestep,
				     bool fromScratch, bool removeOldDir)
{
   d_restarting = true;
   d_restartFromDir = restartFromDir;
   d_restartTimestep = timestep;
   d_restartFromScratch = fromScratch;
   d_restartRemoveOldDir = removeOldDir;
}


#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
double stdDeviation(list<double>& vals, double& mean)
{
  if (vals.size() < 2)
    return -1;

  list<double>::iterator it;

  mean = 0;
  double variance = 0;
  for (it = vals.begin(); it != vals.end(); it++)
    mean += *it;
  mean /= vals.size();

  for (it = vals.begin(); it != vals.end(); it++)
    variance += pow(*it - mean, 2);
  variance /= (vals.size() - 1);
  return sqrt(variance);
}
#endif

void SimulationController::run()
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
   ProblemSpecP debug = ups->findBlock("debug");
   if(debug){
     ProblemSpecP log_mem = debug->findBlock("logmemory");
     if(log_mem)
       log_dw_mem=true;
   }
   
   Output* output = dynamic_cast<Output*>(getPort("output"));
   output->problemSetup(ups);
   
   // Setup the initial grid
   GridP grid=scinew Grid();

   problemSetup(ups, grid);
   
   if(grid->numLevels() == 0){
      cerr << "No problem specified.  Exiting SimulationController.\n";
      return;
   }
   
   // Check the grid
   grid->performConsistencyCheck();
   // Print out meta data
   if (d_myworld->myrank() == 0)
     grid->printStatistics();

   SimulationStateP sharedState = scinew SimulationState(ups);
   
   // Initialize the CFD and/or MPM components
   CFDInterface* cfd       = dynamic_cast<CFDInterface*>(getPort("cfd"));
   MPMInterface* mpm       = dynamic_cast<MPMInterface*>(getPort("mpm"));
   MPMCFDInterface* mpmcfd = dynamic_cast<MPMCFDInterface*>(getPort("mpmcfd"));
   if(cfd && !mpmcfd)
      cfd->problemSetup(ups, grid, sharedState);
   
   if(mpm && !mpmcfd)
      mpm->problemSetup(ups, grid, sharedState);

   if(mpmcfd) {
      sharedState->d_mpm_cfd=true;
      mpmcfd->problemSetup(ups, grid, sharedState);
   }

   // Finalize the shared state/materials
   sharedState->finalizeMaterials();

   // Initialize the MD components --tan
   MDInterface* md = dynamic_cast<MDInterface*>(getPort("md"));
   if(md)
      md->problemSetup(ups, grid, sharedState);
   
   Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
   sched->problemSetup(ups);
   SchedulerP scheduler(sched);

   if(d_myworld->myrank() == 0){
     cout << "Compiling taskgraph...";
     cout.flush();
   }
   double start = Time::currentSeconds();
   scheduler->advanceDataWarehouse(grid);
   
   scheduler->initialize();

   double t;

   // Parse time struct
   SimulationTime timeinfo(ups);

   if(d_restarting){
      // create a temporary DataArchive for reading in the checkpoints
      // archive for restarting.
      Dir restartFromDir(d_restartFromDir);
      Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
      DataArchive archive(checkpointRestartDir.getName(),
			  d_myworld->myrank(), d_myworld->size());

      double delt = 0;
      archive.restartInitialize(d_restartTimestep, grid,
				scheduler->get_new_dw(), &t, &delt);
      
      output->restartSetup(restartFromDir, 0, d_restartTimestep, t,
			   d_restartFromScratch, d_restartRemoveOldDir);
   } else {
      // Initialize the CFD and/or MPM data
      for(int i=0;i<grid->numLevels();i++){
	 LevelP level = grid->getLevel(i);
	 scheduleInitialize(level, scheduler, cfd, mpm, mpmcfd, md);
      }
   }
   
   // For AMR, this will need to change
   if(grid->numLevels() != 1)
      throw ProblemSetupException("AMR problem specified; cannot do it yet");
   LevelP level = grid->getLevel(0);
   
   // Parse time struct
   /* SimulationTime timeinfo(ups); */
   
   double start_time = Time::currentSeconds();
   if (!d_restarting){
     t = timeinfo.initTime;
     scheduleComputeStableTimestep(level,scheduler, cfd, mpm, mpmcfd, md);
   }
   
   Analyze* analyze = dynamic_cast<Analyze*>(getPort("analyze"));
   if(analyze)
      analyze->problemSetup(ups, grid, sharedState);
   
   if(output)
      output->finalizeTimestep(t, 0, level, scheduler);

   scheduler->compile(d_myworld, true);
   
   double dt=Time::currentSeconds()-start;
   if(d_myworld->myrank() == 0)
     cout << "done taskgraph compile (" << dt << " seconds)\n";
   scheduler->execute(d_myworld);

#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
   int n = 0;
   list<double> wallTimes;
   double prevWallTime;
#endif

   bool first=true;
   while(t < timeinfo.maxTime) {
      double wallTime = Time::currentSeconds() - start_time;

      delt_vartype delt_var;
      scheduler->get_new_dw()->get(delt_var, sharedState->get_delt_label());

      double delt = delt_var;
      delt *= timeinfo.delt_factor;
      
      if(delt < timeinfo.delt_min){
	 if(d_myworld->myrank() == 0)
	    cerr << "WARNING: raising delt from " << delt
		 << " to minimum: " << timeinfo.delt_min << '\n';
	 delt = timeinfo.delt_min;
      }
      if(delt > timeinfo.delt_max){
	 if(d_myworld->myrank() == 0)
	    cerr << "WARNING: lowering delt from " << delt 
		 << " to maxmimum: " << timeinfo.delt_max << '\n';
	 delt = timeinfo.delt_max;
      }
      scheduler->get_new_dw()->override(delt_vartype(delt),
					sharedState->get_delt_label());

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

      if(d_myworld->myrank() == 0){
	if( analyze ) analyze->showStepInformation();
	cout << "Time=" << t << ", delT=" << delt 
	     << ", elap T = " << wallTime 
	     << ", DW: " << scheduler->get_new_dw()->getID() << ", Mem Use = ";
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

#ifdef OUTPUT_AVG_ELAPSED_WALLTIME
	if (n > 1) // ignore first set of elapsed times
	  wallTimes.push_back(wallTime - prevWallTime);

	if (wallTimes.size() > 1) {
	  double stdDev, mean;
	  stdDev = stdDeviation(wallTimes, mean);
	  ofstream timefile("avg_elapsed_walltime.txt");
	  timefile << mean << " +- " << stdDev << endl;
	}
	prevWallTime = wallTime;
	n++;
#endif
      }
      scheduler->advanceDataWarehouse(grid);
      // put the current time into the shared state so other components
      // can access it
      sharedState->setElapsedTime(t);
      if(need_recompile(t, delt, level, cfd, mpm, mpmcfd, md, output) || first){
	first=false;
	if(d_myworld->myrank() == 0)
	  cout << "Compiling taskgraph...";
	double start = Time::currentSeconds();
	scheduler->initialize();

	scheduleTimeAdvance(t, delt, level, scheduler,
			    cfd, mpm, mpmcfd, md);

	//data analyze in each step
	if(analyze) {
	  analyze->performAnalyze(t, delt, level, scheduler);
	}
      
	if(output)
	  output->finalizeTimestep(t, delt, level, scheduler);
      
	// Begin next time step...
	scheduleComputeStableTimestep(level, scheduler, cfd, mpm, mpmcfd, md);
	scheduler->compile(d_myworld, false);

	double dt=Time::currentSeconds()-start;
	if(d_myworld->myrank() == 0)
	  cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
      }
      // Execute the current timestep
      scheduler->execute(d_myworld);
      t += delt;
   }
}

void SimulationController::problemSetup(const ProblemSpecP& params,
					GridP& grid)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if(!grid_ps)
      return;
   
   for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
       level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
      // Make two passes through the patches.  The first time, we
      // want to find the spacing and the lower left corner of the
      // problem domain.  Spacing can be specified with a dx,dy,dz
      // on the level, or with a resolution on the patch.  If a
      // resolution is used on a problem with more than one patch,
      // the resulting grid spacing must be consistent.
      Point anchor(MAXDOUBLE, MAXDOUBLE, MAXDOUBLE);
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
	 anchor=Min(lower, anchor);

	 IntVector resolution;
	 if(box_ps->get("resolution", resolution)){
	    if(have_levelspacing){
	       throw ProblemSetupException("Cannot specify level spacing and patch resolution");
	    } else {
	       Vector newspacing = (upper-lower)/resolution;
	       if(have_patchspacing){
		  Vector diff = spacing-newspacing;
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
	 
      LevelP level = scinew Level(grid.get_rep(), anchor, spacing);
      
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
	  if (d_myworld->size() > 1 &&
	      (patches.x() * patches.y() * patches.z() < d_myworld->size()))
	    throw ProblemSetupException("Number of patches must >= the number of processes in an mpi run");
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
      level->finalizeLevel();
      level->assignBCS(grid_ps);
      grid->addLevel(level);
   }
}

void SimulationController::scheduleInitialize(LevelP& level,
					      SchedulerP& sched,
					      CFDInterface* cfd,
					      MPMInterface* mpm,
					      MPMCFDInterface* mpmcfd,
					      MDInterface* md)
{
  if(mpmcfd){
    mpmcfd->scheduleInitialize(level, sched);
  }
  else {
    if(cfd) {
      cfd->scheduleInitialize(level, sched);
    }
    if(mpm) {
      mpm->scheduleInitialize(level, sched);
    }
  }
  if(md) {
    md->scheduleInitialize(level, sched);
  }
}

void SimulationController::scheduleComputeStableTimestep(LevelP& level,
							SchedulerP& sched,
							CFDInterface* cfd,
							MPMInterface* mpm,
							MPMCFDInterface* mpmcfd,
							MDInterface* md)
{
  if(mpmcfd){
    mpmcfd->scheduleComputeStableTimestep(level, sched);
  }
  else {
     if(cfd)
        cfd->scheduleComputeStableTimestep(level, sched);
     if(mpm)
        mpm->scheduleComputeStableTimestep(level, sched);
   }
   if(md)
      md->scheduleComputeStableTimestep(level, sched);
}

void SimulationController::scheduleTimeAdvance(double t, double delt,
					       LevelP& level,
					       SchedulerP& sched,
					       CFDInterface* cfd,
					       MPMInterface* mpm,
					       MPMCFDInterface* mpmcfd,
					       MDInterface* md)
{
  // Temporary - when cfd/mpm are coupled this will need help
  if(mpmcfd){
      mpmcfd->scheduleTimeAdvance(t, delt, level, sched);
  }
  else {
   if(cfd)
     cfd->scheduleTimeAdvance(t, delt, level, sched);
   if(mpm)
      mpm->scheduleTimeAdvance(t, delt, level, sched);
  }
      
   // Added molecular dynamics module, currently it will not be coupled with 
   // cfd/mpm.  --tan
   if(md)
      md->scheduleTimeAdvance(t, delt, level, sched);
   
}

bool
SimulationController::need_recompile(double time, double delt,
				     const LevelP& level, CFDInterface* cfd,
				     MPMInterface* mpm,
				     MPMCFDInterface* mpmcfd,
				     MDInterface* md,
				     Output* output)
{
  // Currently, nothing but output can request a recompile.  This
  // should be fixed - steve
  if(output && output->need_recompile(time, delt, level))
    return true;
  return false;
}
