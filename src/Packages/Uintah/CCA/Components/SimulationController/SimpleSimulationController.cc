#include <sci_defs/malloc_defs.h>

#include <Packages/Uintah/CCA/Components/SimulationController/SimpleSimulationController.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/Array3.h>
#include <Core/Thread/Time.h>
#include <Core/OS/Dir.h>
#include <Core/OS/ProcessInfo.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/CCA/Components/Switcher/Switcher.h>
#include <TauProfilerForSCIRun.h>
#include <iostream>
#include <iomanip>

//#include <sys/param.h>
//#include <list>
//#include <fstream>
//#include <math.h>

#include <Core/Malloc/Allocator.h> // for memory leak tests...

using std::cerr;
using std::cout;

using namespace SCIRun;
using namespace Uintah;

static DebugStream dbg("SimpleSimCont", false);

SimpleSimulationController::SimpleSimulationController(const ProcessorGroup* myworld) :
  SimulationController(myworld, false)
{
   d_restarting = false;
   d_combinePatches = false;
}

SimpleSimulationController::~SimpleSimulationController()
{
}

void
SimpleSimulationController::doRestart(std::string restartFromDir, int timestep,
                                  bool fromScratch, bool removeOldDir)
{
   d_restarting = true;
   d_fromDir = restartFromDir;
   d_restartTimestep = timestep;
   d_restartFromScratch = fromScratch;
   d_restartRemoveOldDir = removeOldDir;
}

void SimpleSimulationController::doCombinePatches(std::string fromDir)
{
   d_combinePatches = true;
   d_fromDir = fromDir;
}

void 
SimpleSimulationController::run()
{
   UintahParallelPort* pp = getPort("problem spec");
   ProblemSpecInterface* psi = dynamic_cast<ProblemSpecInterface*>(pp);
   
   if( !psi ){
     cout << "SimpleSimulationController::run() psi dynamic_cast failed...\n";
     throw InternalError("psi dynamic_cast failed");
   }

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

   SimulationStateP sharedState = scinew SimulationState(ups);
   
   Output* output = dynamic_cast<Output*>(getPort("output"));

   if( !output ){
     cout << "dynamic_cast of 'output' failed!\n";
     throw InternalError("dynamic_cast of 'output' failed!");
   }
   output->problemSetup(ups, sharedState.get_rep());
   
   // Setup the initial grid
   GridP grid=scinew Grid();
   grid->problemSetup(ups, d_myworld, false);
   

   if(grid->numLevels() == 0){
      cerr << "No problem specified.  Exiting SimpleSimulationController.\n";
      return;
   }
   
   // Check the grid
   grid->performConsistencyCheck();
   // Print out meta data
   if (d_myworld->myrank() == 0)
     grid->printStatistics();

   Switcher* switcher = dynamic_cast<Switcher*>(getPort("switcher"));
   if (!switcher)
     throw InternalError("No switcher component");

   // Initialize the CFD and/or MPM components
   SimulationInterface* sim = 
     dynamic_cast<SimulationInterface*>(switcher->getPort("sim"));

   if(!sim)
     throw InternalError("No simulation component");
   sim->problemSetup(ups, grid, sharedState);

   // Finalize the shared state/materials
   sharedState->finalizeMaterials();

   Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
   sched->problemSetup(ups);
   SchedulerP scheduler(sched);
   
   LoadBalancer* lb = sched->getLoadBalancer();
   lb->problemSetup(ups, sharedState);

   // done after the sim->problemSetup to get defaults into the
   // input.xml, which it writes along with index.xml
   output->initializeOutput(ups);

   if(d_myworld->myrank() == 0)
     cout << "Compiling taskgraph...\n";

   calcStartTime();

   scheduler->initialize();
   scheduler->advanceDataWarehouse(grid);
   // for dynamic lb's, set up initial patch config
   lb->possiblyDynamicallyReallocate(grid, false); 

   double t;

   // Parse time struct: get info like max time, init time, num timesteps, etc.
   d_timeinfo = new SimulationTime(ups);

   if (d_combinePatches) {
     Dir combineFromDir(d_fromDir);
     output->combinePatchSetup(combineFromDir);

     // somewhat of a hack, but the patch combiner specifies exact delt's
     // and should not use a delt factor.
     d_timeinfo->delt_factor = 1;
     d_timeinfo->delt_min = 0;
     d_timeinfo->maxTime = static_cast<PatchCombiner*>(sim)->getMaxTime();
     cout << " MaxTime: " << d_timeinfo->maxTime << endl;
     d_timeinfo->delt_max = d_timeinfo->maxTime;
   }   
   
   if(d_restarting){

      dbg << "Restarting... loading data\n";

      // create a temporary DataArchive for reading in the checkpoints
      // archive for restarting.
      Dir restartFromDir(d_fromDir);
      Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
      DataArchive archive(checkpointRestartDir.getName(),
                       d_myworld->myrank(), d_myworld->size());

      double delt = 0;
      archive.restartInitialize(d_restartTimestep, grid,
                            scheduler->get_dw(1), lb, &t, &delt);
      
      sharedState->setCurrentTopLevelTimeStep( d_restartTimestep );
      ProblemSpecP pspec = archive.getRestartTimestepDoc();
      XMLURL url = archive.getRestartTimestepURL();
      //lb->restartInitialize(pspec, url);
      output->restartSetup(restartFromDir, 0, d_restartTimestep, t,
                        d_restartFromScratch, d_restartRemoveOldDir);

      // Tell the scheduler the generation of the re-started simulation.
      scheduler->setGeneration( output->getCurrentTimestep()+1);

      scheduler->get_dw(1)->setID( output->getCurrentTimestep() );
      
      // just in case you want to change the delt on a restart....
      if (d_timeinfo->override_restart_delt != 0) {
        double newdelt = d_timeinfo->override_restart_delt;
        if (d_myworld->myrank() == 0)
          cout << "Overriding restart delt with " << newdelt << endl;
        scheduler->get_dw(1)->override(delt_vartype(newdelt), 
                                       sharedState->get_delt_label());
        double delt_fine = newdelt;
        for(int i=0;i<grid->numLevels();i++){
          const Level* level = grid->getLevel(i).get_rep();
          if(i != 0)
            delt_fine /= level->timeRefinementRatio();
          scheduler->get_dw(1)->override(delt_vartype(delt_fine), sharedState->get_delt_label(),
                                         level);
        }

      }
      scheduler->get_dw(1)->finalize();
      sim->restartInitialize();
   } else {

      dbg << "Setting up initial tasks\n";

      sharedState->setCurrentTopLevelTimeStep( 0 );
      // Initialize the CFD and/or MPM data
      for(int i=0;i<grid->numLevels();i++){
        LevelP level = grid->getLevel(i);
        dbg << "calling scheduleInitialize: \n";
        sim->scheduleInitialize(level, scheduler);
      }
   }
   
   // For AMR, this will need to change
   if(grid->numLevels() != 1)
      throw ProblemSetupException("AMR problem specified; cannot do it yet");
   LevelP level = grid->getLevel(0);
   
   // Parse time struct
   /* SimulationTime timeinfo(ups); - done earlier */
   
   //   calcStartTime();  // DONE EARLIER

   if (!d_restarting){
     t = d_timeinfo->initTime;
     sim->scheduleComputeStableTimestep(level,scheduler);
   }

   setStartSimTime(t);

   if(output)
      output->finalizeTimestep(t, 0, grid, scheduler, 1);

   scheduler->compile();
   
   double dt=Time::currentSeconds() - getStartTime();
   if(d_myworld->myrank() == 0)
     cout << "done taskgraph compile (" << dt << " seconds)\n";
   scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
   scheduler->execute();

   if(output)
     output->executedTimestep(0,grid);

   initSimulationStatsVars();

   bool first=true;
   int  iterations = 0;
   double delt = 0;
   
   // if we end the simulation for a timestep, decide whether to march max_iterations
   // or to end at a certain timestep
   int max_iterations = d_timeinfo->max_iterations;
   if (d_timeinfo->maxTimestep - sharedState->getCurrentTopLevelTimeStep() < max_iterations)
     max_iterations = d_timeinfo->maxTimestep - sharedState->getCurrentTopLevelTimeStep();

   while( t < d_timeinfo->maxTime && iterations < max_iterations) {
      iterations ++;

      calcWallTime();

      delt_vartype delt_var;
      DataWarehouse* newDW = scheduler->get_dw(1);
      newDW->get(delt_var, sharedState->get_delt_label());

      double prev_delt = delt;
      delt = delt_var;
      delt *= d_timeinfo->delt_factor;
      
      // Bind delt to the min and max read from the ups file
      if(delt < d_timeinfo->delt_min){
        if(d_myworld->myrank() == 0)
           cerr << "WARNING: raising delt from " << delt
               << " to minimum: " << d_timeinfo->delt_min << '\n';
        delt = d_timeinfo->delt_min;
      }
      if(iterations > 1 && d_timeinfo->max_delt_increase < 1.e90
        && delt > (1+d_timeinfo->max_delt_increase)*prev_delt){
       if(d_myworld->myrank() == 0)
         cerr << "WARNING: lowering delt from " << delt 
              << " to maxmimum: " << (1+d_timeinfo->max_delt_increase)*prev_delt
              << " (maximum increase of " << d_timeinfo->max_delt_increase
              << ")\n";
       delt = (1+d_timeinfo->max_delt_increase)*prev_delt;
      }
      if(t <= d_timeinfo->initial_delt_range && delt > d_timeinfo->max_initial_delt){
        if(d_myworld->myrank() == 0)
           cerr << "WARNING: lowering delt from " << delt 
               << " to maximum: " << d_timeinfo->max_initial_delt
               << " (for initial timesteps)\n";
        delt = d_timeinfo->max_initial_delt;
      }
      if(delt > d_timeinfo->delt_max){
        if(d_myworld->myrank() == 0)
           cerr << "WARNING: lowering delt from " << delt 
               << " to maximum: " << d_timeinfo->delt_max << '\n';
        delt = d_timeinfo->delt_max;
      }

      // clamp timestep to output/checkpoint
      if (d_timeinfo->timestep_clamping && output) {
        double orig_delt = delt;
        double nextOutput = output->getNextOutputTime();
        double nextCheckpoint = output->getNextCheckpointTime();
        if (nextOutput != 0 && t + delt > nextOutput) {
          delt = nextOutput - t;       
        }
        if (nextCheckpoint != 0 && t + delt > nextCheckpoint) {
          delt = nextCheckpoint - t;
        }
        if (delt != orig_delt) {
          if(d_myworld->myrank() == 0)
            cerr << "WARNING: lowering delt from " << orig_delt 
                 << " to " << delt
                 << " to line up with output/checkpoint time\n";
        }
      }

      newDW->override(delt_vartype(delt), sharedState->get_delt_label());
      double delt_fine = delt;
      for(int i=0;i<grid->numLevels();i++){
       const Level* level = grid->getLevel(i).get_rep();
       if(i != 0)
         delt_fine /= level->timeRefinementRatio();
       newDW->override(delt_vartype(delt_fine), sharedState->get_delt_label(),
                     level);
      }

      printSimulationStats( sharedState, delt, t );

      if(log_dw_mem){
	// Remember, this isn't logged if DISABLE_SCI_MALLOC is set
	// (So usually in optimized mode this will not be run.)
	scheduler->logMemoryUse();
	ostringstream fn;
	fn << "alloc." << setw(5) << setfill('0') << d_myworld->myrank() << ".out";
	string filename(fn.str());
	DumpAllocator(DefaultAllocator(), filename.c_str());
      }

      if(sharedState->needAddMaterial()!=0){
        sim->addMaterial(ups, grid, sharedState);
        sharedState->finalizeMaterials();
        scheduler->initialize();
        sim->scheduleInitializeAddedMaterial(level, scheduler);
        scheduler->compile();
        scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
        scheduler->execute();
      }

      scheduler->advanceDataWarehouse(grid);

      // Put the current time into the shared state so other components
      // can access it.  Also increment (by one) the current time step
      // number so components can tell what timestep they are on.
      sharedState->setElapsedTime(t);
      sharedState->incrementCurrentTopLevelTimeStep();

      if(needRecompile(t, delt, grid, sim, output, lb) || first){
        first=false;
        if(d_myworld->myrank() == 0)
          cout << "COMPILING TASKGRAPH...\n";
        double start = Time::currentSeconds();
        scheduler->initialize();

        sim->scheduleTimeAdvance(level, scheduler, 0, 1);
        
        if(output)
          output->finalizeTimestep(t, delt, grid, scheduler, true, sharedState->needAddMaterial());
        
        // Begin next time step...
        sim->scheduleComputeStableTimestep(level, scheduler);
        scheduler->compile();
        
        double dt=Time::currentSeconds() - start;
        if(d_myworld->myrank() == 0)
          cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";

        sharedState->setNeedAddMaterial(0);
      }
      else {
        if (output)
          output->finalizeTimestep(t, delt, grid, scheduler, 0);
      }

      // Execute the current timestep.  If the timestep needs to be
      // restarted, this loop will execute multiple times.
      bool success = true;
      double orig_delt = delt;
      do {
	bool restartable = sim->restartableTimesteps();
        scheduler->setRestartable(restartable);
	if (restartable || lb->isDynamic() || lb->getNthProc() != 1)
	  scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNone);
	else
          scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubComplete);
	  	
	scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNonPermanent);

	scheduler->execute();

	if(scheduler->get_dw(1)->timestepRestarted()){
	  ASSERT(restartable);
	  // Figure out new delt
	  double new_delt = sim->recomputeTimestep(delt);
	  if(d_myworld->myrank() == 0)
	    cerr << "Restarting timestep at " << t << ", changing delt from " 
		 << delt << " to " << new_delt << '\n';
          output->reEvaluateOutputTimestep(orig_delt, new_delt);
	  delt = new_delt;
	  scheduler->get_dw(0)->override(delt_vartype(new_delt), 
					 sharedState->get_delt_label());
          double delt_fine = delt;
          for(int i=0;i<grid->numLevels();i++){
            const Level* level = grid->getLevel(i).get_rep();
            if(i != 0)
              delt_fine /= level->timeRefinementRatio();
            scheduler->get_dw(0)->override(delt_vartype(delt_fine),
                                           sharedState->get_delt_label(),
                                           level);
          }
	  success = false;
	  
	  scheduler->replaceDataWarehouse(1, grid);
	} else {
	  success = true;
	  if(scheduler->get_dw(1)->timestepAborted()){
	    throw InternalError("Execution aborted, cannot restart timestep\n");
	  }
	}
      } while(!success);
      if(output) {
	output->executedTimestep(delt,grid);
      }

      t += delt;
      TAU_DB_DUMP();
   }
   
   ups->releaseDocument();
}


bool
SimpleSimulationController::needRecompile(double time, double delt,
                                     const GridP& grid,
                                     SimulationInterface* sim,
                                     Output* output,
                                     LoadBalancer* lb)
{
  // Currently, output, sim, and load balancer can request a recompile. --bryan
  
  // It is done in this fashion to give everybody a shot at modifying their 
  // state.  Some get left out if you say 
  // 'return lb->needRecompile() || output->needRecompile()'

  bool recompile = (lb && lb->needRecompile(time, delt, grid));
  recompile = (output && output->needRecompile(time, delt, grid)) || recompile;
  recompile =  (sim && sim->needRecompile(time, delt, grid)) || recompile;
  return recompile;
}
