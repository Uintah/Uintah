#include <Packages/Uintah/CCA/Components/SimulationController/SimpleSimulationController.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
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
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <TauProfilerForSCIRun.h>
#include <iostream>
#include <iomanip>
#include <values.h>

#include <list>
#include <fstream>
#include <math.h>

#include <Core/Malloc/Allocator.h> // for memory leak tests...

using std::cerr;
using std::cout;

using namespace SCIRun;
using namespace Uintah;

static DebugStream dbg("SimpleSimCont", false);

SimpleSimulationController::SimpleSimulationController(const ProcessorGroup* myworld) :
  SimulationController(myworld)
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

// use better stdDeviation formula below
//double stdDeviation(list<double>& vals, double& mean)
//{
//  if (vals.size() < 2)
//    return -1;

//  list<double>::iterator it;

//  mean = 0;
//  double variance = 0;
//  for (it = vals.begin(); it != vals.end(); it++)
//    mean += *it;
//  mean /= vals.size();

//  for (it = vals.begin(); it != vals.end(); it++)
//    variance += pow(*it - mean, 2);
//  variance /= (vals.size() - 1);
//  return sqrt(variance);
//}

double stdDeviation(double sum_of_x, double sum_of_x_squares, int n)
{
  // better formula for stdDev than above - less memory and quicker.
  return sqrt((n*sum_of_x_squares - sum_of_x*sum_of_x)/(n*n));
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

   Output* output = dynamic_cast<Output*>(getPort("output"));

   if( !output ){
     cout << "dynamic_cast of 'output' failed!\n";
     throw InternalError("dynamic_cast of 'output' failed!");
   }
   output->problemSetup(ups);
   
   // Setup the initial grid
   GridP grid=scinew Grid();
   grid->problemSetup(ups, d_myworld);
   

   if(grid->numLevels() == 0){
      cerr << "No problem specified.  Exiting SimpleSimulationController.\n";
      return;
   }
   
   // Check the grid
   grid->performConsistencyCheck();
   // Print out meta data
   if (d_myworld->myrank() == 0)
     grid->printStatistics();

   SimulationStateP sharedState = scinew SimulationState(ups);
   
   // Initialize the CFD and/or MPM components
   SimulationInterface* sim = 
     dynamic_cast<SimulationInterface*>(getPort("sim"));

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

   // done after the sim->problemSetup to get defaults into the
   // input.xml, which it writes along with index.xml
   output->initializeOutput(ups);

   if(d_myworld->myrank() == 0)
     cout << "Compiling taskgraph...\n";

   double start = Time::currentSeconds();
   scheduler->initialize();
   scheduler->advanceDataWarehouse(grid);

   double t;

   // Parse time struct: get info like max time, init time, num timesteps, etc.
   SimulationTime timeinfo(ups);

   if (d_combinePatches) {
     Dir combineFromDir(d_fromDir);
     output->combinePatchSetup(combineFromDir);

     // somewhat of a hack, but the patch combiner specifies exact delt's
     // and should not use a delt factor.
     timeinfo.delt_factor = 1;
     timeinfo.delt_min = 0;
     timeinfo.delt_max = timeinfo.maxTime;
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
   
   double start_time = Time::currentSeconds();
   if (!d_restarting){
     t = timeinfo.initTime;
     sim->scheduleComputeStableTimestep(level,scheduler);
   }
   
   if(output)
      output->finalizeTimestep(t, 0, grid, scheduler, true);

   scheduler->compile(d_myworld);
   
   double dt=Time::currentSeconds()-start;
   if(d_myworld->myrank() == 0)
     cout << "done taskgraph compile (" << dt << " seconds)\n";
   scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
   scheduler->execute(d_myworld);

   if(output)
     output->executedTimestep();

   // vars used to calculate standard deviation
   int n = 0;
   double prevWallTime = Time::currentSeconds();
   double sum_of_walltime = 0; // sum of all walltimes
   double sum_of_walltime_squares = 0; // sum all squares of walltimes
   
   bool first=true;
   int  iterations = 0;
   double prev_delt=0;
   while( ( t < timeinfo.maxTime ) && 
         ( iterations < timeinfo.num_time_steps ) ) {
      iterations ++;

      double wallTime = Time::currentSeconds() - start_time;

      delt_vartype delt_var;
      DataWarehouse* newDW = scheduler->get_dw(1);
      newDW->get(delt_var, sharedState->get_delt_label());

      double delt = delt_var;
      delt *= timeinfo.delt_factor;
      
      // Bind delt to the min and max read from the ups file
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
       
      newDW->override(delt_vartype(delt), sharedState->get_delt_label());
      double delt_fine = delt;
      for(int i=0;i<grid->numLevels();i++){
       const Level* level = grid->getLevel(i).get_rep();
       if(i != 0)
         delt_fine /= level->timeRefinementRatio();
       newDW->override(delt_vartype(delt), sharedState->get_delt_label(),
                     level);
      }
     prev_delt=delt;

     // get memory stats for output
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
       // Remember, this isn't logged if DISABLE_SCI_MALLOC is set
       // (So usually in optimized mode this will not be run.)
       scheduler->logMemoryUse();
       ostringstream fn;
       fn << "alloc." << setw(5) << setfill('0') 
          << d_myworld->myrank() << ".out";
       string filename(fn.str());
       DumpAllocator(DefaultAllocator(), filename.c_str());
      }

      // output timestep statistics
      if(d_myworld->myrank() == 0){
//       cout << "Current Top Level Time Step: " 
//            << sharedState->getCurrentTopLevelTimeStep() << "\n";

       cout << "Time=" << t << ", delT=" << delt 
            << ", elap T = " << wallTime 
            << ", DW: " << newDW->getID() << ", Mem Use = ";
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
         //         ofstream timefile("avg_elapsed_walltime.txt");
         //         timefile << mean << " +- " << stdDev << endl;
         cout << "Timestep mean: " << mean << " +- " << stdDev << endl;
       }
       prevWallTime = wallTime;
       n++;
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
          output->finalizeTimestep(t, delt, grid, scheduler, true);
        
        // Begin next time step...
        sim->scheduleComputeStableTimestep(level, scheduler);
        scheduler->compile(d_myworld);
        
        double dt=Time::currentSeconds()-start;
        if(d_myworld->myrank() == 0)
          cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
      }
      else {
        if (output)
          output->finalizeTimestep(t, delt, grid, scheduler, false);
      }

      // Execute the current timestep.  If the timestep needs to be
      // restarted, this loop will execute multiple times.
      bool success = true;
      do {
	bool restartable = sim->restartableTimesteps();
	if (restartable)
	  scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNone);
	else
	  scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubComplete);
	
	scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNonPermanent);

	scheduler->execute(d_myworld);

	if(scheduler->get_dw(1)->timestepRestarted()){
	  ASSERT(restartable);
	  // Figure out new delt
	  double new_delt = sim->recomputeTimestep(delt);
	  if(d_myworld->myrank() == 0)
	    cerr << "Restarting timestep at " << t << ", changing delt from " 
		 << delt << " to " << new_delt << '\n';
	  delt = new_delt;
	  scheduler->get_dw(0)->override(delt_vartype(delt), 
					 sharedState->get_delt_label());
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
	output->executedTimestep();
      }

      t += delt;
   }
   TAU_DB_DUMP();
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
