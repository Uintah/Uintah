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
   problemSetup(ups, grid);
   

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
      output->finalizeTimestep(t, 0, grid, scheduler);

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
         output->finalizeTimestep(t, delt, grid, scheduler);
      
       // Begin next time step...
       sim->scheduleComputeStableTimestep(level, scheduler);
       scheduler->compile(d_myworld);

       double dt=Time::currentSeconds()-start;
       if(d_myworld->myrank() == 0)
         cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
      }
      // Execute the current timestep
      scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubComplete);
      scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNonPermanent);
      //scheduler->get_dw(0)->setScrubbing(DataWarehouse::ScrubNone);
      //scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
      scheduler->execute(d_myworld);
      if(output)
       output->executedTimestep();

      t += delt;
   }
   TAU_DB_DUMP();
   ups->releaseDocument();
}

void 
SimpleSimulationController::problemSetup(const ProblemSpecP& params,
                                    GridP& grid)
{
   ProblemSpecP grid_ps = params->findBlock("Grid");
   if(!grid_ps)
      return;
   
   for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
       level_ps != 0; level_ps = level_ps->findNextBlock("Level")){
      // Make two passes through the boxes.  The first time, we
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
        

      // first pass - find upper/lower corner, find resolution/spacing
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

                   // all boxes on same level must have same spacing
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

      LevelP level = grid->addLevel(anchor, spacing);
      

      // second pass - set up patches and cells
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
       IntVector extraCells;
       box_ps->getWithDefault("extraCells", extraCells, IntVector(0,0,0));
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
              IntVector inStartCell(startcell);
              IntVector inEndCell(endcell);
              startcell -= IntVector(i == 0? extraCells.x():0,
                                   j == 0? extraCells.y():0,
                                   k == 0? extraCells.z():0);
              endcell += IntVector(i == patches.x()-1? extraCells.x():0,
                                 j == patches.y()-1? extraCells.y():0,
                                 k == patches.z()-1? extraCells.z():0);
              Patch* p = level->addPatch(startcell, endcell,
                                      inStartCell, inEndCell);
              p->setLayoutHint(IntVector(i,j,k));
             }
           }
         }
       } else {
         // actually create the patch
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
   }
} // end problemSetup()

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
