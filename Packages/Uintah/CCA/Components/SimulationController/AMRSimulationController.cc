#include <sci_defs/malloc_defs.h>

#include <Packages/Uintah/CCA/Components/SimulationController/AMRSimulationController.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/Array3.h>
#include <Core/Thread/Time.h>
#include <Core/OS/Dir.h>
#include <Core/OS/ProcessInfo.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarLabelMatlLevel.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Regridder.h>
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
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <TauProfilerForSCIRun.h>
#include <iostream>
#include <iomanip>

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

   SimulationStateP sharedState = scinew SimulationState(ups);
   
   Output* output = dynamic_cast<Output*>(getPort("output"));
   output->problemSetup(ups, sharedState.get_rep());
   
   // Setup the initial grid
   GridP currentGrid=scinew Grid();
   GridP oldGrid = currentGrid;

   currentGrid->problemSetupAMR(ups, d_myworld);
   
   if(currentGrid->numLevels() == 0){
      cerr << "No problem specified.  Exiting AMRSimulationController.\n";
      return;
   }
   
   // Check the grid
   currentGrid->performConsistencyCheck();
   // Print out meta data
   if (d_myworld->myrank() == 0)
     currentGrid->printStatistics();

   // Initialize the CFD and/or MPM components
   SimulationInterface* sim = dynamic_cast<SimulationInterface*>(getPort("sim"));
   if(!sim)
     throw InternalError("No simulation component");
   sim->problemSetup(ups, currentGrid, sharedState);

   // Finalize the shared state/materials
   sharedState->finalizeMaterials();

   Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
   sched->problemSetup(ups);
   SchedulerP scheduler(sched);

   LoadBalancer* lb = dynamic_cast<LoadBalancer*>
     (dynamic_cast<SchedulerCommon*>(sched)->getPort("load balancer"));
   lb->problemSetup(ups, sharedState);

   output->initializeOutput(ups);

   // set up regridder with initial infor about grid
   Regridder* regridder = dynamic_cast<Regridder*>(getPort("regridder"));
   regridder->problemSetup(ups, currentGrid, sharedState);

   if(d_myworld->myrank() == 0){
     cout << "Compiling initialization taskgraph...\n";
   }

   calcStartTime();
   
   scheduler->initialize(1, 1);
   scheduler->advanceDataWarehouse(currentGrid);
   // for dynamic lb's, set up initial patch config
   lb->possiblyDynamicallyReallocate(currentGrid, false); 

   double t;

   // Parse time struct
   SimulationTime timeinfo(ups);

   // This has to change for AMR restarting, to support the readin
   // of grid hierarchies (and with that a scheduler hierarchie)
   // from where it left off.
   if(d_restarting){
     // create a temporary DataArchive for reading in the checkpoints
     // archive for restarting.
     Dir restartFromDir(d_restartFromDir);
     Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
     DataArchive archive(checkpointRestartDir.getName(),
			 d_myworld->myrank(), d_myworld->size());
     
     double delt = 0;
     archive.restartInitialize(d_restartTimestep, currentGrid,
			       scheduler->get_dw(1), &t, &delt);
     
     ProblemSpecP pspec = archive.getRestartTimestepDoc();
     XMLURL url = archive.getRestartTimestepURL();
     lb->restartInitialize(pspec, url);
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
     t = timeinfo.initTime;
     // Initialize the CFD and/or MPM data
     for(int i=0;i<currentGrid->numLevels();i++) {
       sim->scheduleInitialize(currentGrid->getLevel(i), scheduler);
       sim->scheduleComputeStableTimestep(currentGrid->getLevel(i),scheduler);

       // so we can initially regrid
       Task* task = scinew Task("initializeErrorEstimate", this,
                                &AMRSimulationController::initializeErrorEstimate,
                                sharedState);
       task->computes(sharedState->get_refineFlag_label(), sharedState->refineFlagMaterials());
       task->computes(sharedState->get_refinePatchFlag_label(), sharedState->refineFlagMaterials());
       sched->addTask(task, currentGrid->getLevel(i)->eachPatch(),
                      sharedState->allMaterials());
       sim->scheduleInitialErrorEstimate(currentGrid->getLevel(i), scheduler);
     }
   }
   
   if(output)
      output->finalizeTimestep(t, 0, currentGrid, scheduler, true);

   amrout << "Compiling initial schedule\n";
   scheduler->compile();
   
   double dt=Time::currentSeconds() - getStartTime();
   if(d_myworld->myrank() == 0)
     cout << "done taskgraph compile (" << dt << " seconds)\n";
   // No scrubbing for initial step
   scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
   scheduler->execute();

   // this section is for "automatically" generating all the levels we can
   // so far, based on the existence of flagged cells, but limited to
   // the number of levels the regridder can handle .
   // Only do if not restarting

   if (!d_restarting && regridder->isAdaptive())
     while (currentGrid->numLevels() < regridder->maxLevels() &&
            regridder->flaggedCellsOnFinestLevel(currentGrid, scheduler)) {
       oldGrid = currentGrid;
       cout << "  DOING ANOTHER INITIALIZATION REGRID!!!!\n";
       currentGrid = regridder->regrid(oldGrid.get_rep(), scheduler, ups);
       cout << "---------- OLD GRID ----------" << endl << *(oldGrid.get_rep());
       cout << "---------- NEW GRID ----------" << endl << *(currentGrid.get_rep());
       if (currentGrid == oldGrid)
         break;

       // reset the scheduler here - it doesn't hurt anything here
       // as we're going to have to recompile the TG anyway.
       scheduler->initialize(1, 1);
       scheduler->advanceDataWarehouse(currentGrid);

       // for dynamic lb's, set up patch config after changing grid
       lb->possiblyDynamicallyReallocate(currentGrid, false); 

       for(int i=0;i<currentGrid->numLevels();i++) {
	 Task* task = scinew Task("initializeErrorEstimate", this,
				  &AMRSimulationController::initializeErrorEstimate,
				  sharedState);
	 task->computes(sharedState->get_refineFlag_label(), sharedState->refineFlagMaterials());
         task->computes(sharedState->get_refinePatchFlag_label(), sharedState->refineFlagMaterials());
	 sched->addTask(task, currentGrid->getLevel(i)->eachPatch(),
			sharedState->allMaterials());

         sim->scheduleInitialize(currentGrid->getLevel(i), scheduler);
         sim->scheduleComputeStableTimestep(currentGrid->getLevel(i),scheduler);
         sim->scheduleInitialErrorEstimate(currentGrid->getLevel(i), scheduler);
       }
       scheduler->compile();
   
       double dt=Time::currentSeconds() - getStartTime();
       if(d_myworld->myrank() == 0)
         cout << "done taskgraph compile (" << dt << " seconds)\n";
       // No scrubbing for initial step
       scheduler->get_dw(1)->setScrubbing(DataWarehouse::ScrubNone);
       scheduler->execute();
       
     }

   initSimulationStatsVars();

   std::vector<int> levelids;

   ////////////////////////////////////////////////////////////////////////////
   // The main time loop; here the specified problem is actually getting solved
   
   bool first=true;
   int  iterations = 0;
   double prev_delt = 0;
   while( ( t < timeinfo.maxTime ) && 
	  ( iterations < timeinfo.num_time_steps ) ) {

     // After one step (either timestep or initialization) and correction
     // the delta we can finally, finalize our old timestep, eg. 
     // finalize and advance the Datawarehouse

     // Put the current time into the shared state so other components
     // can access it.  Also increment (by one) the current time step
     // number so components can tell what timestep they are on.  Remember the old
     // timestep to print the stats

     // int lastTimestep = sharedState->getCurrentTopLevelTimeStep();  // RNJ Apparently not used.

     sharedState->setElapsedTime(t);
     sharedState->incrementCurrentTopLevelTimeStep();

     oldGrid = currentGrid;

     if (regridder->needsToReGrid() && !first) {
       cout << "REGRIDDING!!!!!\n";
       currentGrid = regridder->regrid(oldGrid.get_rep(), scheduler, ups);
       if (currentGrid != oldGrid) {
         cout << "---------- OLD GRID ----------" << endl << *(oldGrid.get_rep());
         cout << "---------- NEW GRID ----------" << endl << *(currentGrid.get_rep());
         
         cout << "---------- ABOUT TO RESCHEDULE ----------" << endl;
         
         // Compute number of dataWarehouses
         //int numDWs=1;
         //for(int i=1;i<oldGrid->numLevels();i++)
         //numDWs *= oldGrid->getLevel(i)->timeRefinementRatio();
         
         scheduler->advanceDataWarehouse(currentGrid);
         scheduler->initialize(1, 1);
         scheduler->clearMappings();
         scheduler->mapDataWarehouse(Task::OldDW, 0);
         scheduler->mapDataWarehouse(Task::NewDW, 1);
         
         OnDemandDataWarehouse* oldDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(scheduler->get_dw(0));
         OnDemandDataWarehouse* newDataWarehouse = dynamic_cast<OnDemandDataWarehouse*>(scheduler->getLastDW());
         
         for ( int levelIndex = 0; levelIndex < currentGrid->numLevels(); levelIndex++ ) {
           Task* task = new Task("SchedulerCommon::copyDataToNewGrid",
                                 dynamic_cast<SchedulerCommon*>(scheduler.get_rep()),
                                 &SchedulerCommon::copyDataToNewGrid);
           vector<VarLabelMatlPatch> variableInfo;
           oldDataWarehouse->getVarLabelMatlPatchTriples(variableInfo);
           for ( unsigned int i = 0; i < variableInfo.size(); i++ ) {
             VarLabelMatlPatch currentVar = variableInfo[i];
             task->computes(currentVar.label_);
           }
           cout << "RANDY: Going to copy data for level " << levelIndex << " with ID of ";
           cout << currentGrid->getLevel(levelIndex)->getID() << endl;
           scheduler->addTask(task, currentGrid->getLevel(levelIndex)->eachPatch(), sharedState->allMaterials());
           if ( levelIndex != 0 ) {
             sim->scheduleRefine(currentGrid->getLevel(levelIndex), scheduler);
           }
         }
	
         scheduler->compile();
         scheduler->execute();

         vector<VarLabelMatlLevel> reductionVariableInfo;
         oldDataWarehouse->getVarLabelMatlLevelTriples(reductionVariableInfo);
         
         cerr << getpid() << ": RANDY: Copying reduction variables" << endl;

         for ( unsigned int i = 0; i < reductionVariableInfo.size(); i++ ) {
           VarLabelMatlLevel currentReductionVar = reductionVariableInfo[i];
           // cout << "REDUNCTION:  Label(" << setw(15) << currentReductionVar.label_->getName() << "): Patch(" << reinterpret_cast<int>(currentReductionVar.level_) << "): Material(" << currentReductionVar.matlIndex_ << ")" << endl; 
           const Level* oldLevel = currentReductionVar.level_;
           const Level* newLevel = NULL;
           if (oldLevel) {
             newLevel = (newDataWarehouse->getGrid()->getLevel( oldLevel->getIndex() )).get_rep();
           }
           
           if(!oldDataWarehouse->d_reductionDB.exists(currentReductionVar.label_, currentReductionVar.matlIndex_, currentReductionVar.level_))
             SCI_THROW(UnknownVariable(currentReductionVar.label_->getName(), oldDataWarehouse->getID(), currentReductionVar.level_, currentReductionVar.matlIndex_, "in copyDataTo ReductionVariable"));
           ReductionVariableBase* v = oldDataWarehouse->d_reductionDB.get(currentReductionVar.label_, currentReductionVar.matlIndex_, currentReductionVar.level_);
           newDataWarehouse->d_reductionDB.put(currentReductionVar.label_, currentReductionVar.matlIndex_, newLevel, v->clone(), false);
         }
         cout << "---------- DONE RESCHEDULING ----------" << endl;
	 cerr << "The grids are different!" << endl;         
       } else {
	 cerr << "The grids are the same!" << endl;
       }
     }

     scheduler->advanceDataWarehouse(currentGrid);

     // Compute number of dataWarehouses
     int totalFine=1;
     for(int i=1;i<currentGrid->numLevels();i++)
       totalFine *= currentGrid->getLevel(i)->timeRefinementRatio();
     
     iterations ++;
     calcWallTime();
 
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
     // clamp timestep to output/checkpoint
     if (timeinfo.timestep_clamping && output) {
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

     prev_delt=delt;

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

     if(needRecompile(t, delt, currentGrid, sim, output, lb, regridder, levelids) || first){
       first=false;
       if(d_myworld->myrank() == 0)
	 cout << "Compiling taskgraph...\n";
       double start = Time::currentSeconds();
       
       scheduler->initialize(1, totalFine);
       scheduler->fillDataWarehouses(currentGrid);

       // Set up new DWs, DW mappings.
       scheduler->clearMappings();
       scheduler->mapDataWarehouse(Task::OldDW, 0);
       scheduler->mapDataWarehouse(Task::NewDW, totalFine);
       
       sim->scheduleTimeAdvance(currentGrid->getLevel(0), scheduler, 0, 1);
       for(int i=0;i<currentGrid->numLevels();i++){
	 Task* task = scinew Task("initializeErrorEstimate", this,
				  &AMRSimulationController::initializeErrorEstimate,
				  sharedState);
	 task->computes(sharedState->get_refineFlag_label(), sharedState->refineFlagMaterials());
         task->computes(sharedState->get_refinePatchFlag_label(), sharedState->refineFlagMaterials());
	 sched->addTask(task, currentGrid->getLevel(i)->eachPatch(),
			sharedState->allMaterials());
       }
       
       if(currentGrid->numLevels() > 1)
	 subCycle(currentGrid, scheduler, sharedState, 0, totalFine, 1, sim);

       scheduler->clearMappings();
       scheduler->mapDataWarehouse(Task::OldDW, 0);
       scheduler->mapDataWarehouse(Task::NewDW, totalFine);
       for(int i=0;i<currentGrid->numLevels();i++){
	 sim->scheduleErrorEstimate(currentGrid->getLevel(i), scheduler);
	 sim->scheduleComputeStableTimestep(currentGrid->getLevel(i), scheduler);
       }

       if(output)
	 output->finalizeTimestep(t, delt, currentGrid, scheduler,true);

       scheduler->compile();

       double dt=Time::currentSeconds() - start;
       if(d_myworld->myrank() == 0)
	 cout << "DONE TASKGRAPH RE-COMPILE (" << dt << " seconds)\n";
       levelids.resize(currentGrid->numLevels());
       for(int i=0;i<currentGrid->numLevels();i++)
	 levelids[i]=currentGrid->getLevel(i)->getID();
     }
     else {
       if (output)
         output->finalizeTimestep(t, delt, currentGrid, scheduler, false);
     }

     oldDW->override(delt_vartype(delt), sharedState->get_delt_label());
     double delt_fine = delt;
     int skip=totalFine;
     for(int i=0;i<currentGrid->numLevels();i++){
       const Level* level = currentGrid->getLevel(i).get_rep();
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
       scheduler->get_dw(i)->setScrubbing(DataWarehouse::ScrubNone);
     scheduler->get_dw(totalFine)->setScrubbing(DataWarehouse::ScrubNone);
     scheduler->execute();
     if(output)
       output->executedTimestep(delt);
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

bool
AMRSimulationController::needRecompile(double time, double delt,
				       const GridP& grid,
				       SimulationInterface* sim,
				       Output* output,
				       LoadBalancer* lb,
                                       Regridder* regridder,
				       vector<int>& levelids)
{
  // Currently, output, sim, lb, regridder can request a recompile.  --bryan
  bool recompile = false;
  
  // do it this way so everybody can have a chance to maintain their state
  recompile |= (output && output->needRecompile(time, delt, grid));
  recompile |= (sim && sim->needRecompile(time, delt, grid));
  recompile |= (lb && lb->needRecompile(time, delt, grid));
  recompile |= (regridder && regridder->needRecompile(time, delt, grid));
  recompile |= (static_cast<int>(levelids.size()) != grid->numLevels());
  return recompile;
}


void
AMRSimulationController::initializeErrorEstimate(const ProcessorGroup*,
						 const PatchSubset* patches,
						 const MaterialSubset* matls,
						 DataWarehouse*,
						 DataWarehouse* new_dw,
						 SimulationStateP sharedState)
{
  // only make one refineFlag per patch.  Do not loop over matls!
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    CCVariable<int> refineFlag;
    PerPatch<int> refinePatchFlag(0);
    new_dw->allocateAndPut(refineFlag, sharedState->get_refineFlag_label(),
                           0, patch);
    new_dw->put(refinePatchFlag, sharedState->get_refinePatchFlag_label(),
                0, patch);
    refineFlag.initialize(0);
  }
}

