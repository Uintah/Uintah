#include <sci_defs/malloc_defs.h>

#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationTime.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/CCA/Ports/Regridder.h>
#include <Packages/Uintah/CCA/Ports/Output.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/OS/Dir.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Time.h>

#include <sys/param.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#define SECONDS_PER_MINUTE 60.0
#define SECONDS_PER_HOUR   3600.0
#define SECONDS_PER_DAY    86400.0
#define SECONDS_PER_WEEK   604800.0
#define SECONDS_PER_YEAR   31536000.0

using namespace SCIRun;
using namespace std;

static DebugStream dbg("SimulationStats", true);
static DebugStream dbgTime("SimulationTimeStats", false);
static DebugStream simdbg("SimulationController", false);

namespace Uintah {

  // for calculating memory usage when sci-malloc is disabled.
  char* SimulationController::start_addr = NULL;

  double stdDeviation(double sum_of_x, double sum_of_x_squares, int n)
  {
    return sqrt((n*sum_of_x_squares - sum_of_x*sum_of_x)/(n*n));
  }

  SimulationController::SimulationController(const ProcessorGroup* myworld, bool doAMR)
    : UintahParallelComponent(myworld), d_doAMR(doAMR)
  {
    d_n = 0;
    d_wallTime = 0;
    d_startTime = 0;
    d_prevWallTime = 0;
    d_sumOfWallTimes = 0;
    d_sumOfWallTimeSquares = 0;

    d_restarting = false;
    d_combinePatches = false;
    d_archive = NULL;
  }

  SimulationController::~SimulationController()
  {
    delete d_timeinfo;
  }

  void SimulationController::doCombinePatches(std::string fromDir, bool reduceUda)
  {
    ASSERT(!d_doAMR);
    d_combinePatches = true;
    d_reduceUda = reduceUda;
    d_fromDir = fromDir;
  }

  void SimulationController::doRestart(std::string restartFromDir, int timestep,
                                       bool fromScratch, bool removeOldDir)
  {
    d_restarting = true;
    d_fromDir = restartFromDir;
    d_restartTimestep = timestep;
    d_restartFromScratch = fromScratch;
    d_restartRemoveOldDir = removeOldDir;
  }

  void SimulationController::loadUPS( void )
  {
    UintahParallelPort* pp = getPort("problem spec");
    ProblemSpecInterface* psi = dynamic_cast<ProblemSpecInterface*>(pp);
    
    if( !psi ){
      cout << "SimpleSimulationController::run() psi dynamic_cast failed...\n";
      throw InternalError("psi dynamic_cast failed", __FILE__, __LINE__);
    }

    // Get the problem specification
    d_ups = psi->readInputFile();
    d_ups->writeMessages(d_myworld->myrank() == 0);
    if(!d_ups)
      throw ProblemSetupException("Cannot read problem specification", __FILE__, __LINE__);
    
    releasePort("problem spec");
    
    if(d_ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification", __FILE__, __LINE__);
  }

  void SimulationController::preGridSetup( void )
  {
    d_sharedState = scinew SimulationState(d_ups);
    
    d_output = dynamic_cast<Output*>(getPort("output"));
    
    if( !d_output ){
      cout << "dynamic_cast of 'd_output' failed!\n";
      throw InternalError("dynamic_cast of 'd_output' failed!", __FILE__, __LINE__);
    }
    d_output->problemSetup(d_ups, d_sharedState.get_rep());
    
  }

  GridP SimulationController::gridSetup( void ) 
  {
    GridP grid;


    if (d_restarting) {
      // create the DataArchive here, and store it, as we use it a few times...
      // We need to read the grid before ProblemSetup, and we can't load all
      // the data until after problemSetup, so we have to do a few 
      // different DataArchive operations

      Dir restartFromDir(d_fromDir);
      Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
      d_archive = scinew DataArchive(checkpointRestartDir.getName(),
                          d_myworld->myrank(), d_myworld->size());

      vector<int> indices;
      vector<double> times;
      d_archive->queryTimesteps(indices, times);

      unsigned i;
      // find the right time to query the grid
      if (d_restartTimestep == 0) {
        i = 0; // timestep == 0 means use the first timestep
      }
      else if (d_restartTimestep == -1 && indices.size() > 0) {
        i = (unsigned int)(indices.size() - 1); 
      }
      else {
        for (i = 0; i < indices.size(); i++)
          if (indices[i] == d_restartTimestep)
            break;
      }
      
      if (i == indices.size()) {
        // timestep not found
        ostringstream message;
        message << "Timestep " << d_restartTimestep << " not found";
        throw InternalError(message.str(), __FILE__, __LINE__);
      }
      
      d_restartTime = times[i];
    }

    if (!d_doAMR || !d_restarting) {
      grid = scinew Grid;
      grid->problemSetup(d_ups, d_myworld, d_doAMR);
      grid->performConsistencyCheck();
    }
    else {
      // maybe someday enforce to always load the grid from the DataArchive, but for
      // now only do that with AMR.

      grid = d_archive->queryGrid(d_restartTime, d_ups.get_rep());
      
    }
    if(grid->numLevels() == 0){
      throw InternalError("No problem (no levels in grid) specified.", __FILE__, __LINE__);
    }
    
    // Print out meta data
    if (d_myworld->myrank() == 0)
      grid->printStatistics();
    
    return grid;
  }

  void SimulationController::postGridSetup( GridP& grid)
  {
    // Initialize the CFD and/or MPM components
    d_sim = dynamic_cast<SimulationInterface*>(getPort("sim"));
    if(!d_sim)
      throw InternalError("No simulation component", __FILE__, __LINE__);

    if (d_restarting) {
      // do these before calling sim->problemSetup
      XMLURL blah; // unused
      const ProblemSpecP spec = d_archive->getTimestep(d_restartTime, blah);
      d_sim->readFromTimestepXML(spec);

      // set prevDelt to what it was in the last simulation.  If in the last 
      // sim we were clamping delt based on the values of prevDelt, then
      // delt will be off if it doesn't match.
      ProblemSpecP timeSpec = spec->findBlock("Time");
      if (timeSpec) {
        d_sharedState->d_prev_delt = 0.0;
        timeSpec->get("delt", d_sharedState->d_prev_delt);
      }

      // eventually we will also probably query the material properties here.
    }

    d_sim->problemSetup(d_ups, grid, d_sharedState);
    
    // Finalize the shared state/materials
    d_sharedState->finalizeMaterials();
    
    Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
    sched->problemSetup(d_ups, d_sharedState);
    d_scheduler = sched;
    
    d_lb = sched->getLoadBalancer();
    d_lb->problemSetup(d_ups, d_sharedState);
    
    // done after the sim->problemSetup to get defaults into the
    // input.xml, which it writes along with index.xml
    d_output->initializeOutput(d_ups);
    
    // set up regridder with initial infor about grid
    if (d_doAMR) {
      d_regridder = dynamic_cast<Regridder*>(getPort("regridder"));
      d_regridder->problemSetup(d_ups, grid, d_sharedState);
    }

  }

  void SimulationController::restartSetup( GridP& grid, double& t )
  {
    simdbg << "Restarting... loading data\n";
    
    // create a temporary DataArchive for reading in the checkpoints
    // archive for restarting.
    Dir restartFromDir(d_fromDir);
    Dir checkpointRestartDir = restartFromDir.getSubdir("checkpoints");
    
    double delt = 0;
    
    d_archive->restartInitialize(d_restartTimestep, grid, d_scheduler->get_dw(1), d_lb, 
                              &t, &delt);
    
    d_sharedState->setCurrentTopLevelTimeStep( d_restartTimestep );
    // Tell the scheduler the generation of the re-started simulation.
    // (Add +1 because the scheduler will be starting on the next
    // timestep.)
    d_scheduler->setGeneration( d_restartTimestep+1 );
    
    // just in case you want to change the delt on a restart....
    if (d_timeinfo->override_restart_delt != 0) {
      double newdelt = d_timeinfo->override_restart_delt;
      if (d_myworld->myrank() == 0)
        cout << "Overriding restart delt with " << newdelt << endl;
      d_scheduler->get_dw(1)->override(delt_vartype(newdelt), 
                                       d_sharedState->get_delt_label());
      double delt_fine = newdelt;
      for(int i=0;i<grid->numLevels();i++){
        const Level* level = grid->getLevel(i).get_rep();
        if(i != 0)
          delt_fine /= level->timeRefinementRatio();
        d_scheduler->get_dw(1)->override(delt_vartype(delt_fine), d_sharedState->get_delt_label(),
                                         level);
      }
    }
    d_scheduler->get_dw(1)->finalize();
    ProblemSpecP pspec = d_archive->getRestartTimestepDoc();
    XMLURL url = d_archive->getRestartTimestepURL();
    //d_lb->restartInitialize(pspec, url);
    
    d_output->restartSetup(restartFromDir, 0, d_restartTimestep, t,
                           d_restartFromScratch, d_restartRemoveOldDir);

    // don't need it anymore...
    delete d_archive;
  }
  
  void SimulationController::adjustDelT(double& delt, double prev_delt, int iterations, double t) 
  {
#if 0
    cout << "maxTime = " << d_timeinfo->maxTime << endl;
    cout << "initTime = " << d_timeinfo->initTime << endl;
    cout << "delt_min = " << d_timeinfo->delt_min << endl;
    cout << "delt_max = " << d_timeinfo->delt_max << endl;
    cout << "timestep_multiplier = " << d_timeinfo->delt_factor << endl;
    cout << "delt_init = " << d_timeinfo->max_initial_delt << endl;
    cout << "initial_delt_range = " << d_timeinfo->initial_delt_range << endl;
    cout << "max_delt_increase = " << d_timeinfo->max_delt_increase << endl;
#endif

    delt *= d_timeinfo->delt_factor;
      
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
    if (d_timeinfo->timestep_clamping && d_output) {
      double orig_delt = delt;
      double nextOutput = d_output->getNextOutputTime();
      double nextCheckpoint = d_output->getNextCheckpointTime();
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
    
  }

  double SimulationController::getWallTime  ( void )
  {
    return d_wallTime;
  }

  void SimulationController::calcWallTime ( void )
  {
    d_wallTime = Time::currentSeconds() - d_startTime;
  }

  double SimulationController::getStartTime ( void )
  {
    return d_startTime;
  }

  void SimulationController::calcStartTime ( void )
  {
    d_startTime = Time::currentSeconds();
  }

  void SimulationController::setStartSimTime ( double t )
  {
    d_startSimTime = t;
  }

  void SimulationController::initSimulationStatsVars ( void )
  {
    // vars used to calculate standard deviation
    d_n = 0;
    d_wallTime = 0;
    d_prevWallTime = Time::currentSeconds();
    d_sumOfWallTimes = 0; // sum of all walltimes
    d_sumOfWallTimeSquares = 0; // sum all squares of walltimes
  }

  void SimulationController::printSimulationStats ( SimulationStateP sharedState, double delt, double time )
  {
#ifndef _WIN32 // win32 doesn't have sci-malloc or sbrk
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
    unsigned long memuse = 0;
    if ( ProcessInfo::IsSupported( ProcessInfo::MEM_RSS ) ) {
      memuse = ProcessInfo::GetMemoryResident();
    } else {
      memuse = (char*)sbrk(0)-start_addr;
    }
    unsigned long highwater = 0;
#endif
    
    // get memory stats for each proc if MALLOC_PERPROC is in the environent
    if ( getenv( "MALLOC_PERPROC" ) ) {
      ostream* mallocPerProcStream = NULL;
      char* filenamePrefix = getenv( "MALLOC_PERPROC" );
      if ( !filenamePrefix || strlen( filenamePrefix ) == 0 ) {
	mallocPerProcStream = &dbg;
      } else {
	char filename[MAXPATHLEN];
	sprintf( filename, "%s.%d" ,filenamePrefix, d_myworld->myrank() );
	if ( sharedState->getCurrentTopLevelTimeStep() == 0 ) {
	  mallocPerProcStream = new ofstream( filename, ios::out | ios::trunc );
	} else {
	  mallocPerProcStream = new ofstream( filename, ios::out | ios::app );
	}
	if ( !mallocPerProcStream ) {
	  delete mallocPerProcStream;
	  mallocPerProcStream = &dbg;
	}
      }
      *mallocPerProcStream << "Proc "     << d_myworld->myrank() << "   ";
      *mallocPerProcStream << "Timestep " << sharedState->getCurrentTopLevelTimeStep() << "   ";
      *mallocPerProcStream << "Size "     << ProcessInfo::GetMemoryUsed() << "   ";
      *mallocPerProcStream << "RSS "      << ProcessInfo::GetMemoryResident() << "   ";
      *mallocPerProcStream << "Sbrk "     << (char*)sbrk(0) - start_addr << "   ";
#ifndef DISABLE_SCI_MALLOC
      *mallocPerProcStream << "Sci_Malloc_Memuse "    << memuse << "   ";
      *mallocPerProcStream << "Sci_Malloc_Highwater " << highwater;
#endif
      *mallocPerProcStream << endl;
      if ( mallocPerProcStream != &dbg ) {
	delete mallocPerProcStream;
      }
    }
    
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
#endif
    // calculate mean/std dev
    double stdDev = 0;
    double mean = 0;
    if (d_n > 2) { // ignore times 0,1,2
      //walltimes.push_back(d_wallTime - d_prevWallTime);
      d_sumOfWallTimes += (d_wallTime - d_prevWallTime);
      d_sumOfWallTimeSquares += pow(d_wallTime - d_prevWallTime,2);
    }
    if (d_n > 3) {
      // divide by n-2 and not n, because we wait till n>2 to keep track
      // of our stats
      stdDev = stdDeviation(d_sumOfWallTimes, d_sumOfWallTimeSquares, d_n-2);
      mean = d_sumOfWallTimes / (d_n-2);
      //         ofstream timefile("avg_elapsed_wallTime.txt");
      //         timefile << mean << " +- " << stdDev << endl;
    }

    // output timestep statistics
    if(d_myworld->myrank() == 0){
      dbg << "Time="         << time
	  << " (timestep "  << sharedState->getCurrentTopLevelTimeStep() 
	   << "), delT="     << delt
	   << ", elap T = "  << d_wallTime;
      
      if (d_n > 3) {
	dbg << ", mean: "   << mean
	    << " +- "       << stdDev;
      }
#ifndef _WIN32
      dbg << ", Mem Use = ";
      if (avg_memuse == max_memuse && avg_highwater == max_highwater) {
	dbg << avg_memuse;
	if(avg_highwater) {
	  dbg << "/" << avg_highwater;
	}
      } else {
	dbg << avg_memuse;
	if(avg_highwater) {
	  dbg << "/" << avg_highwater;
	}
	dbg << " (avg), " << max_memuse;
	if(max_highwater) {
	  dbg << "/" << max_highwater;
	}
	dbg << " (max)";
      }
#endif
      dbg << endl;

      if ( d_n > 0 ) {
	double realSecondsNow = (d_wallTime - d_prevWallTime)/delt;
	double realSecondsAvg = (d_wallTime - d_startTime)/(time-d_startSimTime);

	dbgTime << "1 sim second takes ";

	dbgTime << left << showpoint << setprecision(3) << setw(4);

	if (realSecondsNow < SECONDS_PER_MINUTE) {
	  dbgTime << realSecondsNow << " seconds (now), ";
	} else if ( realSecondsNow < SECONDS_PER_HOUR ) {
	  dbgTime << realSecondsNow/SECONDS_PER_MINUTE << " minutes (now), ";
	} else if ( realSecondsNow < SECONDS_PER_DAY  ) {
	  dbgTime << realSecondsNow/SECONDS_PER_HOUR << " hours (now), ";
	} else if ( realSecondsNow < SECONDS_PER_WEEK ) {
	  dbgTime << realSecondsNow/SECONDS_PER_DAY << " days (now), ";
	} else if ( realSecondsNow < SECONDS_PER_YEAR ) {
	  dbgTime << realSecondsNow/SECONDS_PER_WEEK << " weeks (now), ";
	} else {
	  dbgTime << realSecondsNow/SECONDS_PER_YEAR << " years (now), ";
	}

	dbgTime << setw(4);

	if (realSecondsAvg < SECONDS_PER_MINUTE) {
	  dbgTime << realSecondsAvg << " seconds (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_HOUR ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_MINUTE << " minutes (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_DAY  ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_HOUR << " hours (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_WEEK ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_DAY << " days (avg) ";
	} else if ( realSecondsAvg < SECONDS_PER_YEAR ) {
	  dbgTime << realSecondsAvg/SECONDS_PER_WEEK << " weeks (avg) ";
	} else {
	  dbgTime << realSecondsAvg/SECONDS_PER_YEAR << " years (avg) ";
	}

	dbgTime << "to calculate." << endl;
      }
 
      d_prevWallTime = d_wallTime;
      d_n++;
    }
  }
  
} // namespace Uintah {
