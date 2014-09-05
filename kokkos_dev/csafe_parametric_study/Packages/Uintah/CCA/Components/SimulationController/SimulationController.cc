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
#include <Core/Exceptions/InternalError.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/OS/Dir.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Time.h>

#ifndef _WIN32
#  include <sys/param.h>
#endif

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#define SECONDS_PER_MINUTE 60.0
#define SECONDS_PER_HOUR   3600.0
#define SECONDS_PER_DAY    86400.0
#define SECONDS_PER_WEEK   604800.0
#define SECONDS_PER_YEAR   31536000.0

#define AVERAGE_WINDOW 10
using namespace SCIRun;
using namespace std;

static DebugStream dbg("SimulationStats", true);
static DebugStream dbgTime("SimulationTimeStats", false);
static DebugStream simdbg("SimulationController", false);
static DebugStream stats("ComponentTimings", false);
static DebugStream istats("IndividualComponentTimings",false);
extern DebugStream amrout;

namespace Uintah {
  struct double_int
  {
     double val;
     int loc;
     double_int(double val, int loc): val(val), loc(loc) {}
     double_int(): val(0), loc(-1) {}
  };

  // for calculating memory usage when sci-malloc is disabled.
  char* SimulationController::start_addr = NULL;

  double stdDeviation(double sum_of_x, double sum_of_x_squares, int n)
  {
    return sqrt((n*sum_of_x_squares - sum_of_x*sum_of_x)/(n*n));
  }

  SimulationController::SimulationController(const ProcessorGroup* myworld, bool doAMR, ProblemSpecP pspec)
    : UintahParallelComponent(myworld), d_ups(pspec), d_doAMR(doAMR)
  {
    d_n = 0;
    d_wallTime = 0;
    d_startTime = 0;
    d_prevWallTime = 0;
    //d_sumOfWallTimes = 0;
    //d_sumOfWallTimeSquares = 0;
    d_movingAverage=0;

    d_restarting = false;
    d_combinePatches = false;
    d_reduceUda = false;
    d_doMultiTaskgraphing = false;
    d_archive = NULL;

    d_ups->writeMessages(d_myworld->myrank() == 0);
  }

  SimulationController::~SimulationController()
  {
    delete d_timeinfo;
  }

  void SimulationController::doCombinePatches(std::string fromDir, bool reduceUda)
  {
    d_doAMR = false;
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

  void SimulationController::preGridSetup( void )
  {
    d_sharedState = scinew SimulationState(d_ups);
    
    d_output = dynamic_cast<Output*>(getPort("output"));
    
    Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));
    sched->problemSetup(d_ups, d_sharedState);
    d_scheduler = sched;
    
    if( !d_output ){
      cout << "dynamic_cast of 'd_output' failed!\n";
      throw InternalError("dynamic_cast of 'd_output' failed!", __FILE__, __LINE__);
    }
    d_output->problemSetup(d_ups, d_sharedState.get_rep());

    ProblemSpecP amr_ps = d_ups->findBlock("AMR");
    if (amr_ps) {
      amr_ps->get("doMultiTaskgraphing", d_doMultiTaskgraphing);
    }
    
    // Parse time struct
    d_timeinfo = scinew SimulationTime(d_ups);
    d_sharedState->d_simTime = d_timeinfo;
    
    if (d_reduceUda && d_timeinfo->max_delt_increase < 1e99) {
      d_timeinfo->max_delt_increase = 1e99;
      if (d_myworld->myrank() == 0)
        cout << "  For UdaReducer: setting max_delt_increase to 1e99\n";
    }
    if (d_reduceUda && d_timeinfo->delt_max < 1e99) {
     d_timeinfo->delt_max = 1e99;
      if (d_myworld->myrank() == 0)
        cout << "  For UdaReducer: setting delt_max to 1e99\n";
    }
    if (d_reduceUda && d_timeinfo->max_initial_delt < 1e99) {
     d_timeinfo->max_initial_delt = 1e99;
      if (d_myworld->myrank() == 0)
        cout << "  For UdaReducer: setting max_initial_delt to 1e99\n";
    }
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

      // find the right time to query the grid
      if (d_restartTimestep == 0) {
        d_restartIndex = 0; // timestep == 0 means use the first timestep
        // reset d_restartTimestep to what it really is
        d_restartTimestep = indices[0];
      }
      else if (d_restartTimestep == -1 && indices.size() > 0) {
        d_restartIndex = (unsigned int)(indices.size() - 1); 
        // reset d_restartTimestep to what it really is
        d_restartTimestep = indices[indices.size() - 1];
      }
      else {
        for (int index = 0; index < (int)indices.size(); index++)
          if (indices[index] == d_restartTimestep) {
            d_restartIndex = index;
            break;
          }
      }
      
      if (d_restartIndex == indices.size()) {
        // timestep not found
        ostringstream message;
        message << "Timestep " << d_restartTimestep << " not found";
        throw InternalError(message.str(), __FILE__, __LINE__);
      }
    }

    if (!d_restarting) {
      grid = scinew Grid;
      grid->problemSetup(d_ups, d_myworld, d_doAMR);
      grid->performConsistencyCheck();
    }
    else {
      grid = d_archive->queryGrid(d_restartIndex, d_ups.get_rep());
    }
    if(grid->numLevels() == 0){
      throw InternalError("No problem (no levels in grid) specified.", __FILE__, __LINE__);
    }
    
    // Print out meta data
    if (d_myworld->myrank() == 0){
      grid->printStatistics();
      amrout << "Restart grid\n" << *grid.get_rep() << endl;
    }

    // set the dimensionality of the problem.
    IntVector low, high, size;
    grid->getLevel(0)->findCellIndexRange(low, high);
    size = high-low - grid->getLevel(0)->getExtraCells()*IntVector(2,2,2);
    d_sharedState->setDimensionality(size[0] > 1, size[1] > 1, size[2] > 1);

    return grid;
  }

  void SimulationController::postGridSetup( GridP& grid, double& t)
  {
    // initialize load balancer.  Do here since we have the dimensionality in the shared state,
    // and we want that at initialization time.
    d_lb = d_scheduler->getLoadBalancer();
    d_lb->problemSetup(d_ups, d_sharedState);
    
    // set up regridder with initial information about grid.
    // do before sim - so that Switcher (being a sim) can reset the state of the regridder
    d_regridder = dynamic_cast<Regridder*>(getPort("regridder"));
    if (d_regridder) {
      d_regridder->problemSetup(d_ups, grid, d_sharedState);
    }

    // Initialize the CFD and/or MPM components
    d_sim = dynamic_cast<SimulationInterface*>(getPort("sim"));
    if(!d_sim)
      throw InternalError("No simulation component", __FILE__, __LINE__);

    ProblemSpecP restart_prob_spec = 0;

    if (d_restarting) {
      // do these before calling archive->restartInitialize, since problemSetup creates VarLabes the DA needs
      restart_prob_spec = d_archive->getTimestepDoc(d_restartIndex);
      d_sim->readFromTimestepXML(restart_prob_spec, d_sharedState);
    }

    // Pass the restart_prob_spec to the problemSetup.  For restarting, 
    // pull the <MaterialProperties> from the restart_prob_spec.  If it is not
    // available, then we will pull the properties from the d_ups instead.
    // Needs to be done before DataArchive::restartInitialize
    d_sim->problemSetup(d_ups, restart_prob_spec, grid, d_sharedState);
    

    if (d_restarting) {
      simdbg << "Restarting... loading data\n";    
      d_archive->restartInitialize(d_restartIndex, grid, d_scheduler->get_dw(1), d_lb, &t);
      

      // set prevDelt to what it was in the last simulation.  If in the last 
      // sim we were clamping delt based on the values of prevDelt, then
      // delt will be off if it doesn't match.
      ProblemSpecP timeSpec = restart_prob_spec->findBlock("Time");
      if (timeSpec) {
        d_sharedState->d_prev_delt = 0.0;
        if (!timeSpec->get("oldDelt", d_sharedState->d_prev_delt))
          // the delt is deprecated since it is misleading, but older udas may have it...
          timeSpec->get("delt", d_sharedState->d_prev_delt);
      }

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
          if(i != 0 && !d_sharedState->isLockstepAMR()) {
            delt_fine /= level->getRefinementRatioMaxDim();
          }
          d_scheduler->get_dw(1)->override(delt_vartype(delt_fine), d_sharedState->get_delt_label(),
                                          level);
        }
      }
      d_scheduler->get_dw(1)->finalize();
      
      // don't need it anymore...
      delete d_archive;
    }

    // Finalize the shared state/materials
    d_sharedState->finalizeMaterials();
    
    // done after the sim->problemSetup to get defaults into the
    // input.xml, which it writes along with index.xml
    d_output->initializeOutput(d_ups);

    if (d_restarting) {
      Dir dir(d_fromDir);
      d_output->restartSetup(dir, 0, d_restartTimestep, t,
                             d_restartFromScratch, d_restartRemoveOldDir);
    }

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
    cout << "iterations = " << iterations << endl;
    cout << "delt = " << delt << endl;
    cout << "prev_delt = " << prev_delt << endl;
#endif


    delt *= d_timeinfo->delt_factor;
      
    if(delt < d_timeinfo->delt_min){
      if(d_myworld->myrank() == 0)
        cerr << "WARNING: raising delt from " << delt
             << " to minimum: " << d_timeinfo->delt_min << '\n';
      delt = d_timeinfo->delt_min;
    }
    if(iterations > 0 && d_timeinfo->max_delt_increase < 1.e90
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
    if (d_timeinfo->end_on_max_time && t + delt > d_timeinfo->maxTime){
       delt = d_timeinfo->maxTime - t;
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
    //d_sumOfWallTimes = 0; // sum of all walltimes
    //d_sumOfWallTimeSquares = 0; // sum all squares of walltimes
  }

string
toHumanUnits( unsigned long value )
{
  char tmp[64];
  
  sprintf( tmp, "%.2lf", value / 1000000.0 );
  return tmp;
}

void
SimulationController::printSimulationStats ( int timestep, double delt, double time )
{
  // get memory stats for output
#if !defined(DISABLE_SCI_MALLOC)
  size_t nalloc,  sizealloc, nfree,  sizefree, nfillbin,
    nmmap, sizemmap, nmunmap, sizemunmap, highwater_alloc,  
    highwater_mmap, bytes_overhead, bytes_free, bytes_fragmented, bytes_inuse, bytes_inhunks;
  
  GetGlobalStats(DefaultAllocator(),
                 nalloc, sizealloc, nfree, sizefree,
                 nfillbin, nmmap, sizemmap, nmunmap,
                 sizemunmap, highwater_alloc, highwater_mmap,
                 bytes_overhead, bytes_free, bytes_fragmented, bytes_inuse, bytes_inhunks);
  unsigned long memuse = sizealloc - sizefree;
  unsigned long highwater = highwater_mmap;
#elif !defined(_WIN32)
  unsigned long memuse = 0;
  if ( ProcessInfo::IsSupported( ProcessInfo::MEM_SIZE ) ) {
    memuse = ProcessInfo::GetMemoryResident();
  } else {
    memuse = (char*)sbrk(0)-start_addr;
  }
  unsigned long highwater = 0;
#else
  unsigned long highwater = 0, memuse = 0;
#endif
  
  // get memory stats for each proc if MALLOC_PERPROC is in the environent
  if ( getenv( "MALLOC_PERPROC" ) ) {
    ostream* mallocPerProcStream = NULL;
    char* filenamePrefix = getenv( "MALLOC_PERPROC" );
    if ( !filenamePrefix || strlen( filenamePrefix ) == 0 ) {
      mallocPerProcStream = &dbg;
    } else {
      char filename[256];
      sprintf( filename, "%s.%d" ,filenamePrefix, d_myworld->myrank() );
      if ( timestep == 0 ) {
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
    *mallocPerProcStream << "Timestep " << timestep << "   ";
    *mallocPerProcStream << "Size "     << ProcessInfo::GetMemoryUsed() << "   ";
    *mallocPerProcStream << "RSS "      << ProcessInfo::GetMemoryResident() << "   ";
#ifndef _WIN32
    *mallocPerProcStream << "Sbrk "     << (char*)sbrk(0) - start_addr << "   ";
#endif
#ifndef DISABLE_SCI_MALLOC
    *mallocPerProcStream << "Sci_Malloc_Memuse "    << memuse << "   ";
    *mallocPerProcStream << "Sci_Malloc_Highwater " << highwater;
#endif
    *mallocPerProcStream << endl;
    if ( mallocPerProcStream != &dbg ) {
      delete mallocPerProcStream;
    }
  }
  
  // with the sum reduces, use double, since with memory it is possible that
  // it will overflow
  double avg_memuse = memuse;
  unsigned long max_memuse = memuse;
  int max_memuse_loc = -1;
  double avg_highwater = highwater;
  unsigned long max_highwater = highwater;
  
  // a little ugly, but do it anyway so we only have to do one reduce for sum and
  // one reduce for max
  std::vector<double> toReduce, avgReduce;
  std::vector<double_int> toReduceMax;
  std::vector<double_int> maxReduce;
  std::vector<const char*> statLabels;
  int rank=d_myworld->myrank();
  double total_time=0, overhead_time=0, percent_overhead=0;
  if (d_myworld->size() > 1) {
    toReduce.push_back(memuse);
    toReduceMax.push_back(double_int(memuse,rank));
    toReduce.push_back(d_sharedState->compilationTime);
    toReduceMax.push_back(double_int(d_sharedState->compilationTime,rank));
    toReduce.push_back(d_sharedState->regriddingTime);
    toReduceMax.push_back(double_int(d_sharedState->regriddingTime,rank));
    toReduce.push_back(d_sharedState->regriddingCompilationTime);
    toReduceMax.push_back(double_int(d_sharedState->regriddingCompilationTime,rank));
    toReduce.push_back(d_sharedState->regriddingCopyDataTime);
    toReduceMax.push_back(double_int(d_sharedState->regriddingCopyDataTime,rank));
    toReduce.push_back(d_sharedState->loadbalancerTime);
    toReduceMax.push_back(double_int(d_sharedState->loadbalancerTime,rank));
    toReduce.push_back(d_sharedState->taskExecTime);
    toReduceMax.push_back(double_int(d_sharedState->taskExecTime,rank));
    toReduce.push_back(d_sharedState->taskGlobalCommTime);
    toReduceMax.push_back(double_int(d_sharedState->taskGlobalCommTime,rank));
    toReduce.push_back(d_sharedState->taskLocalCommTime);
    toReduceMax.push_back(double_int(d_sharedState->taskLocalCommTime,rank));
    toReduce.push_back(d_sharedState->taskWaitCommTime);
    toReduceMax.push_back(double_int(d_sharedState->taskWaitCommTime,rank));
    toReduce.push_back(d_sharedState->outputTime);
    toReduceMax.push_back(double_int(d_sharedState->outputTime,rank));
    statLabels.push_back("Mem usage");
    statLabels.push_back("Recompile");
    statLabels.push_back("Regridding");
    statLabels.push_back("Regrid-schedule");
    statLabels.push_back("Regrid-copydata");
    statLabels.push_back("LoadBalance");
    statLabels.push_back("TaskExec");
    statLabels.push_back("TaskGlobalComm");
    statLabels.push_back("TaskLocalComm");
    statLabels.push_back("TaskWaitCommTime");
    statLabels.push_back("Output");
    
    if (highwater) // add highwater to the end so we know where everything else is (as highwater is conditional)
    {
      toReduce.push_back(highwater);
      toReduceMax.push_back(double_int(highwater,rank));
    }
    avgReduce.resize(toReduce.size());
    maxReduce.resize(toReduce.size());
    
    
    //if AMR and using dynamic dilation use an allreduce
    if(d_regridder && d_regridder->useDynamicDilation())
    {
      MPI_Allreduce(&toReduce[0], &avgReduce[0], toReduce.size(), MPI_DOUBLE, MPI_SUM, d_myworld->getComm());
      MPI_Allreduce(&toReduceMax[0], &maxReduce[0], toReduceMax.size(), MPI_DOUBLE_INT, MPI_MAXLOC, d_myworld->getComm());
    }
    else
    {
      MPI_Reduce(&toReduce[0], &avgReduce[0], toReduce.size(), MPI_DOUBLE, MPI_SUM, 0,
                 d_myworld->getComm());
      MPI_Reduce(&toReduceMax[0], &maxReduce[0], toReduceMax.size(), MPI_DOUBLE_INT, MPI_MAXLOC, 0,
                 d_myworld->getComm());
    }

    // make sums averages
    for (unsigned i = 0; i < avgReduce.size(); i++) {
      avgReduce[i] /= d_myworld->size();
    }

    // get specific values - pop front since we don't know if there is a highwater
    avg_memuse = avgReduce[0];
    max_memuse = maxReduce[0].val;
    max_memuse_loc = maxReduce[0].loc;
    
    if(highwater){
      avg_highwater = avgReduce[avgReduce.size()-1];
      max_highwater = maxReduce[maxReduce.size()-1].val;
    }
    //sum up the average times for simulation components
    total_time=0;
    for(int i=1;i<10;i++)
      total_time+=avgReduce[i];
    //sum up the average time for overhead related components
    for(int i=1;i<6;i++)
      overhead_time+=avgReduce[i];
    
    //calculate percentage of time spent in overhead
    percent_overhead=overhead_time/total_time;
  }
  else
  {
    //sum up the times for simulation components
    total_time=d_sharedState->compilationTime
              +d_sharedState->regriddingTime
              +d_sharedState->regriddingCompilationTime
              +d_sharedState->regriddingCopyDataTime
              +d_sharedState->loadbalancerTime
              +d_sharedState->taskExecTime
              +d_sharedState->taskGlobalCommTime
              +d_sharedState->taskLocalCommTime
              +d_sharedState->taskWaitCommTime;
    
    //sum up the average time for overhead related components
    overhead_time=d_sharedState->compilationTime
              +d_sharedState->regriddingTime
              +d_sharedState->regriddingCompilationTime
              +d_sharedState->regriddingCopyDataTime
              +d_sharedState->loadbalancerTime;
    
    //calculate percentage of time spent in overhead
    percent_overhead=overhead_time/total_time;
    
  }

  //set the overhead sample
  if(d_n>2)  //ignore the first 3 samples, they are not good samples
  {
    d_sharedState->overhead[d_sharedState->overheadIndex]=percent_overhead;
    //increment the overhead index
      
    double overhead=0;
    double weight=0;

    int t=min(d_n-2,OVERHEAD_WINDOW);
    //calcualte total weight by incrementing through the overhead sample array backwards and multiplying samples by the weights
    for(int i=0;i<t;i++)
    {
      overhead+=d_sharedState->overhead[(d_sharedState->overheadIndex+OVERHEAD_WINDOW-i)%OVERHEAD_WINDOW]*d_sharedState->overheadWeights[i];
      weight+=d_sharedState->overheadWeights[i];
    }
    d_sharedState->overheadAvg=overhead/weight; 
    
    d_sharedState->overheadIndex=(d_sharedState->overheadIndex+1)%OVERHEAD_WINDOW;
    //increase overhead size if needed
  } 
  d_sharedState->clearStats();

  // calculate mean/std dev
  //double stdDev = 0;
  double mean = 0;
  double walltime = d_wallTime-d_prevWallTime;
  
  if (d_n > 2) { // ignore times 0,1,2
    //walltimes.push_back();
    //d_sumOfWallTimes += (walltime);
    //d_sumOfWallTimeSquares += pow(walltime,2);

    //alpha=2/(N+1)
    float alpha=2.0/(min(d_n-2,AVERAGE_WINDOW)+1);  
    d_movingAverage=alpha*walltime+(1-alpha)*d_movingAverage;
    mean=d_movingAverage;

  }
  /*
  if (d_n > 3) {
    // divide by n-2 and not n, because we wait till n>2 to keep track
    // of our stats
    stdDev = stdDeviation(d_sumOfWallTimes, d_sumOfWallTimeSquares, d_n-2);
    //mean = d_sumOfWallTimes / (d_n-2);
    //         ofstream timefile("avg_elapsed_wallTime.txt");
    //         timefile << mean << " +- " << stdDev << endl;
  }
  */
  
  // output timestep statistics

    if (istats.active() && d_myworld->size() > 1) {
      for (unsigned i = 1; i < statLabels.size(); i++) { // index 0 is memuse
        if (toReduce[i] > 0)
          istats << "rank: " << d_myworld->myrank() << " " << statLabels[i] << " avg: " << toReduce[i] << endl;
      }
    } 
  
  if(d_myworld->myrank() == 0){
    char walltime[96];
    if (d_n > 3) {
      //sprintf(walltime, ", elap T = %.2lf, mean: %.2lf +- %.3lf", d_wallTime, mean, stdDev);
      sprintf(walltime, ", elap T = %.2lf, mean: %.2lf", d_wallTime, mean);
    }
    else {
      sprintf(walltime, ", elap T = %.2lf", d_wallTime);
    }
    ostringstream message;
    message.setf(ios::scientific,ios::floatfield);
    message.precision(10); 
    message << "Time="         << time;
    message.setf(ios::scientific ,ios::floatfield);
    
    
    message << " (timestep "  << timestep 
        << "), delT="     << delt
        << walltime;
#ifndef _WIN32
    message << ", Mem Use (MB)= ";
    if (avg_memuse == max_memuse && avg_highwater == max_highwater) {
      message << toHumanUnits((unsigned long) avg_memuse);
      if(avg_highwater) {
        message << "/" << toHumanUnits((unsigned long) avg_highwater);
      }
    } else {
      message << toHumanUnits((unsigned long) avg_memuse);
      if(avg_highwater) {
        message << "/" << toHumanUnits((unsigned long)avg_highwater);
      }
      message << " (avg), " << toHumanUnits(max_memuse);
      if(max_highwater) {
        message << "/" << toHumanUnits(max_highwater);
      }
      message << " (max on rank:" << max_memuse_loc << ")";
    }
    
#endif
    dbg << message.str() << endl;
    dbg.flush();
    cout.flush();

    if (stats.active()) {
      if(d_myworld->size()>1)
      {
        for (unsigned i = 1; i < statLabels.size(); i++) { // index 0 is memuse
          if (maxReduce[i].val > 0)
            stats << statLabels[i] << " avg: " << avgReduce[i] << " max: " << maxReduce[i].val << " maxloc:" << maxReduce[i].loc
                  << " LIB%: " << 1-(avgReduce[i]/maxReduce[i].val) << endl;
        }
      }
      if(d_n>2)
        stats << "Percent Time in overhead:" << d_sharedState->overheadAvg*100 <<  endl;
    } 


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
  }
  d_n++;
}
  
} // namespace Uintah {
