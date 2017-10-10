/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

/**
 *  \ingroup SUS
 *  \author Steven G. Parker
 *  \date   February, 2000
 *
 *  \brief sus.cc: Standalone Uintah Simulation (SUS).
 *  <ul>
 *    <li> Parse command line arguments </li>
 *    <li> Initialize MPI </li>
 *    <li> Create: </li>
 *      <ul>
 *        <li> <b>ProblemSpecReader</b> </li>
 *        <li> <b>SimulationController</b> </li>
 *        <li> <b>Regridder</b> </li>
 *        <li> <b>SolverInterface</b> </li>
 *        <li> <b>SimulationInterface and UintahParallelComponent</b> (e.g. ICE, MPM, etc) </li>
 *        <li> <b>ModelMaker</b> </li>
 *        <li> <b>LoadBalancer</b> </li>
 *        <li> <b>DataArchiver</b> </li>
 *        <li> <b>Scheduler</b> and add reference (Schedulers are <b>RefCounted><b>) </li>
 *      </ul>
 *    <li> Call SimulationController::run() - this is the main simulation loop </li>
 *    <li> remove added references (<b>RefCounted</b>) </li>
 *    <li> Cleanup allocated memory and exit </li>
 *  </ul>
 */

#include <CCA/Components/DataArchiver/DataArchiver.h>
#include <CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <CCA/Components/Models/ModelFactory.h>
#include <CCA/Components/Parent/ComponentFactory.h>
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/ReduceUda/UdaReducer.h>
#include <CCA/Components/Regridder/RegridderFactory.h>
#include <CCA/Components/Schedulers/SchedulerFactory.h>
#include <CCA/Components/SimulationController/AMRSimulationController.h>
#include <CCA/Components/Solvers/CGSolver.h>
#include <CCA/Components/Solvers/DirectSolve.h>

#ifdef HAVE_HYPRE
#  include <CCA/Components/Solvers/HypreSolver.h>
#endif

#include <CCA/Components/Solvers/SolverFactory.h>

#ifdef HAVE_CUDA
#  include <CCA/Components/Schedulers/UnifiedScheduler.h>
#endif

#include <CCA/Ports/DataWarehouse.h>

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Malloc/Allocator.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahMPI.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Environment.h>
#include <Core/Util/FileUtils.h>

#include <sci_defs/hypre_defs.h>
#include <sci_defs/malloc_defs.h>
#include <sci_defs/uintah_defs.h>
#include <sci_defs/visit_defs.h>
#include <sci_defs/cuda_defs.h>

#include <svn_info.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef HAVE_VISIT
#  include <VisIt/libsim/visit_libsim.h>
#endif


#if HAVE_IEEEFP_H
#  include <ieeefp.h>
#endif
#if 0
#  include <fenv.h>
#endif

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>

using namespace Uintah;


#if defined( USE_LENNY_HACK )
  // See Core/Malloc/Allocator.cc for more info.
  namespace Uintah {
    extern void shutdown();
  };
#endif


namespace {

std::mutex cerr_mutex{};

Dout g_stack_debug(       "ExceptionStack" , true );
Dout g_wait_for_debugger( "WaitForDebugger", false );

}


static
void
quit(const std::string& msg = "")
{
  if (msg != "") {
    std::cerr << msg << "\n";
  }
  Uintah::Parallel::finalizeManager();
  Parallel::exitAll(2);
}


static
void
usage( const std::string& message, const std::string& badarg, const std::string& progname )
{
  int argc = 0;
  char **argv;
  argv = nullptr;

  // Initialize MPI so that "usage" is only printed by proc 0.
  // (If we are using MPICH, then Uintah::MPI::Init() has already been called.)
  Uintah::Parallel::initializeManager(argc, argv);

  if (Uintah::Parallel::getMPIRank() == 0) {
    std::cerr << "\n";
    if (badarg != "") {
      std::cerr << "Error parsing argument: " << badarg << '\n';
    }

    std::cerr << "\n";
    std::cerr << message << "\n";
    std::cerr << "\n";
    std::cerr << "Usage: " << progname << " [options] <input_file_name>\n\n";
    std::cerr << "Valid options are:\n";
    std::cerr << "-h[elp]              : This usage information\n";
    std::cerr << "-AMR                 : use AMR simulation controller\n";
#ifdef HAVE_CUDA
    std::cerr << "-gpu                 : use available GPU devices, requires multi-threaded Unified scheduler \n";
#endif
    std::cerr << "-gpucheck            : returns 1 if sus was compiled with CUDA and there is a GPU available. \n";
    std::cerr << "                     : returns 2 if sus was not compiled with CUDA or there are no GPUs available. \n";
    std::cerr << "-nthreads <#>        : number of threads per MPI process, requires multi-threaded Unified scheduler\n";
    std::cerr << "-layout NxMxO        : Eg: 2x1x1.  MxNxO must equal number tof boxes you are using.\n";
    std::cerr << "-local_filesystem    : If using MPI, use this flag if each node has a local disk.\n";
    std::cerr << "-emit_taskgraphs     : Output taskgraph information\n";
    std::cerr << "-restart             : Give the checkpointed uda directory as the input file\n";
    std::cerr << "-reduce_uda          : Reads <uda-dir>/input.xml file and removes unwanted labels (see FAQ).\n";
    std::cerr << "-uda_suffix <number> : Make a new uda dir with <number> as the default suffix\n";
    std::cerr << "-t <timestep>        : Restart timestep (last checkpoint is default, you can use -t 0 for the first checkpoint)\n";
    std::cerr << "-svnDiff             : runs svn diff <src/...../Packages/Uintah \n";
    std::cerr << "-svnStat             : runs svn stat -u & svn info <src/...../Packages/Uintah \n";
    std::cerr << "-copy                : Copy from old uda when restarting\n";
    std::cerr << "-move                : Move from old uda when restarting\n";
    std::cerr << "-nocopy              : Default: Don't copy or move old uda timestep when restarting\n";
    std::cerr << "-validate            : Verifies the .ups file is valid and quits!\n";
    std::cerr << "-do_not_validate     : Skips .ups file validation! Please avoid this flag if at all possible.\n";
#ifdef HAVE_VISIT
    std::cerr << "\n";
    std::cerr << "-visit <filename>        : Create a VisIt .sim2 file and perform VisIt in-situ checks\n";
    std::cerr << "-visit_connect           : Wait for a visit connection before executing the simulation\n";
    std::cerr << "-visit_comment <comment> : A comment about the simulation\n";
    std::cerr << "-visit_dir <directory>   : Top level directory for the VisIt installation\n";
    std::cerr << "-visit_option <string>   : Optional args for the VisIt launch script\n";
    std::cerr << "-visit_trace <file>      : Trace file for VisIt's Sim V2 function calls\n";
    std::cerr << "-visit_ui <file>         : Use the named Qt GUI file instead of the default\n";
#endif
    std::cerr << "\n\n";
  }
  quit();
}


void
sanityChecks()
{
#if defined( DISABLE_SCI_MALLOC )
  if (getenv("MALLOC_STATS")) {
    printf("\nERROR:\n");
    printf("ERROR: Environment variable MALLOC_STATS set, but  --enable-sci-malloc was not configured...\n");
    printf("ERROR:\n\n");
    Parallel::exitAll(1);
  }
  if (getenv("MALLOC_TRACE")) {
    printf("\nERROR:\n");
    printf("ERROR: Environment variable MALLOC_TRACE set, but  --enable-sci-malloc was not configured...\n");
    printf("ERROR:\n\n");
    Parallel::exitAll(1);
  }
  if (getenv("MALLOC_STRICT")) {
    printf("\nERROR:\n");
    printf("ERROR: Environment variable MALLOC_STRICT set, but --enable-sci-malloc  was not configured...\n");
    printf("ERROR:\n\n");
    Parallel::exitAll(1);
  }
#endif
}


void
abortCleanupFunc()
{
  Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
}


int
main( int argc, char *argv[], char *env[] )
{
#if defined( USE_LENNY_HACK )
  atexit( Uintah::shutdown );
#endif

  sanityChecks();

#if HAVE_IEEEFP_H
  fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);
#endif
#if 0
  feenableexcept(FE_INVALID|FE_OVERFLOW|FE_DIVBYZERO);
#endif

  /*
   * Default values
   */
  bool   do_AMR              = false;
  bool   emit_graphs         = false;
  bool   local_filesystem    = false;
  bool   restart             = false;
  bool   reduce_uda          = false;
  bool   do_svnDiff          = false;
  bool   do_svnStat          = false;
  bool   restartFromScratch  = true;
  bool   restartRemoveOldDir = false;
  bool   validateUps         = true;
  bool   onlyValidateUps     = false;

  int    restartTimestep     = -1;
  int    udaSuffix           = -1;
  int    numThreads          =  0;
  int    numPartitions       =  0;
  int    threadsPerPartition =  0;

  std::string udaDir;       // for restart
  std::string filename;     // name of the UDA directory
  std::string solver = "";  // empty string defaults to CGSolver

#ifdef HAVE_VISIT
  // Assume if VisIt is compiled in that the user may want to connect
  // with VisIt.
  unsigned int do_VisIt = VISIT_SIMMODE_UNKNOWN;
#endif

  IntVector layout(1,1,1);

  /*
   * Parse arguments
   */
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "-help") || (arg == "-h")) {
      usage("", "", argv[0]);
    }
    else if (arg == "-AMR" || arg == "-amr") {
      if( reduce_uda ) {
        usage( "You may not use '-amr' and '-reduce_uda' at the same time.", arg, argv[0] );
      }
      do_AMR = true;
    }
    else if (arg == "-nthreads") {
      if (++i == argc) {
        usage("You must provide a number of threads for -nthreads", arg, argv[0]);
      }
      numThreads = atoi(argv[i]);
      if( numThreads < 1 ) {
        usage("Number of threads is too small", arg, argv[0]);
      }
      else if( numThreads > MAX_THREADS ) {
        usage( "Number of threads is out of range. Specify fewer threads, "
               "or increase MAX_THREADS (.../src/Core/Parallel/Parallel.h) and recompile.", arg, argv[0] );
      }
      Uintah::Parallel::setNumThreads( numThreads );
    }
    else if (arg == "-npartitions") {
      if (++i == argc) {
        usage("You must provide a number of thread partitions for -npartitions", arg, argv[0]);
      }
      numPartitions = atoi(argv[i]);
      if( numPartitions < 1 ) {
        usage("Number of thread partitions is too small", arg, argv[0]);
      }
      else if( numPartitions > MAX_THREADS ) {
        usage( "Number of thread partitions is out of range. Specify fewer thread partitions, "
               "or increase MAX_THREADS (.../src/Core/Parallel/Parallel.h) and recompile.", arg, argv[0] );
      }
      Uintah::Parallel::setNumPartitions( numPartitions );
    }
    else if (arg == "-nthreadsperpartition") {
      if (++i == argc) {
        usage("You must provide a number of thread partitions for -nthreadsperpartition", arg, argv[0]);
      }
      threadsPerPartition = atoi(argv[i]);
      if( threadsPerPartition < 1 ) {
        usage("Number of threads per partition must be > 0", arg, argv[0]);
      }
#ifdef _OPENMP
      if( threadsPerPartition > omp_get_max_threads() ) {
        usage("Number of threads per partition must be < omp_get_max_threads()", arg, argv[0]);
      }
#endif
      Uintah::Parallel::setThreadsPerPartition(threadsPerPartition);
    }
    else if (arg == "-solver") {
      if (++i == argc) {
        usage("You must provide a solver name for -solver", arg, argv[0]);
      }
      solver = argv[i];
    }
    else if (arg == "-mpi") {
      // TODO: Remove all traces of the need to use "-mpi" on the command line - APH 09/16/16
      //         Most of this will be removing "-mpi" from nightly RT scripts,
      //         as well as removing all prior usage of Parallel::usingMPI().
    }
    else if (arg == "-emit_taskgraphs") {
      emit_graphs = true;
    }
    else if (arg == "-local_filesystem") {
      local_filesystem = true;
    }
    else if (arg == "-restart") {
      restart = true;
    }
    else if (arg == "-handle_mpi_errors") {
      // handled in Parallel.cc
    }
    else if (arg == "-uda_suffix") {
      if (i < argc - 1) {
        udaSuffix = atoi(argv[++i]);
      }
      else {
        usage("You must provide a suffix number for -uda_suffix", arg, argv[0]);
      }
    }
    else if (arg == "-nocopy") {  // default anyway, but that's fine
      restartFromScratch = true;
    }
    else if (arg == "-copy") {
      restartFromScratch = false;
      restartRemoveOldDir = false;
    }
    else if (arg == "-move") {
      restartFromScratch = false;
      restartRemoveOldDir = true;
    }
    else if (arg == "-gpucheck") {
#ifdef HAVE_CUDA
      int retVal = UnifiedScheduler::verifyAnyGpuActive();
      if (retVal == 1) {
        std::cout << "At least one GPU detected!" << std::endl;
      } else {
        std::cout << "No GPU detected!" << std::endl;
      }
      Parallel::exitAll(retVal);
#endif
      std::cout << "No GPU detected!" << std::endl;
      Parallel::exitAll(2); // If the above didn't exit with a 1, then we didn't have a GPU, so exit with a 2.
      std::cout << "This doesn't run" << std::endl;
    }
#ifdef HAVE_CUDA
    else if(arg == "-gpu") {
      Uintah::Parallel::setUsingDevice( true );
    }
#endif
    else if (arg == "-t") {
      if (i < argc - 1) {
        restartTimestep = atoi(argv[++i]);
      }
    }
    else if (arg == "-layout") {
      if (++i == argc) {
        usage("You must provide a vector arg for -layout", arg, argv[0]);
      }
      int ii, jj, kk;
      if (sscanf(argv[i], "%dx%dx%d", &ii, &jj, &kk) != 3) {
        usage("Error parsing -layout", argv[i], argv[0]);
      }
      layout = IntVector(ii, jj, kk);
    }
    else if (arg == "-svnDiff") {
      do_svnDiff = true;
    }
    else if (arg == "-svnStat") {
      do_svnStat = true;
    }
    else if (arg == "-validate") {
      onlyValidateUps = true;
    }
    else if (arg == "-do_not_validate") {
      validateUps = false;
    }
    else if (arg == "-reduce_uda" || arg == "-reduceUda") {
      if( do_AMR ) {
        usage( "You may not use '-amr' and '-reduce_uda' at the same time.", arg, argv[0] );
      }
      reduce_uda = true;
    }
    else if (arg == "-arches" || arg == "-ice" || arg == "-impm" || arg == "-mpm" || arg == "-mpmarches" || arg == "-mpmice"
        || arg == "-poisson1" || arg == "-poisson2" || arg == "-switcher" || arg == "-poisson4" || arg == "-benchmark"
        || arg == "-mpmf" || arg == "-rmpm" || arg == "-smpm" || arg == "-amrmpm" || arg == "-smpmice" || arg == "-rmpmice") {
      usage(std::string("'") + arg + "' is deprecated.  Simulation component must be specified " + "in the .ups file!", arg, argv[0]);
    }
    // If VisIt is included then the user may send optional args to VisIt.
    // The most important is the directory path to where VisIt is located.
#ifdef HAVE_VISIT
    else if (arg == "-visit") {
      if (++i == argc) {
        usage("You must provide file name for -visit", arg, argv[0]);
      }
      else if( do_VisIt == VISIT_SIMMODE_UNKNOWN )
	do_VisIt = VISIT_SIMMODE_RUNNING;
    }
    else if (arg == "-visit_connect" ) {
      do_VisIt = VISIT_SIMMODE_STOPPED;
    }
    else if (arg == "-visit_comment" ) {
      if (++i == argc) {
        usage("You must provide a string for -visit_comment", arg, argv[0]);
      }
      else if( do_VisIt == VISIT_SIMMODE_UNKNOWN )
	do_VisIt = VISIT_SIMMODE_RUNNING;
    }
    else if (arg == "-visit_dir" ) {
      if (++i == argc) {
        usage("You must provide a directory for -visit_dir", arg, argv[0]);
      }
      else if( do_VisIt == VISIT_SIMMODE_UNKNOWN )
	do_VisIt = VISIT_SIMMODE_RUNNING;
    }
    else if (arg == "-visit_option" ) {
      if (++i == argc) {
        usage("You must provide a string for -visit_option", arg, argv[0]);
      }
      else if( do_VisIt == VISIT_SIMMODE_UNKNOWN )
	do_VisIt = VISIT_SIMMODE_RUNNING;
    }
    else if (arg == "-visit_trace" ) {
      if (++i == argc) {
        usage("You must provide a file name for -visit_trace", arg, argv[0]);
      }
      else if( do_VisIt == VISIT_SIMMODE_UNKNOWN )
	do_VisIt = VISIT_SIMMODE_RUNNING;
    }
    else if (arg == "-visit_ui" ) {
      if (++i == argc) {
        usage("You must provide a file name for -visit_ui", arg, argv[0]);
      }
      else if( do_VisIt == VISIT_SIMMODE_UNKNOWN )
	do_VisIt = VISIT_SIMMODE_RUNNING;
    }
#endif
    else {
      if (filename != "") { // A filename was already provided, thus this is an error.
        usage("", arg, argv[0]);
      }
      else if (argv[i][0] == '-') {  // Don't allow 'filename' to begin with '-'.
        usage("Error!  It appears that the filename you specified begins with a '-'.\n"
              "        This is not allowed.  Most likely there is problem with your\n"
              "        command line.",
              argv[i], argv[0]);
      }
      else {
        filename = argv[i];
      }
    }
  }
 
  // Pass the env into the sci env so it can be used there...
  create_sci_environment( env, nullptr, true );

  if( filename == "" ) {
    usage("No input file specified", "", argv[0]);
  }

  if(g_wait_for_debugger) {
    TURN_ON_WAIT_FOR_DEBUGGER();
  }

  //__________________________________
  //  bulletproofing
  if ( restart || reduce_uda ) {
    udaDir = filename;
    filename = filename + "/input.xml";

    // If restarting (etc), make sure that the uda specified is not a symbolic link to an Uda.
    // This is because the sym link can (will) be updated to point to a new uda, thus creating
    // an inconsistency.  Therefore it is just better not to use the sym link in the first place.
    if( isSymLink( udaDir.c_str() ) ) {
      std::cout << "\n";
      std::cout << "ERROR: " + udaDir + " is a symbolic link.  Please use the full name of the UDA.\n";
      std::cout << "\n";
      Uintah::Parallel::finalizeManager();
      Parallel::exitAll( 1 );
    }
  }

  char * start_addr = (char*)sbrk(0);

  bool thrownException = false;

  try {

    // Initialize after parsing the args...
    Uintah::Parallel::initializeManager( argc, argv );

    // Uncomment the following to see what the environment is... this is useful to figure out
    // what environment variable can be checked for (in Uintah/Core/Parallel/Parallel.cc)
    // to automatically determine that sus is running under MPI (instead of having to
    // be explicit with the "-mpi" arg):
    //
//    if( Uintah::Parallel::getMPIRank() == 0 ) {
//      show_env();
//    }

    if( !validateUps ) {
      // Print out warning message here (after Parallel::initializeManager()), so that
      // proc0cout works correctly.
      proc0cout << "\n";
      proc0cout << "WARNING: You have turned OFF .ups file validation... this may cause many unforeseen problems\n";
      proc0cout << "         with your simulation run.  It is strongly suggested that you leave validation on!\n";
      proc0cout << "\n";
    }

#if defined(MALLOC_TRACE)
    ostringstream traceFilename;
    traceFilename << "mallocTrace-" << Uintah::Parallel::getMPIRank();
    MALLOC_TRACE_LOG_FILE( traceFilename.str().c_str() );
    //mallocTraceInfo.setTracingState( false );
#endif

    if (Uintah::Parallel::getMPIRank() == 0) {
      // helpful for cleaning out old stale udas
      time_t t = time(nullptr);
      std::string time_string(ctime(&t));
      char name[256];
      gethostname(name, 256);

      std::cout << "Date:    " << time_string;  // has its own newline
      std::cout << "Machine: " << name << "\n";
      std::cout << "SVN: " << SVN_REVISION << "\n";
      std::cout << "SVN: " << SVN_DATE << "\n";
      std::cout << "SVN: " << SVN_URL << "\n";
      std::cout << "Assertion level: " << SCI_ASSERTION_LEVEL << "\n";
      std::cout << "CFLAGS: " << CFLAGS << "\n";

      // Run svn commands on Packages/Uintah 
      if (do_svnDiff || do_svnStat) {
        std::cout << "____SVN_____________________________________________________________\n";
        std::string sdir = std::string(sci_getenv("SCIRUN_SRCDIR"));
        if (do_svnDiff) {
          std::string cmd = "svn diff --username anonymous --password \"\" " + sdir;
          std::system(cmd.c_str());
        }
        if (do_svnStat) {
          std::string cmd = "svn info  --username anonymous --password \"\" " + sdir;
          std::system(cmd.c_str());
          cmd = "svn stat -u  --username anonymous --password \"\" " + sdir;
          std::system(cmd.c_str());
        }
        std::cout << "____SVN_______________________________________________________________\n";
      }
    }

    char * st = getenv( "INITIAL_SLEEP_TIME" );
    if( st != nullptr ){
      char name[256];
      gethostname(name, 256);
      int sleepTime = atoi( st );
      if (Uintah::Parallel::getMPIRank() == 0) {
        std::cout << "SLEEPING FOR " << sleepTime
             << " SECONDS TO ALLOW DEBUGGER ATTACHMENT\n";
      }
      std::cout << "PID for rank " << Uintah::Parallel::getMPIRank() << " (" << name << ") is " << getpid() << "\n";
      std::cout.flush();

      struct timespec ts;
      ts.tv_sec = (int) sleepTime;
      ts.tv_nsec = (int)(1.e9 * (sleepTime - ts.tv_sec));

      nanosleep(&ts, &ts);
    }

    //__________________________________
    // Read input file
    ProblemSpecP ups;
    try {
      ups = ProblemSpecReader().readInputFile( filename, validateUps );
    }
    catch( ProblemSetupException& err ) {
      proc0cout << "\nERROR caught while parsing UPS file: " << filename << "\nDetails follow.\n"
                << err.message() << "\n";
      Uintah::Parallel::finalizeManager();
      Parallel::exitAll( 0 );
    }
    catch( ... ) {
      // Bulletproofing.  Catches the case where a user accidentally specifies a UDA directory
      // instead of a UPS file.
      proc0cout   << "\n";
      proc0cout   << "ERROR - Failed to parse UPS file: " << filename << ".\n";
      if( validDir( filename ) ) {
        proc0cout << "ERROR - Note: '" << filename << "' is a directory! Did you mistakenly specify a UDA instead of an UPS file?\n";
      }
      proc0cout   << "\n";
      Uintah::Parallel::finalizeManager();
      Parallel::exitAll( 0 );
    }

    if( onlyValidateUps ) {
      std::cout << "\nValidation of .ups File finished... good bye.\n\n";
      ups = nullptr; // This cleans up memory held by the 'ups'.
      Uintah::Parallel::finalizeManager();
      Parallel::exitAll( 0 );
    }

    // If VisIt is included then the user may be attching into Visit's
    // libsim for in-situ analysis and visualization. This call pass
    // optional arguments that VisIt will interpert.
#ifdef HAVE_VISIT
    if( do_VisIt )
    {
      bool have_comment = false;

      for (int i = 1; i < argc; i++)
      {
        std::string arg = argv[i];

	if (arg == "-visit_comment" ) {
	  have_comment = true;
	  break;
	}
      }

      // No user defined comment so use the ups simulation meta data
      // title.
      if( !have_comment )
      {
	// Find the meta data and the title. 
	std::string title;
	
	if( ups->findBlock( "Meta" ) )
	  ups->findBlock( "Meta" )->require( "title", title );
	
	if( title.size() )
	{
	  // Have the title so pass that into the libsim 
	  char **new_argv = (char **) malloc((argc + 2) * sizeof(*new_argv));
	  
	  if (new_argv != nullptr)
	  {
	    memmove(new_argv, argv, sizeof(*new_argv) * argc);
	    
	    argv = new_argv;
	    
	    argv[argc] =
	      (char*) malloc( (strlen("-visit_comment")+1) * sizeof(char) );
	    strcpy( argv[argc], "-visit_comment" );
	    ++argc;
	    
	    argv[argc] =
	      (char*) malloc( (title.size()+1) * sizeof(char) );
	    strcpy( argv[argc], title.c_str() );
	    ++argc;
	  }
	}
      }

      visit_LibSimArguments( argc, argv );      
    }
#endif

    // If the AMR block is defined default to turning AMR on.
    if ( !do_AMR ) {
      ProblemSpecP grid_ps = ups->findBlock( "Grid" );
      if( grid_ps ) {
        grid_ps->getAttribute( "doAMR", do_AMR );
      }
    }

    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();

    SimulationController* ctl = scinew AMRSimulationController( world, do_AMR, ups );

    ctl->getSimulationStateP()->setUseLocalFileSystems( local_filesystem );

#ifdef HAVE_VISIT
    ctl->getSimulationStateP()->setVisIt( do_VisIt );
#endif

    RegridderCommon* regridder = nullptr;
    if( do_AMR ) {
      regridder = RegridderFactory::create( ups, world );
      if ( regridder ) {
        ctl->attachPort( "regridder", regridder );
      }
    }

    //__________________________________
    // Solver
    SolverInterface * solve = SolverFactory::create( ups, world, solver );

    proc0cout << "Implicit Solver: \t" << solve->getName() << "\n";


    //______________________________________________________________________
    // Create the components
    MALLOC_TRACE_TAG("main():create components");


    //__________________________________
    // Component
    // try to make it from the command line first, then look in ups file
    UintahParallelComponent* comp = ComponentFactory::create( ups, world, do_AMR, udaDir );
    SimulationInterface* sim = dynamic_cast<SimulationInterface*>(comp);

    // set sim. controller flags for reduce uda
    if ( reduce_uda ) {
      ctl->setReduceUdaFlags( udaDir );
    }

    ctl->attachPort(  "sim",       sim );
    comp->attachPort( "solver",    solve );
    comp->attachPort( "regridder", regridder );
    
#ifndef NO_ICE
    //__________________________________
    //  Model
    ModelMaker* modelmaker = scinew ModelFactory(world);
    comp->attachPort("modelmaker", modelmaker);
#endif

    //__________________________________
    // Load balancer
    LoadBalancerCommon* lbc = LoadBalancerFactory::create( ups, world );
    lbc->attachPort("sim", sim);
    if( regridder ) {
      regridder->attachPort( "load balancer", lbc );
      lbc->attachPort(       "regridder",     regridder );
    }
    
    //__________________________________
    // Output
    DataArchiver * dataarchiver = scinew DataArchiver( world, udaSuffix );
    Output       * output       = dataarchiver;
    ctl->attachPort( "output", dataarchiver );
    dataarchiver->attachPort( "load balancer", lbc );
    comp->attachPort( "output", dataarchiver );
    dataarchiver->attachPort( "sim", sim );
    
    //__________________________________
    // Scheduler
    SchedulerCommon* sched = SchedulerFactory::create(ups, world, output);
    sched->attachPort( "load balancer", lbc );
    ctl->attachPort(   "scheduler",     sched );
    lbc->attachPort(   "scheduler",     sched );
    comp->attachPort(  "scheduler",     sched );

    sched->setStartAddr( start_addr );
    
    if ( regridder ) {
      regridder->attachPort( "scheduler", sched );
    }
    sched->addReference();

    if ( emit_graphs ) {
      sched->doEmitTaskGraphDocs();
    }

    //__________________________________
    // Start the simulation controller
    if ( restart ) {
      ctl->doRestart( udaDir, restartTimestep, restartFromScratch, restartRemoveOldDir );
    }
    
    // This gives memory held by the 'ups' back before the simulation starts... Assuming
    // no one else is holding on to it...
    ups = nullptr;

    ctl->run();
    delete ctl;

    sched->removeReference();
    delete sched;
    if ( regridder ) {
      delete regridder;
    }
    delete lbc;
    delete sim;
    delete solve;
    delete output;

#ifndef NO_ICE
    delete modelmaker;
#endif
  }
  catch (ProblemSetupException& e) {
    // Don't show a stack trace in the case of ProblemSetupException.
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << "\n\n(Proc: " << Uintah::Parallel::getMPIRank() << ") Caught: " << e.message() << "\n\n";
    thrownException = true;
  }
  catch (Exception& e) {
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << "\n\n(Proc " << Uintah::Parallel::getMPIRank() << ") Caught exception: " << e.message() << "\n\n";
    if(e.stackTrace()) {
      DOUT(g_stack_debug, "Stack trace: " << e.stackTrace());
    }
    thrownException = true;
  }
  catch (std::bad_alloc& e) {
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << Uintah::Parallel::getMPIRank() << " Caught std exception 'bad_alloc': " << e.what() << '\n';
    thrownException = true;
  }
  catch (std::bad_exception& e) {
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << Uintah::Parallel::getMPIRank() << " Caught std exception: 'bad_exception'" << e.what() << '\n';
    thrownException = true;
  }
  catch (std::ios_base::failure& e) {
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << Uintah::Parallel::getMPIRank() << " Caught std exception 'ios_base::failure': " << e.what() << '\n';
    thrownException = true;
  }
  catch (std::runtime_error& e) {
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << Uintah::Parallel::getMPIRank() << " Caught std exception 'runtime_error': " << e.what() << '\n';
    thrownException = true;
  }
  catch (std::exception& e) {
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << Uintah::Parallel::getMPIRank() << " Caught std exception: " << e.what() << '\n';
    thrownException = true;
  }
  catch(...) {
    std::lock_guard<std::mutex> cerr_guard(cerr_mutex);
    std::cerr << Uintah::Parallel::getMPIRank() << " Caught unknown exception\n";
    thrownException = true;
  }
  
  Uintah::TypeDescription::deleteAll();

  /*
   * Finalize MPI
   */
  Uintah::Parallel::finalizeManager( thrownException ? Uintah::Parallel::Abort : Uintah::Parallel::NormalShutdown);

  if (thrownException) {
    if( Uintah::Parallel::getMPIRank() == 0 ) {
      std::cout << "\n\nAN EXCEPTION WAS THROWN... Goodbye.\n\n";
    }
    Parallel::exitAll(1);
  }

  if( Uintah::Parallel::getMPIRank() == 0 ) {
    std::cout << "Sus: going down successfully\n";
  }

  // use exitAll(0) since return does not work
  Parallel::exitAll(0);
  return 0;

} // end main()

