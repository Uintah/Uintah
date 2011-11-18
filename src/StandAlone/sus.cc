/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



/*
 *  sus.cc: Standalone Uintah Simulation - a bare-bones uintah simulation
 *          for development
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 *  Copyright (C) 2000 U of U
 */

#include <TauProfilerForSCIRun.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InvalidGrid.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Tracker/TrackerClient.h>

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <CCA/Components/SimulationController/AMRSimulationController.h>
#include <CCA/Components/Models/ModelFactory.h>
#include <CCA/Components/Solvers/CGSolver.h>
#include <CCA/Components/Solvers/DirectSolve.h>
#include <CCA/Components/Solvers/HypreSolver.h>
#include <CCA/Components/PatchCombiner/PatchCombiner.h>
#include <CCA/Components/PatchCombiner/UdaReducer.h>
#include <CCA/Components/DataArchiver/DataArchiver.h>
#include <CCA/Components/Solvers/SolverFactory.h>
#include <CCA/Components/Regridder/RegridderFactory.h>
#include <CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <CCA/Components/Schedulers/SchedulerFactory.h>
#include <CCA/Components/Parent/ComponentFactory.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FileUtils.h>

#include <sci_defs/hypre_defs.h>
#include <sci_defs/malloc_defs.h>
#include <sci_defs/mpi_defs.h>
#include <sci_defs/uintah_defs.h>

#include <svn_info.h>

#include <Core/Malloc/Allocator.h>

#ifdef USE_VAMPIR
#  include <Core/Parallel/Vampir.h>
#endif

#if HAVE_IEEEFP_H
#  include <ieeefp.h>
#endif
#if 0
#  include <fenv.h>
#endif

#ifdef _WIN32
#  include <process.h>
#  include <winsock2.h>
#endif

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/stat.h>

#include <time.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

#undef UINTAHSHARE
#if defined(_WIN32) && !defined(BUILD_UINTAH_STATIC)
#  define UINTAHSHARE __declspec(dllimport)
#else
#  define UINTAHSHARE
#endif

#if defined( USE_LENNY_HACK )
  // See Core/Malloc/Allocator.cc for more info.
  namespace SCIRun {
    extern void shutdown();
  };
#endif

// If we are using MPICH version 1, 
// we must call MPI_Init() before parsing args
#if defined(HAVE_MPICH) && !defined(MPICH2)
#  define HAVE_MPICH_OLD
#endif


// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)
// Mutex cerrLock( "cerr lock" );
// DebugStream mixedDebug( "MixedScheduler Debug Output Stream", false );
// DebugStream fullDebug( "MixedScheduler Full Debug", false );

extern UINTAHSHARE Mutex cerrLock;
extern UINTAHSHARE DebugStream mixedDebug;
extern UINTAHSHARE DebugStream fullDebug;
static DebugStream stackDebug("ExceptionStack", true);
static DebugStream dbgwait("WaitForDebugger", false);

static
void
quit( const std::string & msg = "" )
{
  if (msg != "") {
    cerr << msg << "\n";
  }
  Uintah::Parallel::finalizeManager();
  Thread::exitAll( 2 );
}

static
void
usage( const std::string & message,
       const std::string& badarg,
       const std::string& progname)
{
#ifndef HAVE_MPICH_OLD
  int argc = 0;
  char **argv;
  argv = 0;

  // Initialize MPI so that "usage" is only printed by proc 0.
  // (If we are using MPICH, then MPI_Init() has already been called.)
  Uintah::Parallel::initializeManager( argc, argv );
#endif

  if( Uintah::Parallel::getMPIRank() == 0 ) {
      cerr << "\n";
      if(badarg != "") {
        cerr << "Error parsing argument: " << badarg << '\n';
      }
      cerr << "\n";
      cerr << message << "\n";
      cerr << "\n";
      cerr << "Usage: " << progname << " [options] <input_file_name>\n\n";
      cerr << "Valid options are:\n";
      cerr << "-h[elp]              : This usage information.\n";
      cerr << "-AMR                 : use AMR simulation controller\n";
      cerr << "-gpu                 : use available GPU devices\n";
      cerr << "-nthreads <#>        : number of threads per MPI process, requires a multi-threaded scheduler\n";
      cerr << "-layout NxMxO        : Eg: 2x1x1.  MxNxO must equal number\n";
      cerr << "                           of boxes you are using.\n";
      cerr << "-emit_taskgraphs     : Output taskgraph information\n";
      cerr << "-restart             : Give the checkpointed uda directory as the input file\n";
      cerr << "-combine_patches     : Give a uda directory as the input file\n";  
      cerr << "-reduce_uda          : Reads <uda-dir>/input.xml file and removes unwanted labels (see FAQ).\n";
      cerr << "-uda_suffix <number> : Make a new uda dir with <number> as the default suffix\n";      
      cerr << "-t <timestep>        : Restart timestep (last checkpoint is default,\n\t\t\tyou can use -t 0 for the first checkpoint)\n";
      cerr << "-svnDiff             : runs svn diff <src/...../Packages/Uintah \n";
      cerr << "-svnStat             : runs svn stat -u & svn info <src/...../Packages/Uintah \n";
      cerr << "-copy                : Copy from old uda when restarting\n";
      cerr << "-move                : Move from old uda when restarting\n";
      cerr << "-nocopy              : Default: Don't copy or move old uda timestep when\n\t\t\trestarting\n";
      cerr << "-validate            : Verifies the .ups file is valid and quits!\n";
      cerr << "-do_not_validate     : Skips .ups file validation! Please avoid this flag if at all possible.\n";
      cerr << "-track               : Turns on (external) simulation tracking... continues w/o tracking if connection fails.\n";
      cerr << "-TRACK               : Turns on (external) simulation tracking... dies if connection fails.\n";
      cerr << "\n\n";
    }
  quit();
}

void
sanityChecks()
{
#if defined( DISABLE_SCI_MALLOC )
  if( getenv("MALLOC_STATS") ) {
    printf( "\nERROR:\n" );
    printf( "ERROR: Environment variable MALLOC_STATS set, but SCI Malloc was not configured...\n" );
    printf( "ERROR:\n\n" );
    Thread::exitAll( 1 );
  }
  if( getenv("MALLOC_TRACE") ) {
    printf( "\nERROR:\n" );
    printf( "ERROR: Environment variable MALLOC_TRACE set, but SCI Malloc was not configured...\n" );
    printf( "ERROR:\n\n" );
    Thread::exitAll( 1 );
  }
  if( getenv("MALLOC_STRICT") ) {
    printf( "\nERROR:\n" );
    printf( "ERROR: Environment variable MALLOC_STRICT set, but SCI Malloc was not configured...\n" );
    printf( "ERROR:\n\n" );
    Thread::exitAll( 1 );
  }
#endif
}

void
abortCleanupFunc()
{
  Uintah::Parallel::finalizeManager( Uintah::Parallel::Abort );
}
#include <iomanip>
int
main( int argc, char *argv[], char *env[] )
{
#if defined( USE_LENNY_HACK )
  atexit( SCIRun::shutdown );
#endif

  sanityChecks();

  string oldTag;
  MALLOC_TRACE_TAG_SCOPE("main()");

  // Turn off Thread asking so sus can cleanly exit on abortive behavior.  
  // Can override this behavior with the environment variable SCI_SIGNALMODE
  Thread::setDefaultAbortMode("exit");
  Thread::self()->setCleanupFunction( &abortCleanupFunc );

#ifdef USE_TAU_PROFILING

  // WARNING:
  //cout << "about to call tau_profile... if it dies now, it is in "
  //       << "sus.cc at the TAU_PROFILE() call.  This has only been "
  //       << "happening in 32 bit tau use.";  
  //
  // Got rid of this print as it appears 100s of times when 100s of procs.
#endif
  // Causes buserr for some reason in 32 bit mode... may be fixed now:
  TAU_PROFILE("main()", "void (int, char **)", TAU_DEFAULT);

  // This seems to be causing a problem when using LAM, disabling for now.
  //   TAU_PROFILE_INIT(argc,argv);
  
#if HAVE_IEEEFP_H
  fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);
#endif
#if 0
  feenableexcept(FE_INVALID|FE_OVERFLOW|FE_DIVBYZERO);
#endif

  /*
   * Default values
   */
  bool   do_AMR=false;
  bool   emit_graphs=false;
  bool   restart=false;
  bool   combine_patches=false;
  bool   reduce_uda=false;
  bool   do_svnDiff = false;
  bool   do_svnStat = false;
  int    restartTimestep = -1;
  int    udaSuffix = -1;
  string udaDir; // for restart or combine_patches
  bool   restartFromScratch = true;
  bool   restartRemoveOldDir = false;
//bool   useScheduler3 = false;
  int    numThreads = 0;
  string filename;
  string solver;
  IntVector layout(1,1,1);
  bool   validateUps = true, onlyValidateUps = false;
  bool   track = false, track_or_die = false;
    
  // Checks to see if user is running an MPI version of sus.
  Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );

#ifdef HAVE_MPICH_OLD
  /*
    * Initialize MPI
    */
  //
  // When using old verison of MPICH, initializeManager() uses the arg list to
  // determine whether sus is running with MPI before calling MPI_Init())
  //
  // NOTE: The main problem with calling initializeManager() before
  // parsing the args is that we don't know if thread MPI is going to
  // However, MPICH veriosn 1 does not support Thread safety, so we will just dis-allow that.
  
  Uintah::Parallel::initializeManager( argc, argv );
#endif
  /*
    * Parse arguments
    */
  for(int i=1;i<argc;i++){
    string arg = argv[i];
    if( (arg == "-help") || (arg == "-h") ) {
      usage( "", "", argv[0]);
    } else if(arg == "-AMR" || arg == "-amr"){
      do_AMR=true;
    } else if(arg == "-nthreads"){
#ifdef HAVE_MPICH_OLD
      usage ("This MPICH version does not support Thread safety! Please recompile with thread-safe MPI library for -nthreads.", arg, argv[0]) ;
#endif
      if(++i == argc){
        usage("You must provide a number of threads for -nthreads",
              arg, argv[0]);
      }
      numThreads = atoi(argv[i]);
      if ( numThreads< 2 || numThreads>32 ) {
        usage("number of threads is out of range 2...32", arg, argv[0]);
      }
      Uintah::Parallel::setMaxThreads(numThreads);
    } else if(arg == "-threadmpi"){
      //used threaded mpi (this option is handled in MPI_Communicator.cc  MPI_Init_thread
    } else if(arg == "-solver") {
      if(++i == argc){
        usage("You must provide a solver name for -solver", arg, argv[0]);
      }
      solver = argv[i];
    } else if(arg == "-mpi") {
      Uintah::Parallel::forceMPI();
    } else if(arg == "-nompi") {
      Uintah::Parallel::forceNoMPI();
    } else if (arg == "-emit_taskgraphs") {
      emit_graphs = true;
    } else if(arg == "-restart") {
      restart=true;
    } else if(arg == "-handle_mpi_errors") {
      // handled in Parallel.cc
    } else if(arg == "-uda_suffix") {
      if (i < argc-1) {
        udaSuffix = atoi(argv[++i]);
      } else {
        usage("You must provide a suffix number for -uda_suffix", arg, argv[0]);
      }
    } else if(arg == "-nocopy") { // default anyway, but that's fine
      restartFromScratch = true;
    } else if(arg == "-copy") {
      restartFromScratch = false;
      restartRemoveOldDir = false;
    } else if(arg == "-move") {
      restartFromScratch = false;
      restartRemoveOldDir = true;
#ifdef HAVE_CUDA
    } else if(arg == "-gpu") {
        Uintah::Parallel::setUsingGPU(true);
#endif
    } else if(arg == "-t") {
      if (i < argc-1) {
        restartTimestep = atoi(argv[++i]);
      }
    } else if(arg == "-layout") {
      if(++i == argc) {
        usage("You must provide a vector arg for -layout", arg, argv[0]);
      }
      int ii, jj, kk;
      if(sscanf(argv[i], "%dx%dx%d", &ii, &jj, &kk) != 3) {
        usage("Error parsing -layout", argv[i], argv[0]);
      }
      layout = IntVector(ii,jj,kk);
    } else if(arg == "-svnDiff") {
      do_svnDiff = true;
    } else if(arg == "-svnStat") {
      do_svnStat = true;
    } else if(arg == "-validate") {
      onlyValidateUps = true;
    } else if(arg == "-do_not_validate") {
      validateUps = false;
    } else if(arg == "-track") {
      track = true;
    } else if(arg == "-TRACK") {
      track = true;
      track_or_die = true;
    } else if (arg=="-reduce_uda")
    {
      reduce_uda=true;
    } else if(arg == "-combine_patches") {
      combine_patches = true;
    } else if( arg == "-arches"  || arg == "-ice"      || arg == "-impm"     || arg == "-mpm"      || arg == "-mpmarches"  ||
               arg == "-mpmice"  || arg == "-poisson1" || arg == "-poisson2" || arg == "-switcher" || arg == "-poisson4" || arg == "-benchmark" ||
               arg == "-mpmf"    || arg == "-rmpm"     || arg == "-smpm"     || arg == "-amrmpm"   || arg == "-smpmice"  ||
               arg == "-rmpmice" || arg == "-angio" ) {
      usage( string( "'" ) + arg + "' is deprecated.  Simulation component must be specified " +
             "in the .ups file!", arg, argv[0] );
    } else {
      if( filename != "" ) {
        usage("", arg, argv[0]);
      }
      else if( argv[i][0] == '-' ) { // Don't allow 'filename' to begin with '-'.
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
  create_sci_environment( env, 0, true );

  if( filename == "" ) {
    usage("No input file specified", "", argv[0]);
  }

  if( track ) {
    string server  = "updraft1.privatearch.arches";
    bool   initialized = TrackerClient::initialize( server );

    if( !initialized ) {
      if ( track_or_die ) {
        cout << "\n";
        cout << "Error: Tracking initialization failed... Good bye.\n";
        cout << "\n";
        Uintah::Parallel::finalizeManager();
        Thread::exitAll( 1 );
      }
      else {
        cout << "\n";
        cout << "WARNING: Tracking initialization failed... (Could not contact Server).  Tracking will not take place.\n";
        cout << "\n";
      }
    }
  }

  if(dbgwait.active()) {
    TURN_ON_WAIT_FOR_DEBUGGER();
  }

  if (restart || combine_patches || reduce_uda) {
    // check if state.xml is present
    // if not do normal
    udaDir = filename;
    filename = filename + "/input.xml";

    // If restarting (etc), make sure that the uda specified is not a symbolic link to an Uda.
    // This is because the sym link can (will) be updated to point to a new uda, thus creating
    // an inconsistency.  Therefore it is just better not to use the sym link in the first place.
    if( isSymLink( udaDir.c_str() ) ) {
      cout << "\n";
      cout << "Error: " + udaDir + " is a symbolic link.  Please use the full name of the UDA.\n";
      cout << "\n";
      Uintah::Parallel::finalizeManager();
      Thread::exitAll( 1 );
    }
  }

  if (!Uintah::Parallel::usingMPI()) {
    TAU_PROFILE_SET_NODE(0);
  }

  #ifdef USE_VAMPIR
  VTsetup();
  #endif

#ifndef _WIN32
  char * start_addr = (char*)sbrk(0);
  mode_t mask_gor = 0022;
  umask(mask_gor);
#endif

#if defined(__SGI__)
  Thread::disallow_sgi_OpenGL_page0_sillyness();
#endif

  bool thrownException = false;

  try {

#ifndef HAVE_MPICH_OLD
    // If regular MPI, then initialize after parsing the args...
    Uintah::Parallel::initializeManager( argc, argv );
#endif

    // Uncomment the following to see what the environment is... this is useful to figure out
    // what environment variable can be checked for (in Uintah/Core/Parallel/Parallel.cc)
    // to automatically determine that sus is running under MPI (instead of having to
    // be explicit with the "-mpi" arg):
    //
    //if( Uintah::Parallel::getMPIRank() == 0 ) {
    //  show_env();
    //}

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

    if( Uintah::Parallel::getMPIRank() == 0 ) {
      // helpful for cleaning out old stale udas
      time_t t = time(NULL) ;
      string time_string(ctime(&t));
      char name[256];
      gethostname(name, 256);
    
      cout << "Date:    " << time_string; // has its own newline
      cout << "Machine: " << name << "\n";

      cout << "SVN: " << SVN_REVISION << "\n";
      cout << "SVN: " << SVN_DATE << "\n";
      cout << "Assertion level: " << SCI_ASSERTION_LEVEL << "\n";
      cout << "CFLAGS: " << CFLAGS << "\n";

      // Run svn commands on Packages/Uintah 
      if (do_svnDiff || do_svnStat) {
#if defined(REDSTORM)
        cout << "WARNING:  SVN DIFF is disabled.\n";
#else
        cout << "____SVN_____________________________________________________________\n";
        string sdir = string(sci_getenv("SCIRUN_SRCDIR"));
        if(do_svnDiff) {
          string cmd = "svn diff " + sdir;
          system(cmd.c_str());
        }
        if(do_svnStat) {
          string cmd = "svn info " + sdir;
          system(cmd.c_str());
          cmd = "svn stat -u " + sdir;
          system(cmd.c_str());
        }
        cout << "____SVN_______________________________________________________________\n";
#endif
      }
    }

#if !defined(REDSTORM)
    char * st = getenv( "INITIAL_SLEEP_TIME" );
    if( st != 0 ){    
      char name[256];
      gethostname(name, 256);
      int sleepTime = atoi( st );
      if (Uintah::Parallel::getMPIRank() == 0) {
        cout << "SLEEPING FOR " << sleepTime 
             << " SECONDS TO ALLOW DEBUGGER ATTACHMENT\n";
      }
      cout << "PID for rank " << Uintah::Parallel::getMPIRank() << " (" << name << ") is " << getpid() << "\n";
      cout.flush();
      Time::waitFor( (double)sleepTime );
    }
#endif
    //__________________________________
    // Read input file
    ProblemSpecP ups = ProblemSpecReader().readInputFile( filename, validateUps );

    if( onlyValidateUps ) {
      cout << "\nValidation of .ups File finished... good bye.\n\n";
      ups = 0; // This cleans up memory held by the 'ups'.
      Uintah::Parallel::finalizeManager();
      Thread::exitAll( 0 );
    }

    //if the AMR block is defined default to turning amr on
    if (!do_AMR) {
      do_AMR = (bool) ups->findBlock("AMR");
    }

    //if doAMR is defined set do_AMR.
    if(do_AMR) {
      ups->get("doAMR",do_AMR);
    }
    
    // don't do AMR on combine-patches or reduce-uda
    if (reduce_uda || combine_patches) {
      do_AMR = false;
    }

    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();

    SimulationController* ctl = 
      scinew AMRSimulationController(world, do_AMR, ups);

    RegridderCommon* reg = 0;
    if(do_AMR) {
      reg = RegridderFactory::create(ups, world);
      if (reg) {
        ctl->attachPort("regridder", reg);
      }
    }

    //__________________________________
    // Solver
    SolverInterface* solve = 0;
    solve = SolverFactory::create(ups, world, solver);
    if(Uintah::Parallel::getMPIRank() == 0 && solve!=0) {
      cout << "Implicit Solver:" << solve->getName() << endl;
    }

    MALLOC_TRACE_TAG("main():create components");
    //______________________________________________________________________
    // Create the components

    //__________________________________
    // Component
    // try to make it from the command line first, then look in ups file
    UintahParallelComponent* comp = ComponentFactory::create(ups, world, do_AMR, udaDir);
    SimulationInterface* sim = dynamic_cast<SimulationInterface*>(comp);

    if (combine_patches || reduce_uda) {
      // the ctl will do nearly the same thing for combinePatches and reduceUda
      ctl->doCombinePatches(udaDir, reduce_uda); // true for reduce_uda, false for combine_patches
    }
    
    ctl->attachPort("sim", sim);
    comp->attachPort("solver", solve);
    comp->attachPort("regridder", reg);
    
#ifndef NO_ICE
    //__________________________________
    //  Model
    ModelMaker* modelmaker = scinew ModelFactory(world);
    comp->attachPort("modelmaker", modelmaker);
#endif

    //__________________________________
    // Load balancer
    LoadBalancerCommon* lbc = LoadBalancerFactory::create(ups, world);
    lbc->attachPort("sim", sim);
    if(reg) {
      reg->attachPort("load balancer", lbc);
      lbc->attachPort("regridder",reg);
    }
    
    //__________________________________
    // Output
    DataArchiver* dataarchiver = scinew DataArchiver(world, udaSuffix);
    Output* output = dataarchiver;
    ctl->attachPort("output", dataarchiver);
    dataarchiver->attachPort("load balancer", lbc);
    comp->attachPort("output", dataarchiver);
    dataarchiver->attachPort("sim", sim);
    
    //__________________________________
    // Scheduler
    SchedulerCommon* sched = SchedulerFactory::create(ups, world, output);
    sched->attachPort("load balancer", lbc);
    ctl->attachPort("scheduler", sched);
    lbc->attachPort("scheduler", sched);
    comp->attachPort("scheduler", sched);

    sched->setStartAddr( start_addr );
    
    if (reg) {
      reg->attachPort("scheduler", sched);
    }
    sched->addReference();
    
    if (emit_graphs) {
      sched->doEmitTaskGraphDocs();
    }
    
    MALLOC_TRACE_TAG(oldTag);
    /*
     * Start the simulation controller
     */
    if (restart) {
      ctl->doRestart(udaDir, restartTimestep, restartFromScratch, restartRemoveOldDir);
    }
    
    // This gives memory held by the 'ups' back before the simulation starts... Assuming
    // no one else is holding on to it...
    ups = 0;

    ctl->run();
    delete ctl;

    sched->removeReference();
    delete sched;
    if (reg) {
      delete reg;
    }
    delete lbc;
    delete sim;
    delete solve;
    delete output;

    // FIXME: don't need anymore... clean up... qwerty
    // ProblemSpecReader* n_reader = static_cast<ProblemSpecReader* >(reader);
    // n_reader->clean();
    // delete reader;

#ifndef NO_ICE
    delete modelmaker;
#endif
  } catch (ProblemSetupException& e) {
    // Don't show a stack trace in the case of ProblemSetupException.
    cerrLock.lock();
    cout << "\n\n" << Uintah::Parallel::getMPIRank() << " Caught exception: " << e.message() << "\n\n";
    cerrLock.unlock();
    thrownException = true;
  } catch (Exception& e) {
    cerrLock.lock();
    cout << "\n\n" << Uintah::Parallel::getMPIRank() << " Caught exception: " << e.message() << "\n\n";
    if(e.stackTrace())
      stackDebug << "Stack trace: " << e.stackTrace() << '\n';
    cerrLock.unlock();
    thrownException = true;
  } catch (std::bad_alloc& e) {
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught std exception 'bad_alloc': " << e.what() << '\n';
    cerrLock.unlock();
    thrownException = true;
  } catch (std::bad_exception& e) {
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught std exception: 'bad_exception'" << e.what() << '\n';
    cerrLock.unlock();
    thrownException = true;
  } catch (std::ios_base::failure& e) {
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught std exception 'ios_base::failure': " << e.what() << '\n';
    cerrLock.unlock();
    thrownException = true;
  } catch (std::runtime_error& e) {
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught std exception 'runtime_error': " << e.what() << '\n';
    cerrLock.unlock();
    thrownException = true;
  } catch (std::exception& e) {
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught std exception: " << e.what() << '\n';
    cerrLock.unlock();
    thrownException = true;
  } catch(...) {
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught unknown exception\n";
    cerrLock.unlock();
    thrownException = true;
  }
  
  Uintah::TypeDescription::deleteAll();
  
  /*
   * Finalize MPI
   */
  Uintah::Parallel::finalizeManager( thrownException ?
                                        Uintah::Parallel::Abort : Uintah::Parallel::NormalShutdown);

  if (thrownException) {
    if( Uintah::Parallel::getMPIRank() == 0 ) {
      cout << "\n\nAN EXCEPTION WAS THROWN... Goodbye.\n\n";
    }
    Thread::exitAll(1);
  }
  
  if( Uintah::Parallel::getMPIRank() == 0 ) {
    cout << "Sus: going down successfully\n";
  }

  // use exitAll(0) since return does not work
  Thread::exitAll(0);
  return 0;

} // end main()

/*
#if !defined(REDSTORM)
extern "C" {
//  void dgesvd_() {
//    cerr << "Error: dgesvd called!\n";
//    Thread::exitAll(1);
//  }

  void dpotrf_() {
    cerr << "Error: dpotrf called!\n";
    Thread::exitAll(1);
  }

  void dgetrf_() {
    cerr << "Error: dgetrf called!\n";
    Thread::exitAll(1);
  }

  void dpotrs_() {
    cerr << "Error: dpotrs called!\n";
    Thread::exitAll(1);
  }

  void dgeev_() {
    cerr << "Error: dgeev called!\n";
    Thread::exitAll(1);
  }

  void dgetrs_() {
    cerr << "Error: dgetrs called!\n";
    Thread::exitAll(1);
  }
}
#endif
*/
