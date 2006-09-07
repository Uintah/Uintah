
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
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/CCA/Components/SimulationController/AMRSimulationController.h>
#include <Packages/Uintah/CCA/Components/Models/ModelFactory.h>
#include <Packages/Uintah/CCA/Components/Solvers/CGSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/DirectSolve.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolver.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/UdaReducer.h>
#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/CCA/Components/Solvers/SolverFactory.h>
#include <Packages/Uintah/CCA/Components/Regridder/RegridderFactory.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerFactory.h>
#include <Packages/Uintah/CCA/Components/Parent/ComponentFactory.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/ieeefp_defs.h>
#include <sci_defs/hypre_defs.h>

#ifdef USE_VAMPIR
#include <Packages/Uintah/Core/Parallel/Vampir.h>
#endif

#if HAVE_IEEEFP_H
#include <ieeefp.h>
#endif
#if 0
#include <fenv.h>
#endif

#ifdef _WIN32
#include <process.h>
#endif

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)
// Mutex cerrLock( "cerr lock" );
// DebugStream mixedDebug( "MixedScheduler Debug Output Stream", false );
// DebugStream fullDebug( "MixedScheduler Full Debug", false );

#undef SCISHARE
#ifdef _WIN32
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif

extern SCISHARE Mutex cerrLock;
extern SCISHARE DebugStream mixedDebug;
extern SCISHARE DebugStream fullDebug;
static DebugStream stackDebug("ExceptionStack", true);
//#define HAVE_MPICH

static
void
quit( const std::string & msg = "" )
{
  if( msg != "" )
    {
      cerr << msg << "\n";
    }
  Uintah::Parallel::finalizeManager();
  exit( 1 );
}

static
void
usage( const std::string & message,
       const std::string& badarg,
       const std::string& progname)
{
#ifndef HAVE_MPICH
  int argc = 0;
  char **argv;
  argv = 0;

  // Initialize MPI so that "usage" is only printed by proc 0.
  // (If we are using MPICH, then MPI_Init() has already been called.)
  Uintah::Parallel::initializeManager( argc, argv, "" );
#endif

  if( Uintah::Parallel::getMPIRank() == 0 )
    {
      cerr << message << "\n";
      if(badarg != "")
	cerr << "Error parsing argument: " << badarg << '\n';
      cerr << "Usage: " << progname << " [options] <input_file_name>\n\n";
      cerr << "Valid options are:\n";
      cerr << "-h[elp]              : This usage information.\n";
      cerr << "-mpm                 : \n";
      cerr << "-mpmf                : option for Fracture\n";
      cerr << "-rmpm                : option for rigid MPM\n";
      cerr << "-smpm                : option for shell MPM\n";
      cerr << "-smpmice             : option for shell MPM with ICE\n";
      cerr << "-rmpmice             : option for rigid MPM with ICE\n";
      cerr << "-fmpmice             : option for Fracture MPM with ICE\n";
      cerr << "-ice                 : \n";
      cerr << "-arches              : \n";
      cerr << "-AMR                 : use AMR simulation controller\n";
      cerr << "-nthreads <#>        : Only good with MixedScheduler\n";
      cerr << "-layout NxMxO        : Eg: 2x1x1.  MxNxO must equal number\n";
      cerr << "                           of boxes you are using.\n";
      cerr << "-emit_taskgraphs     : Output taskgraph information\n";
      cerr << "-restart             : Give the checkpointed uda directory as the input file\n";
      cerr << "-combine_patches     : Give a uda directory as the input file\n";      
      cerr << "-uda_suffix <number> : Make a new uda dir with <number> as the default suffix\n";      
      cerr << "-t <timestep>        : Restart timestep (last checkpoint is default,\n\t\t\tyou can use -t 0 for the first checkpoint)\n";
      cerr << "-copy                : Copy from old uda when restarting\n";
      cerr << "-move                : Move from old uda when restarting\n";
      cerr << "-nocopy              : Default: Don't copy or move old uda timestep when\n\t\t\trestarting\n";
      cerr << "\n\n";
    }
  quit();
}

#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>

int
main( int argc, char** argv )
{
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
  int    restartTimestep = -1;
  int    udaSuffix = -1;
  string udaDir; // for restart or combine_patches
  bool   restartFromScratch = true;
  bool   restartRemoveOldDir = false;
//bool   useScheduler3 = false;
  int    numThreads = 0;
  string filename;
  string component;
  string solver;
  IntVector layout(1,1,1);

  // Checks to see if user is running an MPI version of sus.
  Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );

#ifdef HAVE_MPICH
  /*
    * Initialize MPI
    */
  //
  // If we are using MPICH, then we must call MPI_Init() before
  // parsing args... this is supposed to be fixed at some point in
  // MPICH-2.  (InitializeManager() calls MPI_Init().)
  //
  // (When using MPICH, initializeManager() uses the arg list to
  // determine whether sus is running with MPI before calling MPI_Init())
  //
  // NOTE: The main problem with calling initializeManager() before
  // parsing the args is that we don't know which "scheduler" to
  // use... the MPI or Mixed.  However, MPICH does not support
  // Thread safety, so we will just dis-allow that.
  //
  Uintah::Parallel::initializeManager( argc, argv, "" );
#endif
  /*
    * Parse arguments
    */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if( (s == "-help") || (s == "-h") ) {
      usage( "", "", argv[0]);
    } else if(s == "-AMR" || s == "-amr"){
      do_AMR=true;
    } else if(s == "-nthreads"){
      cerr << "reading number of threads\n";
      if(++i == argc){
        usage("You must provide a number of threads for -nthreads",
              s, argv[0]);
      }
      numThreads = atoi(argv[i]);
    } else if(s == "-threadmpi"){
      //used threaded mpi (this option is handled in MPI_Communicator.cc  MPI_Init_thread
    } else if(s == "-solver") {
      if(++i == argc){
        usage("You must provide a solver name for -solver",
              s, argv[0]);
      }
      solver = argv[i];
    } else if(s == "-mpi") {
      Uintah::Parallel::forceMPI();
    } else if(s == "-nompi") {
      Uintah::Parallel::forceNoMPI();
    } else if (s == "-emit_taskgraphs") {
      emit_graphs = true;
    } else if(s == "-restart") {
      restart=true;
    } else if(s == "-handle_mpi_errors") {
      // handled in Parallel.cc
    } else if(s == "-uda_suffix") {
      if (i < argc-1)
        udaSuffix = atoi(argv[++i]);
      else
        usage("You must provide a suffix number for -uda_suffix",
              s, argv[0]);
    } else if(s == "-nocopy") { // default anyway, but that's fine
      restartFromScratch = true;
    } else if(s == "-copy") {
      restartFromScratch = false;
      restartRemoveOldDir = false;
    } else if(s == "-move") {
      restartFromScratch = false;
      restartRemoveOldDir = true;
    } else if(s == "-t") {
      if (i < argc-1)
        restartTimestep = atoi(argv[++i]);
    } else if(s == "-layout") {
      if(++i == argc)
        usage("You must provide a vector arg for -layout",
              s, argv[0]);
      int ii,jj,kk;
      if(sscanf(argv[i], "%dx%dx%d", &ii, &jj, &kk) != 3)
        usage("Error parsing -layout", argv[i], argv[0]);
      layout = IntVector(ii,jj,kk);
    } else if (s[0] == '-') {
      // component name - must be the only remaining option with a hyphen
      if (component.length() > 0) {
        char errorMsg[256];
        sprintf(errorMsg, "Cannot specify both -%s and %s", component.c_str(), s.c_str());
        usage(errorMsg, argv[i], argv[0]);
      } else {
        component = s.substr(1,s.length()); // strip off the -
        if (component == "combine_patches")
          combine_patches = true;
        else if (component == "reduce_uda")
          reduce_uda = true;
      }
    } else {
      if(filename!="") {
        usage("", s, argv[0]);
      } else if( argv[i][0] == '-' ) { // Don't allow 'filename' to begin with '-'.
        usage("Error!  It appears that the filename you specified begins with a '-'.\n"
              "        This is not allowed.  Most likely there is problem with your\n"
              "        command line.",
              argv[i], argv[0]);        
      } else {
        filename = argv[i];
      }
    }
  }
 
  
  if(filename == ""){
    usage("No input file specified", "", argv[0]);
  }

  if (restart || combine_patches || reduce_uda) {
    // check if state.xml is present
    // if not do normal
    udaDir = filename;
    filename = filename + "/input.xml";
  }

  if (!Uintah::Parallel::usingMPI()) {
    TAU_PROFILE_SET_NODE(0);
  }

  #ifdef USE_VAMPIR
  VTsetup();
  #endif

#ifndef _WIN32
  SimulationController::start_addr = (char*)sbrk(0);
  mode_t mask_gor = 0022;
  umask(mask_gor);
#endif
  Thread::disallow_sgi_OpenGL_page0_sillyness();

#ifndef HAVE_MPICH
  // If regular MPI, then initialize after parsing the args...
  Uintah::Parallel::initializeManager( argc, argv, "");
#endif

  bool thrownException = false;
  
  
 if( Uintah::Parallel::getMPIRank() == 0 ) {
    cerr << "Date:    ";
    system("date");         // helpful for cleaning out old stale udas
    cerr << "Machine: ";
    system("hostname");
  }
  
  //______________________________________________________________________
  // Create the components
  try {

    char * st = getenv( "INITIAL_SLEEP_TIME" );
    if( st != 0 ){
      int sleepTime = atoi( st );
      cout << "SLEEPING FOR " << sleepTime 
           << " SECONDS TO ALLOW DEBUGGER ATTACHMENT\n";
      cout << "PID for rank " << Uintah::Parallel::getMPIRank() << " is " << getpid() << "\n";
      Time::waitFor( (double)sleepTime );
    }

    //__________________________________
    // Read input file
    ProblemSpecInterface* reader = scinew ProblemSpecReader(filename);
    ProblemSpecP ups = reader->readInputFile();
    if(ups->getNodeName() != "Uintah_specification")
      throw ProblemSetupException("Input file is not a Uintah specification", __FILE__, __LINE__);


    //__________________________________
    // grab AMR from the ups file if not specified on the command line
    if (!do_AMR)
      do_AMR = (bool) ups->findBlock("AMR");

    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
    SimulationController* ctl = 
      scinew AMRSimulationController(world, do_AMR, ups);

    Regridder* reg = 0;
    if(do_AMR) {
      reg = RegridderFactory::create(ups, world);
      if (reg)
        ctl->attachPort("regridder", reg);
    }

    //__________________________________
    // Solver
    SolverInterface* solve = 0;
    solve = SolverFactory::create(ups, world, solver);

    //__________________________________
    // Component
    // try to make it from the command line first, then look in ups file
    UintahParallelComponent* comp = ComponentFactory::create(ups, world, do_AMR, component, udaDir);
    SimulationInterface* sim = dynamic_cast<SimulationInterface*>(comp);

    if (combine_patches || reduce_uda) {
      // the ctl will do nearly the same thing for combinePatches and reduceUda
      ctl->doCombinePatches(udaDir, reduce_uda); // true for reduce_uda, false for combine_patches
    }
    
    ctl->attachPort("sim", sim);
    comp->attachPort("solver", solve);
    
    //__________________________________
    //  Model
    ModelMaker* modelmaker = scinew ModelFactory(world);
    comp->attachPort("modelmaker", modelmaker);

    //__________________________________
    // Load balancer
    LoadBalancer* bal;
    UintahParallelComponent* lb; // to add scheduler as a port
    LoadBalancerCommon* lbc = LoadBalancerFactory::create(ups, world);
    lb = lbc;
    lb->attachPort("sim", sim);
    bal = lbc;
    
    //__________________________________
    // Output
    DataArchiver* dataarchiver = scinew DataArchiver(world, udaSuffix);
    Output* output = dataarchiver;
    ctl->attachPort("output", output);
    dataarchiver->attachPort("load balancer", bal);
    comp->attachPort("output", output);
    dataarchiver->attachPort("sim", sim);
    
    //__________________________________
    // Scheduler
    SchedulerCommon* sched = SchedulerFactory::create(ups, world, output);
    Scheduler* sch = sched;
    sched->attachPort("load balancer", bal);
    ctl->attachPort("scheduler", sched);
    lb->attachPort("scheduler", sched);
    comp->attachPort("scheduler", sched);
    sch->addReference();
    
    if (emit_graphs) sch->doEmitTaskGraphDocs();
    
    /*
     * Start the simulation controller
     */
    if (restart) {
      ctl->doRestart(udaDir, restartTimestep, restartFromScratch, restartRemoveOldDir);
    }
    
    ctl->run();
    delete ctl;
    

    sch->removeReference();
    delete sch;
    if (reg) delete reg;
    delete bal;
    delete sim;
    delete solve;
    delete output;
    delete reader;
    delete modelmaker;
  } catch (Exception& e) {
    
    cerrLock.lock();
    cout << Uintah::Parallel::getMPIRank() << " Caught exception: " << e.message() << '\n';
    if(e.stackTrace())
      stackDebug << "Stack trace: " << e.stackTrace() << '\n';
    cerrLock.unlock();
    thrownException = true;
  } catch (std::exception e){
    
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught std exception: " << e.what() << '\n';
    cerrLock.unlock();
    thrownException = true;
    
  } catch(...){
    
    cerrLock.lock();
    cerr << Uintah::Parallel::getMPIRank() << " Caught unknown exception\n";
    cerrLock.unlock();
    thrownException = true;
  }
  
  Uintah::TypeDescription::deleteAll();
  
  /*
   * Finalize MPI
   */
  Uintah::Parallel::finalizeManager(thrownException?
				    Uintah::Parallel::Abort:
				    Uintah::Parallel::NormalShutdown);

  if (thrownException) {
    if( Uintah::Parallel::getMPIRank() == 0 ) {
      cout << "An exception was thrown... Goodbye.\n";
    }
    Thread::exitAll(1);
  }
  
  if( Uintah::Parallel::getMPIRank() == 0 ) {
    cout << "Sus: going down successfully\n";
  }
}

extern "C" {
  void dgesvd_() {
    cerr << "Error: dgesvd called!\n";
    Thread::exitAll(1);
  }

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
