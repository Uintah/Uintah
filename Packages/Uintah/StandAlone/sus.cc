
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
#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>
#include <Packages/Uintah/CCA/Components/SimulationController/SimpleSimulationController.h>
#include <Packages/Uintah/CCA/Components/SimulationController/AMRSimulationController.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/FractureMPM.h> // for Fracture
#include <Packages/Uintah/CCA/Components/MPM/RigidMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ShellMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ImpMPM.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArches.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson1.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson2.h>
#include <Packages/Uintah/CCA/Components/Examples/Burger.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson3.h>
#include <Packages/Uintah/CCA/Components/Examples/SimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Examples/AMRSimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Models/ModelFactory.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MixedScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NirvanaLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/ParticleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Solvers/CGSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/DirectSolve.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolver.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>

#include <sci_config.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif

#ifdef USE_VAMPIR
#include <Packages/Uintah/Core/Parallel/Vampir.h>
#endif

#if HAVE_IEEEFP_H
#include <ieeefp.h>
#endif
#if 0
#include <fenv.h>
#endif

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

// Debug: Used to sync cerr so it is readable (when output by
// multiple threads at the same time)
// Mutex cerrLock( "cerr lock" );
// DebugStream mixedDebug( "MixedScheduler Debug Output Stream", false );
// DebugStream fullDebug( "MixedScheduler Full Debug", false );

extern Mutex cerrLock;
extern DebugStream mixedDebug;
extern DebugStream fullDebug;

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
      cerr << "-mpmf                : \n";  // option for Fracture
      cerr << "-rmpm                : \n";  // option for rigid MPM
      cerr << "-smpm                : \n";  // option for shell MPM
      cerr << "-ice                 : \n";
      cerr << "-arches              : \n";
      cerr << "-AMR                 : use AMR simulation controller\n";
      cerr << "-nthreads <#>        : Only good with MixedScheduler\n";
      cerr << "-scheduler <name>    : Don't specify, use system default!\n";
      cerr << "-loadbalancer <name> : Usually use system default.\n";
      cerr << "          NirvanaLoadBalancer [or NLB for short]\n";
      cerr << "-layout NxMxO        : Eg: 2x1x1.  MxNxO must equal number\n";
      cerr << "                           of boxes you are using.\n";
      cerr << "-emit_taskgraphs     : Output taskgraph information\n";
      cerr << "-restart             : Give the checkpointed uda directory as the input file\n";
      cerr << "-combine_patches     : Give a uda directory as the input file\n";      
      cerr << "-t <timestep>        : Restart timestep (last checkpoint is default,\n\t\t\tyou can use -t 0 for the first checkpoint)\n";
      cerr << "-copy                : Copy from old uda when restarting\n";
      cerr << "-move                : Move from old uda when restarting\n";
      cerr << "-nocopy              : Default: Don't copy or move old uda timestep when\n\t\t\trestarting\n";
      cerr << "\n\n";
    }
  quit();
}

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

  TAU_PROFILE_INIT(argc,argv);
  
#if HAVE_IEEEFP_H
    fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);
#endif
#if 0
    feenableexcept(FE_INVALID|FE_OVERFLOW|FE_DIVBYZERO);
#endif

    /*
     * Default values
     */
    bool   do_mpm=false;
    bool   do_mpmf=false;      // for Fracture
    bool   do_rmpm=false;      // for rigid MPM
    bool   do_smpm=false;      // for shell MPM
    bool   do_impmpm=false;
    bool   do_arches=false;
    bool   do_ice=false;
    bool   do_burger=false;
    bool   do_poisson1=false;
    bool   do_poisson2=false;
    bool   do_poisson3=false;
    bool   do_simplecfd=false;
    bool   do_AMR=false;
    bool   emit_graphs=false;
    bool   restart=false;
    bool   combine_patches=false;
    int    restartTimestep = -1;
    string udaDir; // for restart or combine_patches
    bool   restartFromScratch = true;
    bool   restartRemoveOldDir = false;
    int    numThreads = 0;
    string filename;
    string scheduler;
    string loadbalancer;
    string solver = "CGSolver";
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
    Uintah::Parallel::initializeManager( argc, argv, scheduler );
#endif
    /*
     * Parse arguments
     */
    for(int i=1;i<argc;i++){
	string s=argv[i];
	if( (s == "-help") || (s == "-h") ) {
	  usage( "", "", argv[0]);
	} else if(s == "-mpm"){
	  do_mpm=true;
	} else if(s == "-mpmf"){    // for Fracture
          do_mpmf=true;
	} else if(s == "-rmpm"){
          do_rmpm=true;
	} else if(s == "-smpm"){
          do_smpm=true;
	} else if(s == "-impm"){
	  do_impmpm=true;
	} else if(s == "-arches"){
	    do_arches=true;
	} else if(s == "-ice"){
	    do_ice=true;
	} else if(s == "-mpmice"){
	    do_ice=true;
	    do_mpm=true;
	} else if(s == "-burger"){
	    do_burger=true;
	} else if(s == "-poisson1"){
	    do_poisson1=true;
	} else if(s == "-poisson2"){
	    do_poisson2=true;
	} else if(s == "-poisson3"){
	    do_poisson3=true;
	} else if(s == "-scfd" || s == "-simplecfd"){
	    do_simplecfd=true;
	} else if(s == "-mpmarches"){
	    do_arches=true;
	    do_mpm=true;
	} else if(s == "-AMR" || s == "-amr"){
	    do_AMR=true;
	} else if(s == "-nthreads"){
	  cerr << "reading number of threads\n";
	  if(++i == argc){
	    usage("You must provide a number of threads for -nthreads",
		  s, argv[0]);
	  }
	  numThreads = atoi(argv[i]);
	} else if(s == "-scheduler"){
	  if(++i == argc){
	    usage("You must provide a scheduler name for -scheduler",
		  s, argv[0]);
	  }
	  scheduler = argv[i]; 
	} else if(s == "-loadbalancer"){
	  if(++i == argc){
	    usage("You must provide a load balancer name for -loadbalancer",
		  s, argv[0]);
	  }
	  loadbalancer = argv[i];
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
	} else if(s == "-combine_patches") {
	   combine_patches=true;	   
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
	} else if(s == "-solver") {
	  if(++i == argc){
	    usage("You must provide a solver name for -solver", s, argv[0]);
	  }
	  solver = argv[i];
	} else if(s == "-layout") {
	  if(++i == argc)
	    usage("You must provide a vector arg for -layout",
		  s, argv[0]);
	  int ii,jj,kk;
	  if(sscanf(argv[i], "%dx%dx%d", &ii, &jj, &kk) != 3)
	    usage("Error parsing -layout", argv[i], argv[0]);
	  layout = IntVector(ii,jj,kk);
	} else {
	    if(filename!="")
		usage("", s, argv[0]);
	    else
		filename = argv[i];
	}
    }

    if(filename == ""){
      usage("No input file specified", "", argv[0]);
    }

    if (restart || combine_patches) {
       udaDir = filename;
       filename = filename + "/input.xml";
    }

    #ifdef USE_VAMPIR
    VTsetup();
    #endif

    /*
     * Check for valid argument combinations
     */
    if(do_ice && do_arches){
	usage( "ICE and Arches do not work together", "", argv[0]);
    }

    if(!(do_ice || do_arches || do_mpm || do_mpmf  || do_rmpm || do_smpm || do_impmpm || do_burger || do_poisson1 || do_poisson2 || do_poisson3 || do_simplecfd || combine_patches)){
	usage( "You need to specify -arches, -ice, -mpmf, -rmpm, -smpm or -mpm", "", argv[0]);
    }

    SimulationController::start_addr = (char*)sbrk(0);
    Thread::disallow_sgi_OpenGL_page0_sillyness();

    if(scheduler == ""){
       if(Uintah::Parallel::usingMPI()){
	  scheduler="MPIScheduler"; // Default for parallel runs
	  if(loadbalancer == "")
	    loadbalancer="SimpleLoadBalancer";
	  Uintah::Parallel::noThreading();
       } else {
	  TAU_PROFILE_SET_NODE(0);
	  scheduler="SingleProcessorScheduler"; // Default for serial runs
	  if(loadbalancer == "")
	    loadbalancer="SingleProcessorLoadBalancer";
       }
    }

#ifndef HAVE_MPICH
    // If regular MPI, then initialize after parsing the args...
    Uintah::Parallel::initializeManager( argc, argv, scheduler );
#endif

    bool thrownException = false;
    
    /*
     * Create the components
     */
    try {

	char * st = getenv( "INITIAL_SLEEP_TIME" );
	if( st != 0 ){
	  int sleepTime = atoi( st );
	  cerr << "SLEEPING FOR " << sleepTime 
	       << " SECONDS TO ALLOW DEBUGGER ATTACHMENT\n";
	  cerr << "PID is " << getpid() << "\n";
	  sleep( sleepTime );
	}

	const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
	SimulationController* ctl;
	if(do_AMR)
	   ctl = scinew AMRSimulationController(world);
        else
	   ctl = scinew SimpleSimulationController(world);

	// Reader
	ProblemSpecInterface* reader = scinew ProblemSpecReader(filename);
	ctl->attachPort("problem spec", reader);

	// Output
	Output* output = scinew DataArchiver(world);
	ctl->attachPort("output", output);

	// Solver
	SolverInterface* solve = 0;
	if(solver == "CGSolver") {
	  solve = new CGSolver(world);
	} else if(solver == "DirectSolve") {
	  solve = new DirectSolve(world);
	} else if(solver == "HypreSolver" || solver == "hypre"){
#if HAVE_HYPRE
	  solve = new HypreSolver2(world);
#else
	  cerr << "Hypre solver not available, hypre not configured\n";
	  exit(1);
#endif
	} else {
         cerr <<"\n\n__________________________________\n";
         cerr << "sus command line error, unknown solver: " << solver << '\n';
         cerr << "Valid Solvers: CGSolver, DirectSolve, hypre \n";
	  exit(1);
	}

	// Connect a MPM module if applicable
	SimulationInterface* sim = 0;
	UintahParallelComponent* comp = 0;
	if(do_mpm && do_ice){
	  MPMICE* mpmice = scinew MPMICE(world);
	  sim = mpmice;
	  comp = mpmice;
	} else if(do_mpm && do_arches){
	  MPMArches* mpmarches = scinew MPMArches(world);
	  sim = mpmarches;
	  comp = mpmarches;
	} else if(do_mpm){
	  SerialMPM* mpm = scinew SerialMPM(world);
	  sim = mpm;
	  comp = mpm;
	} else if(do_mpmf){
          FractureMPM* mpmf = scinew FractureMPM(world); // for Fracture
	  sim = mpmf;
	  comp = mpmf;
	} else if(do_rmpm){
          RigidMPM* rmpm = scinew RigidMPM(world);
	  sim = rmpm;
	  comp = rmpm;
	} else if(do_smpm){
          ShellMPM* smpm = scinew ShellMPM(world);
	  sim = smpm;
	  comp = smpm;
	} else if(do_impmpm){
	  ImpMPM* impm = scinew ImpMPM(world);
	  sim = impm;
	  comp = impm;
	} else if(do_arches){
	  Arches* arches = scinew Arches(world);
	  sim = arches;
	  comp = arches;
	} else if(do_ice) {
	  ICE* ice = scinew ICE(world);
	  sim = ice;
	  comp = ice;
	} else if(do_burger){
	  Burger* burger = scinew Burger(world);
	  sim = burger;
	  comp = burger;
	} else if(do_poisson1){
	  Poisson1* poisson1 = scinew Poisson1(world);
	  sim = poisson1;
	  comp = poisson1;
	} else if(do_poisson2){
	  Poisson2* poisson2 = scinew Poisson2(world);
	  sim = poisson2;
	  comp = poisson2;
	} else if(do_poisson3){
	  Poisson3* poisson3 = scinew Poisson3(world);
	  sim = poisson3;
	  comp = poisson3;
	} else if(do_simplecfd){
	  SimpleCFD* simplecfd;
	  if(do_AMR)
	    simplecfd = scinew AMRSimpleCFD(world);
	  else
	    simplecfd = scinew SimpleCFD(world);
	  sim = simplecfd;
	  comp = simplecfd;
	} else if (combine_patches) {
	  PatchCombiner* pc = scinew PatchCombiner(world, udaDir);
	  sim = pc;
	  comp = pc;
	  ctl->doCombinePatches(udaDir);
	} else {
	  usage("You need to specify a simulation: -arches, -ice, -mpm, "
		"-impm -mpmice, -mpmarches, -burger, -poisson1, -poisson2, or -poisson3",
		"", argv[0]);
	}

	ctl->attachPort("sim", sim);
	comp->attachPort("solver", solve);
	comp->attachPort("output", output);

	ModelMaker* modelmaker = scinew ModelFactory(world);
	comp->attachPort("modelmaker", modelmaker);

	if(world->myrank() == 0){
	   cerr << "Using scheduler: " << scheduler 
		<< " and load balancer: " << loadbalancer << '\n';
	}

	// Load balancer
	LoadBalancer* bal;
	if(loadbalancer == "SingleProcessorLoadBalancer"){
	   bal = scinew SingleProcessorLoadBalancer(world);
	} else if(loadbalancer == "RoundRobinLoadBalancer" || 
		  loadbalancer == "RoundRobin" || 
		  loadbalancer == "roundrobin"){
	   bal = scinew RoundRobinLoadBalancer(world);
	} else if(loadbalancer == "SimpleLoadBalancer") {
	   bal = scinew SimpleLoadBalancer(world);
	} else if( (loadbalancer == "NirvanaLoadBalancer") ||
		   (loadbalancer == "NLB") ) {
	  bal = scinew NirvanaLoadBalancer(world, layout);
	} else if( (loadbalancer == "ParticleLoadBalancer") ||
		   (loadbalancer == "PLB") ) {
	  //bal = 0;
	  bal = scinew ParticleLoadBalancer(world);
	} else {
	   bal = 0;
	   quit( "Unknown load balancer: " + loadbalancer );
	}

	// Scheduler
	Scheduler * sch = 0;
	if(scheduler == "SingleProcessorScheduler"){
	   SingleProcessorScheduler* sched = 
	      scinew SingleProcessorScheduler(world, output);
	   ctl->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	   sch=sched;
	} else if(scheduler == "SimpleScheduler"){
	   SimpleScheduler* sched = 
	      scinew SimpleScheduler(world, output);
	   ctl->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	   sch=sched;
	} else if(scheduler == "MPIScheduler"){
	   MPIScheduler* sched =
	      scinew MPIScheduler(world, output);
	   ctl->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	   sch=sched;
	} else if(scheduler == "MixedScheduler"){
#ifdef HAVE_MPICH
	  cerr << "MPICH does not support the MixedScheduler.  Exiting\n";
	  Thread::exitAll(1);	  
#endif
	  if( numThreads > 0 ){
	    if( Uintah::Parallel::getMaxThreads() == 1 ){
	      Uintah::Parallel::setMaxThreads( numThreads );
	    }
	  }
	  MixedScheduler* sched =
	    scinew MixedScheduler(world, output);
	  ctl->attachPort("scheduler", sched);
	  sched->attachPort("load balancer", bal);
	  sch=sched;
	} else if(scheduler == "NullScheduler"){
	   NullScheduler* sched =
	      scinew NullScheduler(world, output);
	   ctl->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	   sch=sched;
	} else {
	   quit( "Unknown scheduler: " + scheduler );
	}

	sch->addReference();
	if (emit_graphs) sch->doEmitTaskGraphDocs();

	/*
	 * Start the simulation controller
	 */

	if (restart) {
	  ctl->doRestart(udaDir, restartTimestep,
			 restartFromScratch, restartRemoveOldDir);
	}

	ctl->run();


	sch->removeReference();
	delete sch;
	delete bal;
	delete sim;
	delete solve;
	delete output;
	delete reader;
	delete ctl;
	delete modelmaker;
    } catch (Exception& e) {

      cerrLock.lock();
      cerr << Uintah::Parallel::getMPIRank() << " Caught exception: " << e.message() << '\n';
      if(e.stackTrace())
	cerr << "Stack trace: " << e.stackTrace() << '\n';
      cerrLock.unlock();
      
      // Dd: I believe that these cause error messages
      // to be lost when the program dies...
      //Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
      //abort();
      thrownException = true;
    } catch (std::exception e){

      cerrLock.lock();
      cerr << Uintah::Parallel::getMPIRank() << " Caught std exception: " << e.what() << '\n';
      cerrLock.unlock();
      //Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
      //abort();
      thrownException = true;

    } catch(...){

      cerrLock.lock();
      cerr << Uintah::Parallel::getMPIRank() << " Caught unknown exception\n";
      cerrLock.unlock();
      //Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
      //abort();
      thrownException = true;
    }

    // Shutdown XML crap
    XMLPlatformUtils::Terminate();
    Uintah::TypeDescription::deleteAll();

    /*
     * Finalize MPI
     */
    Uintah::Parallel::finalizeManager(thrownException?
				      Uintah::Parallel::Abort:
				      Uintah::Parallel::NormalShutdown);

    if (thrownException) {
      if( Uintah::Parallel::getMPIRank() == 0 )
	{
	  cout << "An exception was thrown... Goodbye.\n";
	}
      Thread::exitAll(1);
    }

    if( Uintah::Parallel::getMPIRank() == 0 )
      {
	cout << "Sus: going down successfully\n";
      }

    //Thread::exitAll(0);
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
