
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
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MixedScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NirvanaLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif

#ifdef USE_VAMPIR
#include <Packages/Uintah/Core/Parallel/Vampir.h>
#endif

#if HAVE_IEEEFP_H
#include <ieeefp.h>
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
      cerr << "-ice                 : \n";
      cerr << "-arches              : \n";
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

    /*
     * Default values
     */
    bool   do_mpm=false;
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

    if(!(do_ice || do_arches || do_mpm || do_impmpm || do_burger || do_poisson1 || do_poisson2 || do_poisson3 || do_simplecfd || combine_patches)){
	usage( "You need to specify -arches, -ice, or -mpm", "", argv[0]);
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

	// Connect a MPM module if applicable
	SimulationInterface* sim = 0;
	if(do_mpm && do_ice){
	  MPMICE* mpmice = scinew MPMICE(world);
	  mpmice->attachPort("output", output);
	  sim = mpmice;
	} else if(do_mpm && do_arches){
	  sim = scinew MPMArches(world);
	} else if(do_mpm){
	  sim = scinew SerialMPM(world);
	} else if(do_impmpm){
	  sim = scinew ImpMPM(world);
	} else if(do_arches){
	  sim = scinew Arches(world);
	} else if(do_ice) {
	  ICE* ice = scinew ICE(world);
	  ice->attachPort("output", output);
	  sim = ice;
	} else if(do_burger){
	  sim = scinew Burger(world);
	} else if(do_poisson1){
	  sim = scinew Poisson1(world);
	} else if(do_poisson2){
	  sim = scinew Poisson2(world);
	} else if(do_poisson3){
	  sim = scinew Poisson3(world);
	} else if(do_simplecfd){
	  sim = scinew SimpleCFD(world);
	} else if (combine_patches) {
	  sim = scinew PatchCombiner(world, udaDir);
	  ctl->doCombinePatches(udaDir);
	} else {
	  usage("You need to specify a simulation: -arches, -ice, -mpm, "
		"-impm -mpmice, -mpmarches, -burger, -poisson1, -poisson2, or -poisson3",
		"", argv[0]);
	}

	ctl->attachPort("sim", sim);

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
	} else {
	   bal = 0;
	   quit( "Unknown load balancer: " + loadbalancer );
	}

	// Scheduler
	Scheduler* sch;
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

	char * st = getenv( "INITIAL_SLEEP_TIME" );
	if( st != 0 ){
	  int sleepTime = atoi( st );
	  cerr << "SLEEPING FOR " << sleepTime 
	       << " SECONDS TO ALLOW DEBUGGER ATTACHMENT\n";
	  cerr << "PID is " << getpid() << "\n";
	  sleep( sleepTime );
	}

	if (restart) {
	  ctl->doRestart(udaDir, restartTimestep,
			 restartFromScratch, restartRemoveOldDir);
	}

	ctl->run();

	delete sim;
	delete output;
	delete bal;
	delete ctl;
	delete reader;
	sch->removeReference();
	delete sch;
    } catch (Exception& e) {

      cerrLock.lock();
      cerr << "Caught exception: " << e.message() << '\n';
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
      cerr << "Caught std exception: " << e.what() << '\n';
      cerrLock.unlock();
      //Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
      //abort();
      thrownException = true;

    } catch(...){

      cerrLock.lock();
      cerr << "Caught unknown exception\n";
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
