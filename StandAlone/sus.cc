
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
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArches.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MPIScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MixedScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/NirvanaLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Core/Exceptions/Exception.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/DebugStream.h>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#ifdef __sgi
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
Mutex cerrLock( "cerr lock" );
DebugStream mixedDebug( "MixedScheduler Debug Output Stream", false );
DebugStream fullDebug( "MixedScheduler Full Debug", false );

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
  if( !Uintah::Parallel::usingMPI() || 
      ( Uintah::Parallel::usingMPI() &&
	Uintah::Parallel::getRootProcessorGroup()->myrank() == 0 ) )
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
      cerr << "-t <timestep>        : Restart timestep (last checkpoint is default)\n";
      cerr << "-copy                : Copy from old uda when restarting\n";
      cerr << "-move                : Move from old uda when restarting\n";
      cerr << "-nocopy              : Default: Don't copy or move old uda timesteps when restarting\n";
      cerr << "\n\n";
    }
  quit();
}

int
main(int argc, char** argv)
{
  TAU_PROFILE_INIT(argc,argv);
  // Causes buserr for some reason:
  //TAU_PROFILE("main()", "void (int, char **)", TAU_DEFAULT);

#if HAVE_IEEEFP_H
    fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);
#endif

    /*
     * Default values
     */
    bool   do_mpm=false;
    bool   do_arches=false;
    bool   do_ice=false;
    bool   emit_graphs=false;
    bool   restart=false;
    int    restartTimestep = -1;
    string restartFromDir;
    bool   restartFromScratch = true;
    bool   restartRemoveOldDir = false;
    int    numThreads = 0;
    string filename;
    string scheduler;
    string loadbalancer;
    IntVector layout(1,1,1);

    /*
     * Initialize MPI
     */
    Uintah::Parallel::initializeManager( argc, argv, scheduler );
    #ifdef USE_VAMPIR
    VTsetup();
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
	} else if(s == "-arches"){
	    do_arches=true;
	} else if(s == "-ice"){
	    do_ice=true;
	} else if(s == "-mpmice"){
	    do_ice=true;
	    do_mpm=true;
	} else if(s == "-mpmarches"){
	    do_arches=true;
	    do_mpm=true;
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
	} else if(s.substr(0,3) == "-p4") {
	   // mpich - skip the rest
	   break;
	} else if (s == "-emit_taskgraphs") {
	   emit_graphs = true;
	} else if(s == "-restart") {
	   restart=true;
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

    if (restart) {
       restartFromDir = filename;
       filename = filename + "/input.xml";
    }

    /*
     * Check for valid argument combinations
     */
    if(do_ice && do_arches){
	usage( "ICE and Arches do not work together", "", argv[0]);
    }

    if(!(do_ice || do_arches || do_mpm)){
	usage( "You need to specify -arches, -ice, or -mpm", "", argv[0]);
    }

    SimulationController::start_addr = (char*)sbrk(0);
    Thread::disallow_sgi_OpenGL_page0_sillyness();

    if(scheduler == ""){
       if(Uintah::Parallel::usingMPI()){
	  scheduler="MPIScheduler"; // Default for parallel runs
	  loadbalancer="SimpleLoadBalancer";
	  Uintah::Parallel::noThreading();
       } else {
	  scheduler="SingleProcessorScheduler"; // Default for serial runs
	  loadbalancer="SingleProcessorLoadBalancer";
       }
    }

    bool thrownException = false;
    
    /*
     * Create the components
     */
    try {
	const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
	SimulationController* sim = scinew SimulationController(world);

	// Reader
	ProblemSpecInterface* reader = scinew ProblemSpecReader(filename);
	sim->attachPort("problem spec", reader);

	// Connect a MPM module if applicable
	MPMInterface* mpm = 0;
	if(do_mpm){
	  mpm = scinew SerialMPM(world);
	  sim->attachPort("mpm", mpm);
	}

	// Connect a CFD module if applicable
	CFDInterface* cfd = 0;
	ICE* ice = 0;
	if(do_arches){
	    cfd = scinew Arches(world);
	}
	if(do_ice){
	    cfd = ice = scinew ICE(world);
	}
	if(cfd)
	    sim->attachPort("cfd", cfd);

	// Connect an MPMICE module if do_mpm and do_ice are both true
	MPMCFDInterface* mpmcfd = 0;
	if(do_mpm && do_ice){
	    mpmcfd = scinew MPMICE(world);
	}
	if(do_mpm && do_arches){
	    mpmcfd = scinew MPMArches(world);
	}
	if (mpmcfd)
	  sim->attachPort("mpmcfd", mpmcfd);
	// Output
	Output* output = scinew DataArchiver(world);
	sim->attachPort("output", output);

	// This port is only needed for knowing when to spit out debugging
	// information.
	if (ice)
	  ice->attachPort("output", output);

	if(world->myrank() == 0){
	   cerr << "Using scheduler: " << scheduler << " and load balancer: " << loadbalancer << '\n';
	}

	// Load balancer
	LoadBalancer* bal;
	if(loadbalancer == "SingleProcessorLoadBalancer"){
	   bal = scinew SingleProcessorLoadBalancer(world);
	} else if(loadbalancer == "RoundRobinLoadBalancer" || loadbalancer == "RoundRobin" || loadbalancer == "roundrobin"){
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
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	   sch=sched;
	} else if(scheduler == "SimpleScheduler"){
	   SimpleScheduler* sched = 
	      scinew SimpleScheduler(world, output);
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	   sch=sched;
	} else if(scheduler == "MPIScheduler"){
	   MPIScheduler* sched =
	      scinew MPIScheduler(world, output);
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	   sch=sched;
	} else if(scheduler == "MixedScheduler"){
	  if( numThreads > 0 ){
	    if( Uintah::Parallel::getMaxThreads() == 1 ){
	      Uintah::Parallel::setMaxThreads( numThreads );
	    }
	  }
	  MixedScheduler* sched =
	    scinew MixedScheduler(world, output);
	  sim->attachPort("scheduler", sched);
	  sched->attachPort("load balancer", bal);
	  sch=sched;
	} else if(scheduler == "NullScheduler"){
	   NullScheduler* sched =
	      scinew NullScheduler(world, output);
	   sim->attachPort("scheduler", sched);
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
	  sim->doRestart(restartFromDir, restartTimestep,
			 restartFromScratch, restartRemoveOldDir);
	}
	sim->run();

	delete mpm;
	delete cfd;
	delete mpmcfd;
	delete output;
	delete bal;
	delete sim;
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

    /*
     * Finalize MPI
     */
    Uintah::Parallel::finalizeManager(thrownException?Uintah::Parallel::Abort:Uintah::Parallel::NormalShutdown);

    if (thrownException) {
      Thread::exitAll(1);
    }

    //Thread::exitAll(0);
}
