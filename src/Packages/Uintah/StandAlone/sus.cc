
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
#include <Packages/Uintah/CCA/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/Exception.h>

#ifdef USE_VAMPIR
#include <Packages/Uintah/Core/Parallel/Vampir.h>
#endif

#if HAVE_FPSETMASK
#include <ieeefp.h>
#endif

#include <iostream>
#include <string>
#include <vector>


using namespace SCIRun;
using namespace Uintah;
using namespace std;

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
      cerr << "-mpm                 : \n";
      cerr << "-ice                 : \n";
      cerr << "-arches              : \n";
      cerr << "-nthreads <#>        : \n";
      cerr << "-scheduler <name>    : Don't specify, use system default!\n";
      cerr << "-loadbalancer <name> : Don't specify, use system default!\n";
      cerr << "\n\n";
    }
  quit();
}

int
main(int argc, char** argv)
{
    TAU_PROFILE("main()", "void (int, char **)", TAU_DEFAULT);
    TAU_PROFILE_INIT(argc,argv);

    SimulationController::start_addr = (char*)sbrk(0);
    Thread::disallow_sgi_OpenGL_page0_sillyness();
  
    /*
     * Initialize MPI
     */
    Uintah::Parallel::initializeManager(argc, argv);
    #ifdef USE_VAMPIR
    VTsetup();
    #endif

#if HAVE_FPSETMASK
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
    bool   restartRemoveOldDir=false;
    int    numThreads = 0;
    string filename;
    string scheduler;
    string loadbalancer;

    /*
     * Parse arguments
     */
    for(int i=1;i<argc;i++){
	string s=argv[i];
	if(s == "-mpm"){
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
	} else if(s == "-nocopy") {
	   restartRemoveOldDir = true;
	} else if(s == "-copy") { // default anyway, but that's fine
	   restartRemoveOldDir = false;
	} else if(s == "-t") {
           if (i < argc-1)
	      restartTimestep = atoi(argv[++i]);
	}
	else {
	    if(filename!="")
		usage("", s, argv[0]);
	    else
		filename = argv[i];
	}
    }

    if(filename == ""){
      usage("No input file specified", "", argv[0]);
    }

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
	if(do_arches){
	    cfd = scinew Arches(world);
	}
	if(do_ice){
	    cfd = scinew ICE(world);
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
	  sleep( sleepTime );
	}

	if (restart) {
	  sim->doRestart(restartFromDir, restartTimestep,
			 restartRemoveOldDir);
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
	cerr << "Caught exception: " << e.message() << '\n';
	if(e.stackTrace())
	   cerr << "Stack trace: " << e.stackTrace() << '\n';
	// Dd: I believe that these cause error messages
	// to be lost when the program dies...
	//Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
	//abort();
	thrownException = true;
    } catch (std::exception e){
        cerr << "Caught std exception: " << e.what() << '\n';
	//Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
	//abort();
	thrownException = true;
    } catch(...){
	cerr << "Caught unknown exception\n";
	//Uintah::Parallel::finalizeManager(Uintah::Parallel::Abort);
	//abort();
	thrownException = true;
    }

    // Shutdown XML crap
    XMLPlatformUtils::Terminate();

    /*
     * Finalize MPI
     */
    Uintah::Parallel::finalizeManager();

    if (thrownException) {
      Thread::exitAll(1);
    }
}
