
/*
 *  sus.cc: Standalone Packages/Uintah Simulation - a bare-bones uintah simulation
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

#include <Packages/Uintah/Parallel/Parallel.h>
#include <Uintah/Core/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Uintah/Core/CCA/Components/SimulationController/SimulationController.h>
#include <Uintah/Core/CCA/Components/MPM/SerialMPM.h>
#include <Uintah/Core/CCA/Components/Arches/Arches.h>
#include <Uintah/Core/CCA/Components/ICE/ICE.h>
#include <Uintah/Core/CCA/Components/MPMICE/MPMICE.h>
#include <Uintah/Core/CCA/Components/Schedulers/SingleProcessorScheduler.h>
#include <Uintah/Core/CCA/Components/Schedulers/MPIScheduler.h>
#include <Uintah/Core/CCA/Components/Schedulers/MixedScheduler.h>
#include <Uintah/Core/CCA/Components/Schedulers/NullScheduler.h>
#include <Uintah/Core/CCA/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Uintah/Core/CCA/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Uintah/Core/CCA/Components/Schedulers/SimpleLoadBalancer.h>
#include <Uintah/Core/CCA/Components/DataArchiver/DataArchiver.h>
#include <Packages/Uintah/Interface/DataWarehouse.h>
#include <Packages/Uintah/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/Exception.h>

#ifdef USE_VAMPIR
#include <Packages/Uintah/Parallel/Vampir.h>
#endif

#if HAVE_FPSETMASK
#include <ieeefp.h>
#endif

#include <iostream>
#include <string>
#include <vector>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

void quit( const std::string & msg = "" )
{
  if( msg != "" )
    {
      cerr << msg << "\n";
    }
  Parallel::finalizeManager();
  exit( 1 );
}

void usage( const std::string & message,
	    const std::string& badarg,
	    const std::string& progname)
{
  if( !Parallel::usingMPI() || 
      ( Parallel::usingMPI() &&
	Parallel::getRootProcessorGroup()->myrank() == 0 ) )
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

int main(int argc, char** argv)
{
    /*
     * Initialize MPI
     */
    Parallel::initializeManager(argc, argv);

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

    if(scheduler == ""){
       if(Parallel::usingMPI()){
	  scheduler="MPIScheduler"; // Default for parallel runs
	  loadbalancer="RoundRobinLoadBalancer";
	  Parallel::noThreading();
       } else {
	  scheduler="SingleProcessorScheduler"; // Default for serial runs
	  loadbalancer="SingleProcessorLoadBalancer";
       }
    }

    /*
     * Check for valid argument combinations
     */
    if(do_mpm && do_arches){
	usage( "MPM doesn't yet work with ICE/Arches", "", argv[0]);
    }
    if(do_ice && do_arches){
	usage( "ICE and Arches do not work together", "", argv[0]);
    }

    if(!(do_ice || do_arches || do_mpm)){
	usage( "You need to specify -arches, -ice, or -mpm", "", argv[0]);
    }

    /*
     * Create the components
     */
    try {
	const ProcessorGroup* world = Parallel::getRootProcessorGroup();
	SimulationController* sim = scinew SimulationController(world);

	// Reader
	ProblemSpecInterface* reader = scinew ProblemSpecReader(filename);
	sim->attachPort("problem spec", reader);

	// Connect a MPM module if applicable
	MPMInterface* mpm = 0;
	if(do_mpm){
	  mpm = scinew MPM::SerialMPM(world);
	  sim->attachPort("mpm", mpm);
	}

	// Connect a CFD module if applicable
	CFDInterface* cfd = 0;
	if(do_arches){
	    cfd = scinew ArchesSpace::Arches(world);
	}
	if(do_ice){
	    cfd = scinew ICESpace::ICE(world);
	}
	if(cfd)
	    sim->attachPort("cfd", cfd);

	// Connect an MPMICE module if do_mpm and do_ice are both true
	MPMCFDInterface* mpmcfd = 0;
	if(do_mpm && do_ice){
	    mpmcfd = scinew MPMICESpace::MPMICE(world);
	    sim->attachPort("mpmcfd", mpmcfd);
	}

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
	} else if(loadbalancer == "RoundRobinLoadBalancer"){
	   bal = scinew RoundRobinLoadBalancer(world);
	} else if(loadbalancer == "SimpleLoadBalancer") {
	   bal = scinew SimpleLoadBalancer(world);
	} else {
	   bal = 0;
	   quit( "Unknown load balancer: " + loadbalancer );
	}

	// Scheduler
	if(scheduler == "SingleProcessorScheduler"){
	   SingleProcessorScheduler* sched = 
	      scinew SingleProcessorScheduler(world, output);
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	} else if(scheduler == "MPIScheduler"){
	   MPIScheduler* sched =
	      scinew MPIScheduler(world, output);
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	} else if(scheduler == "MixedScheduler"){
	   if( numThreads > 0 ){
	     if( Parallel::getMaxThreads() == 1 ){
	       Parallel::setMaxThreads( numThreads );
	     }
	   }
	   MixedScheduler* sched =
	      scinew MixedScheduler(world, output);
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	} else if(scheduler == "NullScheduler"){
	   NullScheduler* sched =
	      scinew NullScheduler(world, output);
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	} else {
	   quit( "Unknown scheduler: " + scheduler );
	}

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

	sim->run();

    delete sim;
    delete reader;
    delete mpm;
    delete cfd;
    delete mpmcfd;
    delete output;
    delete bal;
    //    delete sched;
    } catch (Exception& e) {
	cerr << "Caught exception: " << e.message() << '\n';
	if(e.stackTrace())
	   cerr << "Stack trace: " << e.stackTrace() << '\n';
	Parallel::finalizeManager(Parallel::Abort);
	abort();
    } catch (std::exception e){
        cerr << "Caught std exception: " << e.what() << '\n';
	Parallel::finalizeManager(Parallel::Abort);
	abort();       
    } catch(...){
	cerr << "Caught unknown exception\n";
	Parallel::finalizeManager(Parallel::Abort);
	abort();
    }

    /*
     * Finalize MPI
     */
    Parallel::finalizeManager();
}

