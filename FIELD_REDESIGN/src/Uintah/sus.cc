/* REFERENCED */
//static char *id="$Id$";

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

#include <Uintah/Parallel/Parallel.h>
#include <Uintah/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Uintah/Components/SimulationController/SimulationController.h>
#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Components/Schedulers/SingleProcessorScheduler.h>
#include <Uintah/Components/Schedulers/MPIScheduler.h>
#include <Uintah/Components/Schedulers/MixedScheduler.h>
#include <Uintah/Components/Schedulers/SingleProcessorLoadBalancer.h>
#include <Uintah/Components/Schedulers/RoundRobinLoadBalancer.h>
#include <Uintah/Components/Schedulers/SimpleLoadBalancer.h>
#include <Uintah/Components/DataArchiver/DataArchiver.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Parallel/ProcessorGroup.h>
#include <SCICore/Exceptions/Exception.h>

#if HAVE_FPSETMASK
#include <ieeefp.h>
#endif

#include <iostream>
#include <string>
#include <vector>

using SCICore::Exceptions::Exception;
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

#if HAVE_FPSETMASK
    fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);
#endif

    /*
     * Default values
     */
    bool do_mpm=false;
    bool do_arches=false;
    bool do_ice=false;
    bool numThreads = 0;
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
       } else {
	  scheduler="SingleProcessorScheduler"; // Default for serial runs
	  loadbalancer="SingleProcessorLoadBalancer";
       }
    }

    /*
     * Check for valid argument combinations
     */
    if(do_mpm && (do_ice || do_arches)){
	usage( "MPM doesn't yet work with ICE/Arches", "", argv[0]);
    }
    if(do_ice && do_arches){
	usage( "ICE and Arches do not work together", "", argv[0]);
    }
    if(do_ice && numThreads>0){
	usage( "ICE doesn't support threads yet", "", argv[0]);
    }
    if(do_arches && numThreads>0){
	usage( "Arches doesn't do threads yet", "", argv[0]);
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

	    if(numThreads == 0){
		mpm = scinew MPM::SerialMPM(world);
	    } else {
#ifdef WONT_COMPILE_YET
		mpm = scinew ThreadedMPM();
#else
		mpm = 0;
#endif
	    }
	    sim->attachPort("mpm", mpm);
	}

	// Connect a CFD module if applicable
	CFDInterface* cfd = 0;
	if(do_arches){
	    cfd = scinew ArchesSpace::Arches(world);
	}
	if(do_ice){
	    cfd = scinew ICESpace::ICE();
	}
	if(cfd)
	    sim->attachPort("cfd", cfd);

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
	   MixedScheduler* sched =
	      scinew MixedScheduler(world, output);
	   sim->attachPort("scheduler", sched);
	   sched->attachPort("load balancer", bal);
	} else {
	   quit( "Unknown schduler: " + scheduler );
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

//
// $Log$
// Revision 1.23  2000/09/26 21:49:00  dav
// Added the ability to specify the MixedScheduler on the command line.
// Added the ability to set an environment var to allow time for debuggers
// to attach to the program before it starts running.
//
// Revision 1.22  2000/09/25 18:05:27  sparker
// Deal with mpich arguments
// Started FP_SETMASK configuration
//
// Revision 1.21  2000/09/20 16:04:07  sparker
// Fixed code to set LoadBalancer
//
// Revision 1.20  2000/09/12 15:10:34  sparker
// Catch stray std exceptions
//
// Revision 1.19  2000/08/09 03:17:55  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.18  2000/08/08 21:36:52  dav
// fixed 'usage' fix
//
// Revision 1.17  2000/08/08 21:23:20  dav
// Changes so that 'usage' under MPI works correctly.
//
// Revision 1.16  2000/07/27 22:39:40  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.15  2000/07/26 20:14:14  jehall
// Moved taskgraph/dependency output files to UDA directory
// - Added output port parameter to schedulers
// - Added getOutputLocation() to Uintah::Output interface
// - Renamed output files to taskgraph[.xml]
//
// Revision 1.14  2000/06/17 07:06:21  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.13  2000/06/15 23:14:04  sparker
// Cleaned up scheduler code
// Renamed BrainDamagedScheduler to SingleProcessorScheduler
// Created MPIScheduler to (eventually) do the MPI work
//
// Revision 1.12  2000/06/15 21:56:56  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.11  2000/05/30 20:18:40  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.10  2000/05/15 19:39:29  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.9  2000/05/09 22:58:34  sparker
// Changed namespace names
//
// Revision 1.8  2000/04/26 06:47:56  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/19 22:43:51  dav
// more mpi stuff
//
// Revision 1.6  2000/04/19 21:19:59  dav
// more MPI stuff
//
// Revision 1.5  2000/04/13 06:50:49  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/04/11 07:10:29  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/20 17:17:03  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.2  2000/03/17 21:01:02  dav
// namespace mods
//
// Revision 1.1  2000/02/27 07:48:34  sparker
// Homebrew code all compiles now
// First step toward PSE integration
// Added a "Standalone Uintah Simulation" (sus) executable
// MPM does NOT run yet
//
//
