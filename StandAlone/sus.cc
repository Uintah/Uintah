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
#include <Uintah/Components/MPM/ThreadedMPM.h>
#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Components/Schedulers/BrainDamagedScheduler.h>
#include <SCICore/Exceptions/Exception.h>

#include <iostream>
#include <string>
#include <vector>

using SCICore::Exceptions::Exception;
using std::cerr;
using std::string;
using std::vector;

using Uintah::Parallel::Parallel;
using Uintah::Components::SimulationController;
using Uintah::Components::SerialMPM;
using Uintah::Components::ThreadedMPM;
using Uintah::Components::Arches;
using Uintah::Components::ICE;
using Uintah::Components::BrainDamagedScheduler;
using Uintah::Interface::MPMInterface;
using Uintah::Interface::CFDInterface;

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
	cerr << "Error parsing argument: " << badarg << '\n';
    cerr << "Usage: " << progname << " [options]\n\n";
    cerr << "Valid options are:\n";
    cerr << "NOT FINISHED\n";
    exit(1);
}

int main(int argc, char** argv)
{
    /*
     * Default values
     */
    bool do_mpm=false;
    bool do_arches=false;
    bool do_ice=false;
    bool numThreads = 0;
    string filename;

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
		cerr << "You must provide a number of threads for -nthreads\n";
		usage(s, argv[0]);
	    }
	    numThreads = atoi(argv[i]);
	} else {
	    if(filename!="")
		usage(s, argv[0]);
	    else
		filename = argv[i];
	}
    }

    if(filename == ""){
	cerr << "No input file specified\n";
	usage("", argv[0]);
    }

    /*
     * Check for valid argument combinations
     */
    if(do_mpm && (do_ice || do_arches)){
	cerr << "MPM doesn't yet work with ICE/Arches\n";
	usage("", argv[0]);
    }
    if(do_ice && do_arches){
	cerr << "ICE and Arches do not work together\n";
	usage("", argv[0]);
    }
    if(do_ice && numThreads>0){
	cerr << "ICE doesn't support threads yet\n";
	usage("", argv[0]);
    }
    if(do_arches && numThreads>0){
	cerr << "Arches doesn't do threads yet\n";
	usage("", argv[0]);
    }

    if(!(do_ice || do_arches || do_mpm)){
	cerr << "You need to specify -arches, -ice, or -mpm\n";
	usage("", argv[0]);
    }

    /*
     * Initialize MPI
     */
    Parallel::initializeManager(argc, argv);

    /*
     * Create the components
     */
    try {
	SimulationController* sim = new SimulationController();

	// Reader
	ProblemSpecInterface* reader = new ProblemSpecReader(filename);
	sim->attachPort("problem spec", reader);

	// Connect a MPM module if applicable
	if(do_mpm){
	    MPMInterface* mpm;
	    if(numThreads == 0){
		mpm = new SerialMPM();
	    } else {
#ifdef WONT_COMPILE_YET
		mpm = new ThreadedMPM();
#else
		mpm = 0;
#endif
	    }
	    sim->attachPort("MPM", mpm);
	}

	// Connect a CFD module if applicable
	CFDInterface* cfd = 0;
	if(do_arches){
	    cfd = new Arches();
	}
	if(do_ice){
	    cfd = new ICE();
	}
	if(cfd)
	    sim->attachPort("CFD", cfd);

	// Output


	// Scheduler
	BrainDamagedScheduler* sched = new BrainDamagedScheduler();
	sched->setNumThreads(numThreads);
	sim->attachPort("Scheduler", sched);

	/*
	 * Start the simulation controller
	 */
	sim->run();
    } catch (Exception& e) {
	cerr << "Caught exception: " << e.message() << '\n';
	abort();
    } catch(...){
	cerr << "Caught unknown exception\n";
	abort();
    }

    /*
     * Finalize MPI
     */
    Parallel::finalizeManager();
}

//
// $Log$
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
