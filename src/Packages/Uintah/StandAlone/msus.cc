
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
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>


#include <Packages/Uintah/CCA/Components/Models/ModelFactory.h>

#include <Packages/Uintah/CCA/Components/PatchCombiner/PatchCombiner.h>
#include <Packages/Uintah/CCA/Components/PatchCombiner/UdaReducer.h>
#include <Packages/Uintah/CCA/Components/DataArchiver/DataArchiver.h>

#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <Packages/Uintah/CCA/Components/SimulationController/SimulationControllerFactory.h>
#include <Packages/Uintah/CCA/Components/SimulationController/MultipleSimulationController.h>
#include <Packages/Uintah/CCA/Components/Solvers/SolverFactory.h>
#include <Packages/Uintah/CCA/Components/Regridder/RegridderFactory.h>
#include <Packages/Uintah/CCA/Components/LoadBalancers/LoadBalancerFactory.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerFactory.h>
#include <Packages/Uintah/CCA/Components/ComponentFactory.h>
#include <Packages/Uintah/CCA/Components/Switcher/Switcher.h>

#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/ieeefp_defs.h>
#include <sci_defs/hypre_defs.h>

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

#include <iostream>
#include <sstream>
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

#define HAVE_MPICH

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

    int    udaSuffix = -1;
    bool   emit_graphs=false;
    bool   do_AMR=false;
    string filename;
    string scheduler;
    string loadbalancer;

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

#ifdef USE_VAMPIR
    VTsetup();
#endif

#ifndef _WIN32
    SimulationController::start_addr = (char*)sbrk(0);
#endif
    Thread::disallow_sgi_OpenGL_page0_sillyness();

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
          stringstream s(st);
          int sleepTime = 0;
          s >> sleepTime;
	  cerr << "SLEEPING FOR " << sleepTime 
	       << " SECONDS TO ALLOW DEBUGGER ATTACHMENT\n";
	  cerr << "PID is " << getpid() << "\n";
          Time::waitFor( (double)sleepTime );
	}

	const ProcessorGroup* world = 
          Uintah::Parallel::getRootProcessorGroup();

	// Reader
	ProblemSpecInterface* reader = scinew ProblemSpecReader(filename);
        ProblemSpecP d_ups = reader->readInputFile();

        vector<string> input_files;
        vector<string> simulations;
        vector<ProblemSpecInterface*> readers;

        for (ProblemSpecP in_ps = d_ups->findBlock("input_file"); in_ps != 0;
             in_ps = in_ps->findNextBlock("input_file")) {
          string in("");
          if (in_ps->get(in)) {
            input_files.push_back(in);
            readers.push_back(scinew ProblemSpecReader(in));
          }
        }

        for (ProblemSpecP sim_ps = d_ups->findBlock("SimulationComponent"); 
             sim_ps != 0; 
             sim_ps = sim_ps->findNextBlock("SimulationComponent")) {
          string sim("");
          if (sim_ps->get(sim))
            simulations.push_back(sim);
        }

        for (unsigned int i = 0; i < input_files.size(); i++) {
          cout << "SimCom = " << simulations[i] << " in_file = " 
               << input_files[i] << endl;
        }

        string sim_ctl("");
        d_ups->get("SimulationController",sim_ctl);
        cout << "Simulation Controller = " << sim_ctl << endl;

        if (!d_ups)
          throw ProblemSetupException("Cannot read problem specification");

#if 1
        SimulationController* ctl = SimulationControllerFactory::create(d_ups,
                                                                        world);
#else
        SimulationController* ctl = scinew MultipleSimulationController(world);
#endif

        // Switcher component
        Switcher* switcher = 0;

        
        if (readers.empty()) {
          switcher = scinew Switcher(world,1);
          ctl->attachPort("problem spec", reader);

          ctl->attachPort("sim", switcher);

          // SimulationComponentFactory
          SimulationComponent* simcomp = ComponentFactory::create(d_ups,world);
          SimulationInterface* sim = simcomp->d_sim;
          UintahParallelComponent* comp = simcomp->d_comp;
          switcher->attachPort("sim", sim);

          // SolverFactory
          SolverInterface* solve = SolverFactory::create(d_ups,world);
          comp->attachPort("solver",solve);

          // RegridderFactory
          RegridderCommon* regrid_com = RegridderFactory::create(d_ups,world);
          Regridder* regrid = regrid_com;
          if (!regrid)
            ctl->attachPort("regridder",regrid_com);

          ModelMaker* modelmaker = scinew ModelFactory(world);
          comp->attachPort("modelmaker", modelmaker);

          // Load balancer
          // LoadBalancerFactory
          LoadBalancerCommon* bal_com = 
            LoadBalancerFactory::create(d_ups,world);

          LoadBalancer* bal = bal_com;
          UintahParallelComponent* lb = bal_com;
          
          
          // Output
          DataArchiver* dataarchiver = scinew DataArchiver(world, udaSuffix);
          Output* output = dataarchiver;
          ctl->attachPort("output", output);
          dataarchiver->attachPort("load balancer", bal);
          comp->attachPort("output", output);
          
          // Scheduler
          // SchdulerFactory
          
          SchedulerCommon* sch_com=SchedulerFactory::create(d_ups,world,output);
          Scheduler* sch = sch_com;
          
          sch_com->attachPort("load balancer", bal);
          ctl->attachPort("scheduler", sch_com);
          
          lb->attachPort("scheduler", sch);
          
          sch->addReference();
          if (emit_graphs) 
            sch->doEmitTaskGraphDocs();
          
        }
        else {
          switcher = scinew Switcher(world,input_files.size());
          ctl->attachPort("problem spec", reader);
          ctl->attachPort("sim",switcher);

          // RegridderFactory
          // Should only be listed once
          RegridderCommon* regrid_com = 
            RegridderFactory::create(d_ups,world);
          Regridder* regrid = regrid_com;
          if (!regrid)
            ctl->attachPort("regridder",regrid_com);
          
          // Load balancer
          // LoadBalancerFactory
          // Should only be listed once
          LoadBalancerCommon* bal_com = 
            LoadBalancerFactory::create(d_ups,world);
          
          LoadBalancer* bal = bal_com;
          UintahParallelComponent* lb = bal_com;
          
          
          // Output
          DataArchiver* dataarchiver = scinew DataArchiver(world, udaSuffix);
          Output* output = dataarchiver;
          ctl->attachPort("output", output);
          dataarchiver->attachPort("load balancer", bal);

          // Scheduler
          // SchdulerFactory
          // Should only be listed once
          
          SchedulerCommon* sch_com=
            SchedulerFactory::create(d_ups,world,output);
          
          Scheduler* sch = sch_com;
          
          sch_com->attachPort("load balancer", bal);
          ctl->attachPort("scheduler", sch_com);
          
          lb->attachPort("scheduler", sch);
          
          sch->addReference();
          if (emit_graphs) 
            sch->doEmitTaskGraphDocs();
          
          // Load up the individual simulation component input files
          // and add to the switcher component.
          for (unsigned int i = 0; i < readers.size(); i++) {
            cout << "numConnections(problem spec) = " 
                 << switcher->numConnections("problem spec") << endl;
            switcher->attachPort("problem spec", readers[i]);

          // SimulationComponentFactory
            ProblemSpecP sim_ups = readers[i]->readInputFile();

            SimulationComponent* simcomp = 
              ComponentFactory::create(sim_ups,world);
            SimulationInterface* sim = simcomp->d_sim;
            UintahParallelComponent* comp = simcomp->d_comp;
            switcher->attachPort("sim", sim);

            // SolverFactory
            SolverInterface* solve = SolverFactory::create(sim_ups,world);
            comp->attachPort("solver",solve);

            // ModelMaker
            ModelMaker* modelmaker = scinew ModelFactory(world);
            comp->attachPort("modelmaker", modelmaker);

            // Output
            comp->attachPort("output", output);
          }
        }
       
	/*
	 * Start the simulation controller
	 */


        ctl->run();
	delete ctl;
        delete switcher;

#if 0
	sch->removeReference();
	delete sch;
	delete bal;
        delete simcomp;
	delete output;
	delete reader;
	delete modelmaker;
#endif
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

