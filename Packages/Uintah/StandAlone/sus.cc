
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
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/FractureMPM.h> // for Fracture
#include <Packages/Uintah/CCA/Components/MPM/RigidMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ShellMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ImpMPM.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/AMRICE.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArches.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson1.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson2.h>
#include <Packages/Uintah/CCA/Components/Examples/Burger.h>
#include <Packages/Uintah/CCA/Components/Examples/Wave.h>
#include <Packages/Uintah/CCA/Components/Examples/AMRWave.h>
#include <Packages/Uintah/CCA/Components/Examples/ParticleTest1.h>
#include <Packages/Uintah/CCA/Components/Examples/RegridderTest.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson3.h>
#include <Packages/Uintah/CCA/Components/Examples/SimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Examples/AMRSimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Examples/SolverTest1.h>
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
#include <Packages/Uintah/CCA/Components/ComponentFactory.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>
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
#if 0
#include <fenv.h>
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
  bool   do_smpmice=false;   // for shell MPM with ICE
  bool   do_rmpmice=false;   // for rigid MPM with ICE
  bool   do_fmpmice=false;   // for Fracture MPM with ICE
  bool   do_impmpm=false;
  bool   do_arches=false;
  bool   do_ice=false;
  bool   do_particletest1=false;
  bool   do_regriddertest=false;
  bool   do_burger=false;
  bool   do_wave=false;
  bool   do_poisson1=false;
  bool   do_poisson2=false;
  bool   do_poisson3=false;
  bool   do_solvertest1=false;
  bool   do_simplecfd=false;
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
    } else if(s == "-mpm"){
      do_mpm=true;
    } else if(s == "-mpmf"){    // for Fracture
      do_mpmf=true;
    } else if(s == "-rmpm"){
      do_rmpm=true;
    } else if(s == "-smpm"){
      do_smpm=true;
    } else if(s == "-smpmice"){
      do_smpmice = true;
    } else if(s == "-rmpmice"){
      do_rmpmice = true;
    } else if(s == "-fmpmice"){
      do_fmpmice = true;
    } else if(s == "-impm"){
      do_impmpm=true;
    } else if(s == "-arches"){
      do_arches=true;
    } else if(s == "-ice"){
      do_ice=true;
    } else if(s == "-mpmice"){
      do_ice=true;
      do_mpm=true;
    } else if(s == "-particletest1"){
      do_particletest1=true;
    } else if(s == "-regriddertest"){
      do_regriddertest=true;
    } else if(s == "-burger"){
      do_burger=true;
    } else if(s == "-wave"){
      do_wave=true;
    } else if(s == "-poisson1"){
      do_poisson1=true;
    } else if(s == "-poisson2"){
      do_poisson2=true;
    } else if(s == "-poisson3"){
      do_poisson3=true;
    } else if(s == "-solvertest1"){
      do_solvertest1=true;
    } else if(s == "-scfd" || s == "-simplecfd"){
      do_simplecfd=true;
    } else if(s == "-mpmarches"){
      do_arches=true;
      do_mpm=true;
    } else if(s == "-AMR" || s == "-amr"){
      do_AMR=true;
//  } else if(s == "-s3"){
//    useScheduler3=true;
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
    } else if(s == "-combine_patches") {
      combine_patches=true;	   
    } else if(s == "-reduce_uda") {
      reduce_uda=true;	   
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

  if (restart || combine_patches || reduce_uda) {
    udaDir = filename;
    filename = filename + "/input.xml";
  }

  if (!Uintah::Parallel::usingMPI()) {
    TAU_PROFILE_SET_NODE(0);
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
      Time::waitFor( (double)sleepTime );
    }

    // Reader
    ProblemSpecInterface* reader = scinew ProblemSpecReader(filename);
    ProblemSpecP ups = reader->readInputFile();

    // grab AMR from the ups file if not specified on the command line
    if (!do_AMR)
      ups->get("doAMR", do_AMR);

    const ProcessorGroup* world = Uintah::Parallel::getRootProcessorGroup();
    SimulationController* ctl = 
      scinew AMRSimulationController(world, do_AMR);
    ctl->attachPort("problem spec", reader);

    Regridder* reg = 0;
    if(do_AMR) {
      reg = RegridderFactory::create(ups, world);
      ctl->attachPort("regridder", reg);
    }

    // Solver
    SolverInterface* solve = 0;
    solve = SolverFactory::create(ups, world, solver);

    // try to make it from the command line first, then look in ups file
    SimulationInterface* sim = 0;
    UintahParallelComponent* comp = 0;
    if(do_mpm && do_ice){
      MPMICE* mpmice = scinew MPMICE(world,STAND_MPMICE,do_AMR);
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
    } else if(do_smpmice){
      MPMICE* mpmice = scinew MPMICE(world, SHELL_MPMICE, do_AMR);
      sim = mpmice;
      comp = mpmice;
    } else if(do_rmpmice){
      MPMICE* mpmice = scinew MPMICE(world, RIGID_MPMICE, do_AMR);
      sim = mpmice;
      comp = mpmice;
    } else if(do_fmpmice){
      MPMICE* mpmice = scinew MPMICE(world, FRACTURE_MPMICE, do_AMR);
      sim = mpmice;
      comp = mpmice;
    } else if(do_impmpm){
      ImpMPM* impm = scinew ImpMPM(world);
      sim = impm;
      comp = impm;
    } else if(do_arches){
      Arches* arches = scinew Arches(world);
      sim = arches;
      comp = arches;
    } else if(do_ice) {
      ICE* ice = NULL;
      if(do_AMR){
        ice = scinew AMRICE(world);
      }else{
        ice = scinew ICE(world, do_AMR);
      }
      sim = ice;
      comp = ice;
    } else if(do_burger){
      Burger* burger = scinew Burger(world);
      sim = burger;
      comp = burger;
    } else if(do_wave){
      Wave* wave;
      if(do_AMR)
        wave = scinew AMRWave(world);
      else
        wave = scinew Wave(world);
      sim = wave;
      comp = wave;
    } else if(do_poisson1){
      Poisson1* poisson1 = scinew Poisson1(world);
      sim = poisson1;
      comp = poisson1;
    } else if(do_particletest1){
      ParticleTest1* pt1 = scinew ParticleTest1(world);
      sim = pt1;
      comp = pt1;
    } else if(do_regriddertest){
      RegridderTest* rgt = scinew RegridderTest(world);
      sim = rgt;
      comp = rgt;
    } else if(do_poisson2){
      Poisson2* poisson2 = scinew Poisson2(world);
      sim = poisson2;
      comp = poisson2;
    } else if(do_poisson3){
      Poisson3* poisson3 = scinew Poisson3(world);
      sim = poisson3;
      comp = poisson3;
    } else if(do_solvertest1){
      SolverTest1* solvertest1 = scinew SolverTest1(world);
      sim = solvertest1;
      comp = solvertest1;
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
      ctl->doCombinePatches(udaDir, false);
    } else if (reduce_uda) {
      UdaReducer* pc = scinew UdaReducer(world, udaDir);
      sim = pc;
      comp = pc;
      // the ctl will do nearly the same thing for combinePatches and reduceUda
      ctl->doCombinePatches(udaDir, true);
    } else { // try it from the ups file
      comp = ComponentFactory::create(ups, world, do_AMR);
      sim = dynamic_cast<SimulationInterface*>(comp);
    }
    
    ctl->attachPort("sim", sim);
    comp->attachPort("solver", solve);
    
    ModelMaker* modelmaker = scinew ModelFactory(world);
    comp->attachPort("modelmaker", modelmaker);
    
    // Load balancer
    LoadBalancer* bal;
    UintahParallelComponent* lb; // to add scheduler as a port
    LoadBalancerCommon* lbc = LoadBalancerFactory::create(ups, world);
    lb = lbc;
    bal = lbc;
    
    // Output
    DataArchiver* dataarchiver = scinew DataArchiver(world, udaSuffix);
    Output* output = dataarchiver;
    ctl->attachPort("output", output);
    dataarchiver->attachPort("load balancer", bal);
    comp->attachPort("output", output);
    dataarchiver->attachPort("sim", sim);
    
    // Scheduler
    SchedulerCommon* sched = SchedulerFactory::create(ups, world, output);
    Scheduler* sch = sched;
    sched->attachPort("load balancer", bal);
    ctl->attachPort("scheduler", sched);
    lb->attachPort("scheduler", sched);
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
    cerr << Uintah::Parallel::getMPIRank() << " Caught exception: " << e.message() << '\n';
    if(e.stackTrace())
      cerr << "Stack trace: " << e.stackTrace() << '\n';
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
