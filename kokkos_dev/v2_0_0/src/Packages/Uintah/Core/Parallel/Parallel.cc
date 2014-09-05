#define MPI_VERSION_CHECK

#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Thread.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using SCIRun::Thread;
using SCIRun::InternalError;
using SCIRun::AllocatorSetDefaultTagMalloc;
using SCIRun::AllocatorMallocStatsAppendNumber;

using std::cerr;
using std::cout;
using std::string;
using std::ostringstream;

#define THREADED_MPI_AVAILABLE
#if defined(__digital__) || defined(_AIX)
#undef THREADED_MPI_AVAILABLE
#endif

//static bool            allowThreads;
static bool            determinedIfUsingMPI = false;
static bool            usingMPI = false;
static int             maxThreads = 1;
static MPI_Comm        worldComm = MPI_Comm(-1);
static int             worldRank = -1;
static ProcessorGroup* rootContext = 0;

static
void
MpiError(char* what, int errorcode)
{
   // Simple error handling for now...
   char string_name[1000];
   int resultlen=1000;
   MPI_Error_string(errorcode, string_name, &resultlen);
   cerr << "MPI Error in " << what << ": " << string_name << '\n';
   exit(1);
}

bool
Parallel::usingMPI()
{
   if( !determinedIfUsingMPI ) {
      cerr << "Must call determineIfRunningUnderMPI() before usingMPI().\n";
      throw InternalError( "Bad coding... call determineIfRunningUnderMPI()"
			   " before usingMPI()" );
   }
   return ::usingMPI;
}

int
Parallel::getMaxThreads()
{
   return ::maxThreads;
}

void
Parallel::setMaxThreads( int maxNumThreads )
{
   ::maxThreads = maxNumThreads;
   //::allowThreads = true;
}

void
Parallel::noThreading()
{
  //::allowThreads = false;
  ::maxThreads = 1;
}

void
Parallel::forceMPI()
{
  determinedIfUsingMPI=true;
  ::usingMPI=true;
}

void
Parallel::forceNoMPI()
{
  determinedIfUsingMPI=true;
  ::usingMPI=false;
}

void
Parallel::determineIfRunningUnderMPI( int argc, char** argv )
{
  if(determinedIfUsingMPI)
    return;
  if( char * max = getenv( "PSE_MAX_THREADS" ) ){
    ::maxThreads = atoi( max );
    //::allowThreads = true;
    cerr << "PSE_MAX_THREADS set to " << ::maxThreads << "\n";

    if( ::maxThreads <= 0 || ::maxThreads > 16 ){
      // Empirical evidence points to 16 being the most threads
      // that we should use... (this isn't conclusive evidence)
      cerr << "PSE_MAX_THREADS is out of range 1..16\n";
      throw InternalError( "PSE_MAX_THREADS is out of range 1..16\n" );
    }
  }
  
  // Look for SGI MPI
  if(getenv("MPI_ENVIRONMENT")){
    ::usingMPI=true;
  } else if(getenv("RMS_PROCS") || getenv("RMS_STOPONELANINIT")){
    // Look for CompaqMPI - that latter is set on ASCI Q
    // This isn't conclusive, but we will go with it.
    ::usingMPI=true;
  } else if(getenv("LAMWORLD") || getenv("LAMRANK")){
    // Look for LAM-MPI
    ::usingMPI=true;
  } else {
    // Look for mpich
    for(int i=0;i<argc;i++){
      string s = argv[i];
      if(s.substr(0,3) == "-p4")
	::usingMPI=true;
    }
  }
  determinedIfUsingMPI = true;

#if defined(_AIX)
  // Hardcoded for AIX xlC for now...need to figure out how to do this 
  // automagically.
  ::usingMPI=true;
#endif
}

void
Parallel::initializeManager(int& argc, char**& argv, const string & scheduler)
{
   if( !determinedIfUsingMPI ) {
      cerr << "Must call determineIfRunningUnderMPI() " 
	   << "before initializeManager().\n";
      throw InternalError( "Bad coding... call determineIfRunningUnderMPI()"
			   " before initializeManager()" );
   }

   if( worldRank != -1 ) { // IF ALREADY INITIALIZED, JUST RETURN...
      return;
      // If worldRank is not -1, then we have already been initialized..
      // This only happens (I think) if usage() is called (due to bad
      // input parameters (to sus)) and usage() needs to init mpi so that
      // it only displays the usage to the root process.
   }
#ifdef THREADED_MPI_AVAILABLE
   int provided = -1;
   int required = MPI_THREAD_SINGLE;
#endif
   if(::usingMPI){	

#ifdef THREADED_MPI_AVAILABLE
     if( scheduler == "MixedScheduler" ) {
       required = MPI_THREAD_MULTIPLE;
     } else {
       required = MPI_THREAD_SINGLE;
     }
#endif

     int status;
     const char* oldtag = AllocatorSetDefaultTagMalloc("MPI initialization");

#ifdef __sgi
     if(Thread::isInitialized()){
       cerr << "Thread library initialization occurs before MPI_Init.  This causes problems\nwith MPI exit.  Look for and remove global Thread primitives.\n";
       throw InternalError("Bad MPI Init");
     }
#endif

#ifdef THREADED_MPI_AVAILABLE
     if( ( status = MPI_Init_thread( &argc, &argv, required, &provided ) )
	                                                     != MPI_SUCCESS)
#else
     if( ( status = MPI_Init( &argc, &argv ) ) != MPI_SUCCESS)
#endif
       {
	 MpiError("MPI_Init", status);
       }

#ifdef THREADED_MPI_AVAILABLE
     if( provided < required ){
       ostringstream msg;
       msg << "Provided MPI parallel support of " << provided 
	   << " is not enough for the required level of " << required;
       throw InternalError( msg.str() );
     }
#endif

     worldComm=MPI_COMM_WORLD;
     int worldSize;
     if((status=MPI_Comm_size(worldComm, &worldSize)) != MPI_SUCCESS)
       MpiError("MPI_Comm_size", status);

     if((status=MPI_Comm_rank(worldComm, &worldRank)) != MPI_SUCCESS)
       MpiError("MPI_Comm_rank", status);
     AllocatorSetDefaultTagMalloc(oldtag);
     AllocatorMallocStatsAppendNumber(worldRank);
     rootContext = scinew ProcessorGroup(0, worldComm, true,
					 worldRank,worldSize);

     if(rootContext->myrank() == 0){
       cerr << "Parallel: " << rootContext->size() 
	    << " processors (using MPI)\n";
#ifdef THREADED_MPI_AVAILABLE
       cerr << "Parallel: MPI Level Required: " << required << ", provided: " 
	    << provided << "\n";
#endif
     }
   } else {
     worldRank = 0;
     rootContext = scinew ProcessorGroup(0,0, false, 0, 1);
   }
}

int
Parallel::getMPIRank()
{
  return worldRank;
}

void
Parallel::finalizeManager(Circumstances circumstances)
{
  // worldRank is not reset here as even after finalizeManager,
  // some things need to knoww their rank...

  if(::usingMPI){
    if(circumstances == Abort){
      cerr.flush();
      cout.flush();
      sleep(1);
      MPI_Abort(worldComm, 1);
    } else {
      int status;
      if((status=MPI_Finalize()) != MPI_SUCCESS)
	MpiError("MPI_Finalize", status);
    }
  }
  if(rootContext){
    delete rootContext;
    rootContext=0;
  }
}

ProcessorGroup*
Parallel::getRootProcessorGroup()
{
   if(!rootContext)
      throw InternalError("Parallel not initialized");

   return rootContext;
}

