/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#define MPI_VERSION_CHECK

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <cstdlib>

#include   <sstream>
#include   <iostream>

using namespace Uintah;
using SCIRun::Thread;
using SCIRun::Time;
using SCIRun::InternalError;

using std::cerr;
using std::cout;
using std::string;
using std::ostringstream;

#define THREADED_MPI_AVAILABLE

#if defined(__digital__) || defined(_AIX)
#  undef THREADED_MPI_AVAILABLE
#endif

//static bool            allowThreads;
static bool            determinedIfUsingMPI = false;
static bool            initialized = false;
static bool            usingMPI = false;
static bool            usingGPU = false;
static int             numThreads = -1;
static MPI_Comm        worldComm = MPI_Comm(-1);
static int             worldRank = -1;
static int             worldSize = -1;
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
			   " before usingMPI()", __FILE__, __LINE__ );
   }
   return ::usingMPI;
}

bool
Parallel::usingGPU()
{
  return ::usingGPU;
}

void
Parallel::setUsingGPU( bool useGPU )
{
  ::usingGPU = useGPU;
}

int
Parallel::getNumThreads()
{
  return ::numThreads;
}

void
Parallel::setNumThreads( int num)
{
   ::numThreads = num;
   //::allowThreads = true;
}

void
Parallel::noThreading()
{
  //::allowThreads = false;
  ::numThreads = 1;
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

bool 
Parallel::isInitialized()
{
  return initialized;
}


void
Parallel::determineIfRunningUnderMPI( int argc, char** argv )
{
  if(determinedIfUsingMPI)
    return;
  if( char * max = getenv( "PSE_MAX_THREADS" ) ){
    ::numThreads = atoi( max );
    //::allowThreads = true;
    cerr << "PSE_MAX_THREADS set to " << ::numThreads << "\n";

    if( ::numThreads <= 0 || ::numThreads > 16 ){
      // Empirical evidence points to 16 being the most threads
      // that we should use... (this isn't conclusive evidence)
      cerr << "PSE_MAX_THREADS is out of range 1..16\n";
      throw InternalError( "PSE_MAX_THREADS is out of range 1..16\n", __FILE__, __LINE__ );
    }
  }
  
  // Look for SGI MPI
  if(getenv("MPI_ENVIRONMENT")){
    ::usingMPI=true;
  } else if(getenv("RMS_PROCS") || getenv("RMS_STOPONELANINIT")) {           // CompaqMPI (ASCI Q)
    // Above check isn't conclusive, but we will go with it.
    ::usingMPI=true;
  } else if(getenv("LAMWORLD") || getenv("LAMRANK")) {                       // LAM-MPI
    ::usingMPI=true;
  } else if(getenv("SLURM_PROCID") || getenv("SLURM_NPROCS")) {              // ALC's MPI (LLNL)
    ::usingMPI=true;
  } else if(getenv("MPIRUN_RANK")) {                                         // Hera's MPI (LLNL)
    ::usingMPI=true;
  } else if(getenv("OMPI_MCA_ns_nds_num_procs") || getenv("OMPI_MCA_pls") || getenv("OMPI_COMM_WORLD_RANK")) { // Open MPI
    ::usingMPI=true;
  } else if( getenv("MPI_LOCALNRANKS") ) {                                   // Updraft.chpc.utah.edu
    ::usingMPI = true;
  } else {
    // Look for mpich
    for(int i=0;i<argc;i++){
      string s = argv[i];

      // on the turing machine, using mpich, we can't find the mpich var, so
      // search for our own -mpi, as (according to sus.cc),
      // we haven't parsed the args yet
      if(s.substr(0,3) == "-p4" || s == "-mpi")
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
Parallel::initializeManager(int& argc, char**& argv)
{
   if( !determinedIfUsingMPI ) {
      cerr << "Must call determineIfRunningUnderMPI() " << "before initializeManager().\n";
      throw InternalError( "Bad coding... call determineIfRunningUnderMPI()"
			   " before initializeManager()", __FILE__, __LINE__ );
   }
   initialized = true;

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
     if( ::numThreads > 0 ) {
       required = MPI_THREAD_MULTIPLE;
     } else {
       required = MPI_THREAD_SINGLE;
     }
#endif

     int status;
#if !defined( _WIN32 ) && ( !defined( DISABLE_SCI_MALLOC ) || defined( SCI_MALLOC_TRACE ) )
     const char* oldtag = SCIRun::AllocatorSetDefaultTagMalloc("MPI initialization");
#endif
#ifdef __sgi
     if(Thread::isInitialized()){
       cerr << "Thread library initialization occurs before MPI_Init.  This causes problems\nwith MPI exit.  Look for and remove global Thread primitives.\n";
       throw InternalError("Bad MPI Init", __FILE__, __LINE__);
     }
#endif
#ifdef THREADED_MPI_AVAILABLE
     if( ( status = MPI_Init_thread( &argc, &argv, required, &provided ) )
	                                                     != MPI_SUCCESS)
#else
     if( ( status = MPI_Init( &argc, &argv ) ) != MPI_SUCCESS)
#endif
       {
         MpiError(const_cast<char*>("MPI_Init"), status);
       }

#ifdef THREADED_MPI_AVAILABLE
     if( provided < required ) {
       cerr  << "Provided MPI parallel support of " << provided 
	     << " is not enough for the required level of " << required <<"\n";
       throw InternalError( "Bad MPI level", __FILE__, __LINE__ );
     }
#endif

     worldComm=MPI_COMM_WORLD;
     if((status=MPI_Comm_size(worldComm, &worldSize)) != MPI_SUCCESS)
       MpiError(const_cast<char*>("MPI_Comm_size"), status);

     if((status=MPI_Comm_rank(worldComm, &worldRank)) != MPI_SUCCESS)
       MpiError(const_cast<char*>("MPI_Comm_rank"), status);
#if !defined( _WIN32 ) && ( !defined( DISABLE_SCI_MALLOC ) || defined( SCI_MALLOC_TRACE ) )
     SCIRun::AllocatorSetDefaultTagMalloc(oldtag);
     SCIRun::AllocatorMallocStatsAppendNumber(worldRank);
#endif
     rootContext = scinew ProcessorGroup(0, worldComm, true,
					 worldRank,worldSize, ::numThreads);

     if(rootContext->myrank() == 0) {
       cout << "Parallel: " << rootContext->size() << " processors (using MPI)\n";
#ifdef THREADED_MPI_AVAILABLE
       cout << "Parallel: MPI Level Required: " << required << ", provided: " << provided << "\n";
#endif
     }
     //MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
   } else {
     worldRank = 0;
     rootContext = scinew ProcessorGroup(0, 0, false, 0, 1, 0);
   }
}

int
Parallel::getMPIRank()
{
  if( worldRank == -1 ) {
    // Can't throw an exception here because it won't get trapped
    // properly because 'getMPIRank()' is called in the exception
    // handler...
    cout << "ERROR:\n";
    cout << "ERROR: getMPIRank() called before initializeManager()...\n";
    cout << "ERROR:\n";
    Thread::exitAll(1);
  }
  return worldRank;
}

int
Parallel::getMPISize()
{
  return worldSize;
}

void
Parallel::finalizeManager(Circumstances circumstances)
{
  // worldRank is not reset here as even after finalizeManager,
  // some things need to know their rank...

  // only finalize if MPI is initialized
  if(determinedIfUsingMPI==false) {
    return;
  }

  if(::usingMPI) {
    if(circumstances == Abort) {
      int errorcode = 1;
      if(getenv("LAMWORLD") || getenv("LAMRANK")) {
        errorcode = (errorcode << 16) + 1; // see LAM man MPI_Abort
      }
      if (Parallel::getMPIRank() == 0) {
        cout << "An exception was thrown... Goodbye.\n";
      }
      cerr.flush();
      cout.flush();
      Time::waitFor(1.0);
      MPI_Abort(worldComm, errorcode);
    } else {
      int status;
      if ((status = MPI_Finalize()) != MPI_SUCCESS) {
        MpiError(const_cast<char*>("MPI_Finalize"), status);
      }
    }
  }
  if(rootContext) {
    delete rootContext;
    rootContext=0;
  }

  //MPI can no longer be used
  determinedIfUsingMPI=false;
}

ProcessorGroup*
Parallel::getRootProcessorGroup()
{
   if(!rootContext) {
      throw InternalError("Parallel not initialized", __FILE__, __LINE__);
   }

   return rootContext;
}
