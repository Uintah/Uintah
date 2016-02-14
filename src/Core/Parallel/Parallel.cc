/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
#include <sstream>
#include <iostream>

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

// bool         Parallel::allowThreads_;

int             Parallel::numThreads_           = -1;
bool            Parallel::determinedIfUsingMPI_ = false;

bool            Parallel::initialized_          = false;
bool            Parallel::usingMPI_             = false;
bool            Parallel::usingDevice_          = false;
int             Parallel::worldRank_            = -1;
int             Parallel::worldSize_            = -1;
ProcessorGroup* Parallel::rootContext_          = 0;

namespace Uintah {

  // While worldComm_ should be declared in Parallel.h, I would need to
  // #include mpi.h, which then makes about everything in Uintah
  // depend on mpi.h, so I'm just going to create it here.

  static MPI_Comm           worldComm_            = MPI_Comm(-1);
}

static
void
MpiError(char* what, int errorcode)
{
  // Simple error handling for now...
  int  resultlen = -1;
  char string_name[ MPI_MAX_ERROR_STRING ];

  MPI_Error_string( errorcode, string_name, &resultlen );
  cerr << "MPI Error in " << what << ": " << string_name << '\n';

  exit(1);
}

bool
Parallel::usingMPI()
{
   if( !determinedIfUsingMPI_ ) {
      cerr << "Must call determineIfRunningUnderMPI() before usingMPI().\n";
      throw InternalError( "Bad coding... call determineIfRunningUnderMPI()"
                           " before usingMPI()", __FILE__, __LINE__ );
   }
   return usingMPI_;
}

bool
Parallel::usingDevice()
{
  return usingDevice_;
}

void
Parallel::setUsingDevice( bool state )
{
  usingDevice_ = state;
}

int
Parallel::getNumThreads()
{
  return numThreads_;
}

void
Parallel::setNumThreads( int num)
{
   numThreads_ = num;
   //allowThreads = true;
}

void
Parallel::noThreading()
{
  //allowThreads_ = false;
  numThreads_ = 1;
}

void
Parallel::forceMPI()
{
  determinedIfUsingMPI_ = true;
  usingMPI_             = true;
}

void
Parallel::forceNoMPI()
{
  determinedIfUsingMPI_ = true;
  usingMPI_             = false;
}

bool 
Parallel::isInitialized()
{
  return initialized_;
}


void
Parallel::determineIfRunningUnderMPI( int argc, char** argv )
{
  if( determinedIfUsingMPI_ ) {
    return;
  }
  if( char * max = getenv( "PSE_MAX_THREADS" ) ){
    numThreads_ = atoi( max );
    // allowThreads_ = true;
    cerr << "PSE_MAX_THREADS set to " << numThreads_ << "\n";

    if( numThreads_ <= 0 || numThreads_ > 16 ){
      // Empirical evidence points to 16 being the most threads
      // that we should use... (this isn't conclusive evidence)
      cerr << "PSE_MAX_THREADS is out of range 1..16\n";
      throw InternalError( "PSE_MAX_THREADS is out of range 1..16\n", __FILE__, __LINE__ );
    }
  }
  
  // Try to automatically determine if we are running under MPI (many MPIs set environment variables
  // that can be used for this.) 
  if(getenv("MPI_ENVIRONMENT")){                                  // Look for SGI MPI
    usingMPI_ =true;
  }
  else if(getenv("RMS_PROCS") || getenv("RMS_STOPONELANINIT")) {  // CompaqMPI (ASCI Q)
    // Above check isn't conclusive, but we will go with it.
    usingMPI_ = true;
  }
  else if(getenv("LAMWORLD") || getenv("LAMRANK")) {              // LAM-MPI
    usingMPI_ = true;
  }
  else if(getenv("SLURM_PROCID") || getenv("SLURM_NPROCS")) {     // ALC's MPI (LLNL)
    usingMPI_ = true;
  }
  else if(getenv("MPIRUN_RANK")) {                                // Hera's MPI (LLNL)
    usingMPI_ = true;
  }
  else if(getenv("OMPI_MCA_ns_nds_num_procs") || getenv("OMPI_MCA_pls") || getenv("OMPI_COMM_WORLD_RANK")) { // Open MPI
    usingMPI_ = true;
  }
  else if( getenv("MPI_LOCALNRANKS") ) {                          // Updraft.chpc.utah.edu
    usingMPI_ = true;
  }
  else {
    // Look for mpich
    for (int i = 0; i < argc; i++) {
      string s = argv[i];

      // on the turing machine, using mpich, we can't find the mpich var, so
      // search for our own -mpi, as (according to sus.cc),
      // we haven't parsed the args yet
      if (s.substr(0, 3) == "-p4" || s == "-mpi") {
        usingMPI_ = true;
      }
    }
  }
  determinedIfUsingMPI_ = true;
}

void
Parallel::initializeManager(int& argc, char**& argv)
{
  if( !determinedIfUsingMPI_ ) {
    cerr << "Must call determineIfRunningUnderMPI() " << "before initializeManager().\n";
    throw InternalError( "Bad coding... call determineIfRunningUnderMPI()"
                         " before initializeManager()", __FILE__, __LINE__ );
  }
  initialized_ = true;

  if( worldRank_ != -1 ) { // IF ALREADY INITIALIZED, JUST RETURN...
    return;
    // If worldRank_ is not -1, then we have already been initialized..
    // This only happens (I think) if usage() is called (due to bad
    // input parameters (to sus)) and usage() needs to init mpi so that
    // it only displays the usage to the root process.
  }
#ifdef THREADED_MPI_AVAILABLE
  int provided = -1;
  int required = MPI_THREAD_SINGLE;
#endif
  if( usingMPI_ ){     
#ifdef THREADED_MPI_AVAILABLE
    if( numThreads_ > 0 ) {
      required = MPI_THREAD_MULTIPLE;
    } else {
      required = MPI_THREAD_SINGLE;
    }
#endif

    int status;
#if ( !defined( DISABLE_SCI_MALLOC ) || defined( SCI_MALLOC_TRACE ) )
    const char* oldtag = SCIRun::AllocatorSetDefaultTagMalloc("MPI initialization");
#endif
#ifdef THREADED_MPI_AVAILABLE
    if( ( status = MPI_Init_thread( &argc, &argv, required, &provided ) ) != MPI_SUCCESS) {
#else
    if( ( status = MPI_Init( &argc, &argv ) ) != MPI_SUCCESS) {
#endif
      MpiError(const_cast<char*>("MPI_Init"), status);
    }

#ifdef THREADED_MPI_AVAILABLE
    if( provided < required ) {
      cerr << "Provided MPI parallel support of " << provided << " is not enough for the required level of " << required << "\n"
           << "To use multi-threaded schedulers, your MPI implementation needs to support MPI_THREAD_MULTIPLE (level-3)" << std::endl;
      throw InternalError( "Bad MPI level", __FILE__, __LINE__ );
    }
#endif

    Uintah::worldComm_ = MPI_COMM_WORLD;
    if( ( status=MPI_Comm_size( Uintah::worldComm_, &worldSize_ ) ) != MPI_SUCCESS ) {
      MpiError(const_cast<char*>("MPI_Comm_size"), status);
    }

    if((status=MPI_Comm_rank( Uintah::worldComm_, &worldRank_ )) != MPI_SUCCESS) {
      MpiError(const_cast<char*>("MPI_Comm_rank"), status);
    }

#if ( !defined( DISABLE_SCI_MALLOC ) || defined( SCI_MALLOC_TRACE ) )
    SCIRun::AllocatorSetDefaultTagMalloc(oldtag);
    SCIRun::AllocatorMallocStatsAppendNumber( worldRank_ );
#endif
    rootContext_ = scinew ProcessorGroup( 0, Uintah::worldComm_, true, worldRank_, worldSize_, numThreads_ );

    if(rootContext_->myrank() == 0) {
      std::string plural = (rootContext_->size() > 1) ? "processes" : "process" ;
      std::cout << "Parallel: " << rootContext_->size() << " MPI " << plural << " (using MPI)\n";
#ifdef THREADED_MPI_AVAILABLE
      if (numThreads_ > 0) {
        cout << "Parallel: " << numThreads_ << " threads per MPI process\n";
      }
      cout << "Parallel: MPI Level Required: " << required << ", provided: " << provided << "\n";
#endif
    }
     //MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  }
  else {
    worldRank_   = 0;
    rootContext_ = scinew ProcessorGroup(0, 0, false, 0, 1, 0);
  }
}

int
Parallel::getMPIRank()
{
  if( worldRank_ == -1 ) {
    // Can't throw an exception here because it won't get trapped
    // properly because 'getMPIRank()' is called in the exception
    // handler...
    cout << "ERROR:\n";
    cout << "ERROR: getMPIRank() called before initializeManager()...\n";
    cout << "ERROR:\n";
    Thread::exitAll(1);
  }
  return worldRank_;
}

int
Parallel::getMPISize()
{
  return worldSize_;
}

void
Parallel::finalizeManager( Circumstances circumstances /* = NormalShutdown */ )
{
  static bool finalized = false;

  if( finalized ) {
    // Due to convoluted logic, signal, and exception handling,
    // finalizeManager() can be easily/mistakenly called multiple
    // times.  This catches that case and returns harmlessly.
    //
    // (One example of this occurs when MPI_Abort causes an SIG_TERM
    // to be thrown, which is caught by Uintah's exit handler, which
    // in turn calls finalizeManager.)
    return;
  }

  finalized = true;

  // worldRank_ is not reset here as even after finalizeManager,
  // some things need to know their rank...

  if( determinedIfUsingMPI_ == false ) {
  // Only finalize if MPI is initialized.
    return;
  }

  if( usingMPI_ ) {
    if(circumstances == Abort) {
      int errorcode = 1;
      if(getenv("LAMWORLD") || getenv("LAMRANK")) {
        errorcode = (errorcode << 16) + 1; // see LAM man MPI_Abort
      }
      if( worldRank_ == 0 ) {
        cout << "FinalizeManager() called... Calling MPI_Abort on rank " << worldRank_ << ".\n";
      }
      cerr.flush();
      cout.flush();
      Time::waitFor(1.0);
      MPI_Abort( Uintah::worldComm_, errorcode );
    } else {
      int status;
      if ((status = MPI_Finalize()) != MPI_SUCCESS) {
        MpiError(const_cast<char*>("MPI_Finalize"), status);
      }
    }
  }
  if( rootContext_ ) {
    delete rootContext_;
    rootContext_ = 0;
  }

  // MPI can no longer be used.
  determinedIfUsingMPI_ = false;
}

ProcessorGroup*
Parallel::getRootProcessorGroup()
{
   if( !rootContext_ ) {
      throw InternalError("Parallel not initialized", __FILE__, __LINE__);
   }

   return rootContext_;
}
