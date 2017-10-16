/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <Core/Parallel/Parallel.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahMPI.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace Uintah;


#define THREADED_MPI_AVAILABLE

#if defined(__digital__) || defined(_AIX)
#  undef THREADED_MPI_AVAILABLE
#endif

bool             Parallel::s_initialized             = false;
bool             Parallel::s_using_device            = false;
int              Parallel::s_num_threads             = -1;
int              Parallel::s_num_partitions          = -1;
int              Parallel::s_threads_per_partition   = -1;
int              Parallel::s_world_rank              = -1;
int              Parallel::s_world_size              = -1;
std::thread::id  Parallel::s_main_thread_id          = std::this_thread::get_id();
ProcessorGroup*  Parallel::s_root_context            = nullptr;


namespace Uintah {

  // While worldComm_ should be declared in Parallel.h, I would need to
  // #include mpi.h, which then makes about everything in Uintah
  // depend on mpi.h, so I'm just going to create it here.

  static MPI_Comm   worldComm_ = MPI_Comm(-1);

}  // namespace Uintah


//_____________________________________________________________________________
//
static
void
MpiError( char * what, int errorcode )
{
  // Simple error handling for now...
  int  resultlen = -1;
  char string_name[ MPI_MAX_ERROR_STRING ];

  Uintah::MPI::Error_string( errorcode, string_name, &resultlen );
  std::cerr << "MPI Error in " << what << ": " << string_name << '\n';

  std::exit(1);
}

//_____________________________________________________________________________
//
bool
Parallel::usingMPI()
{
  // TODO: Remove this method once all prior usage of Parallel::usingMPI() is gone - APH 09/17/16
  //         We now assume this to be an invariant for Uintah, and hence this is always true.
  return true;
}

//_____________________________________________________________________________
//
bool
Parallel::usingDevice()
{
  return s_using_device;
}

//_____________________________________________________________________________
//
void
Parallel::setUsingDevice( bool state )
{
  s_using_device = state;
}

//_____________________________________________________________________________
//
int
Parallel::getNumThreads()
{
  return s_num_threads;
}

//_____________________________________________________________________________
//
int
Parallel::getNumPartitions()
{
  return s_num_partitions;
}

//_____________________________________________________________________________
//
int
Parallel::getThreadsPerPartition()
{
  return s_threads_per_partition;
}

//_____________________________________________________________________________
//
std::thread::id
Parallel::getMainThreadID()
{
  return s_main_thread_id;
}

//_____________________________________________________________________________
//
void
Parallel::setNumThreads( int num )
{
  s_num_threads = num;
}

//_____________________________________________________________________________
//
void
Parallel::setNumPartitions( int num )
{
  s_num_partitions = num;
}

//_____________________________________________________________________________
//
void
Parallel::setThreadsPerPartition( int num )
{
  s_threads_per_partition = num;
}

//_____________________________________________________________________________
//
bool
Parallel::isInitialized()
{
  return s_initialized;
}

//_____________________________________________________________________________
//
void
Parallel::initializeManager( int& argc , char**& argv )
{
  s_initialized = true;

  if (s_world_rank != -1) {  // IF ALREADY INITIALIZED, JUST RETURN...
    return;
    // If s_world_rank is not -1, then we have already been initialized..
    // This only happens (I think) if usage() is called (due to bad
    // input parameters (to sus)) and usage() needs to init MPI so that
    // it only displays the usage to the root process.
  }

#ifdef UINTAH_ENABLE_KOKKOS
  if (s_num_partitions <= 0) {
    const char* num_cores = getenv("HPCBIND_NUM_CORES");
    if (num_cores != nullptr) {
      s_num_partitions = atoi(num_cores);
    }
    else {
#ifdef _OPENMP
      s_num_partitions = omp_get_max_threads();
#else
      s_num_partitions = 1;
#endif
    }
  }
  if (s_threads_per_partition <= 0) {
#ifdef _OPENMP
    s_threads_per_partition = omp_get_max_threads() / getNumPartitions();
#else
    s_threads_per_partition = 1;
#endif
  }
#endif

#ifdef THREADED_MPI_AVAILABLE
  int provided = -1;
  int required = MPI_THREAD_SINGLE;
  if ( s_num_threads > 0 || s_num_partitions > 0 ) {
    required = MPI_THREAD_MULTIPLE;
  }
  else {
    required = MPI_THREAD_SINGLE;
  }
#endif

  int status;

#if ( !defined( DISABLE_SCI_MALLOC ) )
  const char* oldtag = Uintah::AllocatorSetDefaultTagMalloc("MPI initialization");
#endif
#ifdef THREADED_MPI_AVAILABLE
  if ((status = Uintah::MPI::Init_thread(&argc, &argv, required, &provided)) != MPI_SUCCESS) {
#else
    if( ( status = Uintah::MPI::Init( &argc, &argv ) ) != MPI_SUCCESS) {
#endif
    MpiError(const_cast<char*>("Uinath::MPI::Init"), status);
  }


#ifdef THREADED_MPI_AVAILABLE
  if (provided < required) {
    std::cerr << "Provided MPI parallel support of " << provided << " is not enough for the required level of " << required << "\n"
              << "To use multi-threaded scheduler, your MPI implementation needs to support MPI_THREAD_MULTIPLE (level-3)"
              << std::endl;
    throw InternalError("Bad MPI level", __FILE__, __LINE__);
  }
#endif


  Uintah::worldComm_ = MPI_COMM_WORLD;
  if ((status = Uintah::MPI::Comm_size(Uintah::worldComm_, &s_world_size)) != MPI_SUCCESS) {
    MpiError(const_cast<char*>("Uintah::MPI::Comm_size"), status);
  }

  if ((status = Uintah::MPI::Comm_rank(Uintah::worldComm_, &s_world_rank)) != MPI_SUCCESS) {
    MpiError(const_cast<char*>("Uintah::MPI::Comm_rank"), status);
  }


#if ( !defined( DISABLE_SCI_MALLOC ) )
  Uintah::AllocatorSetDefaultTagMalloc(oldtag);
  Uintah::AllocatorMallocStatsAppendNumber( s_world_rank );
#endif


#ifdef UINTAH_ENABLE_KOKKOS
    s_root_context = scinew ProcessorGroup(nullptr, Uintah::worldComm_, s_world_rank, s_world_size, s_num_partitions);
#else
    s_root_context = scinew ProcessorGroup(nullptr, Uintah::worldComm_, s_world_rank, s_world_size, s_num_threads);
#endif

  if (s_root_context->myrank() == 0) {
    std::string plural = (s_root_context->size() > 1) ? "processes" : "process";
    std::cout << "Parallel: " << s_root_context->size() << " MPI " << plural << " (using MPI)\n";


#ifdef THREADED_MPI_AVAILABLE

#ifdef UINTAH_ENABLE_KOKKOS
    if ( s_num_partitions > 0 ) {
      std::string plural = (s_num_partitions > 1) ? "partitions" : "partition";
      std::cout << "Parallel: " << s_num_partitions << " OMP thread " << plural << " per MPI process\n";
    }
    if ( s_threads_per_partition > 0 ) {
      std::string plural = (s_threads_per_partition > 1) ? "threads" : "thread";
      std::cout << "Parallel: " << s_threads_per_partition << " OMP " << plural << " per partition\n";
    }
#else
    if (s_num_threads > 0) {
      std::cout << "Parallel: " << s_num_threads << " threads per MPI process\n";
    }
#endif

    std::cout << "Parallel: MPI Level Required: " << required << ", provided: " << provided << "\n";
#endif
  }

    Uintah::MPI::Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

}

  //_____________________________________________________________________________
  //
int
Parallel::getMPIRank()
{
  if( s_world_rank == -1 ) {
    // Can't throw an exception here because it won't get trapped
    // properly because 'getMPIRank()' is called in the exception
    // handler...
    std::cout << "ERROR:\n";
    std::cout << "ERROR: getMPIRank() called before initializeManager()...\n";
    std::cout << "ERROR:\n";
    exitAll(1);
  }
  return s_world_rank;
}

//_____________________________________________________________________________
//
int
Parallel::getMPISize()
{
  return s_world_size;
}

//_____________________________________________________________________________
//
void
Parallel::finalizeManager( Circumstances circumstances /* = NormalShutdown */ )
{
  static bool finalized = false;

  if (finalized) {
    // Due to convoluted logic, signal, and exception handling,
    // finalizeManager() can be easily/mistakenly called multiple
    // times.  This catches that case and returns harmlessly.
    //
    // (One example of this occurs when Uintah::MPI::Abort causes a SIG_TERM
    // to be thrown, which is caught by Uintah's exit handler, which
    // in turn calls finalizeManager.)
    return;
  }

  finalized = true;

  // s_world_rank is not reset here as even after finalizeManager,
  // some things need to know their rank...

  // Only finalize if MPI was initialized.
  if (s_initialized == false) {
    throw InternalError("Trying to finalize without having MPI initialized", __FILE__, __LINE__);
  }

  if (circumstances == Abort) {
    int errorcode = 1;
    if (s_world_rank == 0) {
      std::cout << "FinalizeManager() called... Calling Uintah::MPI::Abort on rank " << s_world_rank << ".\n";
    }
    std::cerr.flush();
    std::cout.flush();

    double seconds = 1.0;
    
    struct timespec ts;
    ts.tv_sec = (int)seconds;
    ts.tv_nsec = (int)(1.e9 * (seconds - ts.tv_sec));

    nanosleep(&ts, &ts);
    
    Uintah::MPI::Abort(Uintah::worldComm_, errorcode);
  }
  else {
    int status;
    if ((status = Uintah::MPI::Finalize()) != MPI_SUCCESS) {
      MpiError(const_cast<char*>("Uintah::MPI::Finalize"), status);
    }
  }

  if (s_root_context != nullptr) {
    delete s_root_context;
    s_root_context = nullptr;
  }
}

//_____________________________________________________________________________
//
ProcessorGroup*
Parallel::getRootProcessorGroup()
{
   if( s_root_context == nullptr ) {
      throw InternalError("Parallel not initialized", __FILE__, __LINE__);
   }

   return s_root_context;
}

//_____________________________________________________________________________
//
void
Parallel::exitAll( int code )
{
  std::exit(code);
}
