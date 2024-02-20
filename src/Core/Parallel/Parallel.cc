/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <sci_defs/gpu_defs.h>

#if defined(_OPENMP)
  #include <omp.h>
#endif

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <thread>
#include <string>

using namespace Uintah;


#define THREADED_MPI_AVAILABLE

#if defined(__digital__) || defined(_AIX)
#  undef THREADED_MPI_AVAILABLE
#endif

// Default to pthreads unless specified otherwise.
Parallel::CpuThreadEnvironment Parallel::s_cpu_thread_environment =
  Parallel::CpuThreadEnvironment::PTHREADS;

bool             Parallel::s_initialized             = false;
#if defined(HAVE_KOKKOS)
bool             Parallel::s_using_cpu               = false;
#else
bool             Parallel::s_using_cpu               = true;
#endif
bool             Parallel::s_using_device            = false;

std::string      Parallel::s_task_name_to_time       = "";
int              Parallel::s_amount_task_name_expected_to_run = -1;
int              Parallel::s_num_threads             = -1;
int              Parallel::s_num_partitions          = -1;
int              Parallel::s_threads_per_partition   = -1;
int              Parallel::s_world_rank              = -1;
int              Parallel::s_world_size              = -1;

int              Parallel::s_kokkos_instances_per_task =  1;
int              Parallel::s_kokkos_teams_per_league   = -1;
int              Parallel::s_kokkos_leagues_per_loop   = -1;

Parallel::Kokkos_Policy Parallel::s_kokkos_policy =
  Parallel::Kokkos_MDRange_Policy;
int              Parallel::s_kokkos_chunk_size       = -1;
int              Parallel::s_kokkos_tile_i_size      = -1;
int              Parallel::s_kokkos_tile_j_size      = -1;
int              Parallel::s_kokkos_tile_k_size      = -1;

int              Parallel::s_provided      = -1;
int              Parallel::s_required      = -1;

std::thread::id  Parallel::s_main_thread_id = std::this_thread::get_id();
ProcessorGroup*  Parallel::s_root_context   = nullptr;


namespace Uintah {

  // While worldComm_ should be declared in Parallel.h, but it would
  // require #include mpi.h, which then makes most everything in
  // Uintah depend on mpi.h, so create it here.

  static MPI_Comm worldComm_ = MPI_Comm(-1);

}  // namespace Uintah


//_____________________________________________________________________________
//
static
void
MPIError( char * what, int errorcode )
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
Parallel::isInitialized()
{
  return s_initialized;
}

//_____________________________________________________________________________
//
bool
Parallel::usingMPI()
{
  // TODO: Remove this method once all prior usage of
  // Parallel::usingMPI() is gone - APH 09/17/16

  // We now assume this to be an invariant for Uintah, and hence this
  // is always true.
  return true;
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
int
Parallel::getMPIRank()
{
  if(s_world_rank == -1) {
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
ProcessorGroup*
Parallel::getRootProcessorGroup()
{
   if(s_root_context == nullptr) {
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
//_____________________________________________________________________________
//
void
Parallel::setCpuThreadEnvironment( CpuThreadEnvironment threadType )
{
  s_cpu_thread_environment = threadType;
}

//_____________________________________________________________________________
//
Parallel::CpuThreadEnvironment
Parallel::getCpuThreadEnvironment()
{
  return s_cpu_thread_environment;
}

//_____________________________________________________________________________
//
void
Parallel::setUsingCPU( bool state )
{
  s_using_cpu = state;
}

//_____________________________________________________________________________
//
bool
Parallel::usingCPU()
{
  return s_using_cpu;
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
bool
Parallel::usingDevice()
{
  return s_using_device;
}

//_____________________________________________________________________________
//
void
Parallel::setTaskNameToTime( const std::string& taskNameToTime )
{
  s_task_name_to_time = taskNameToTime;
}

//_____________________________________________________________________________
//
std::string
Parallel::getTaskNameToTime()
{
  return s_task_name_to_time;
}

//_____________________________________________________________________________
//
void
Parallel::setAmountTaskNameExpectedToRun( unsigned int amountTaskNameExpectedToRun )
{
  s_amount_task_name_expected_to_run = amountTaskNameExpectedToRun;
}

//_____________________________________________________________________________
//
unsigned int
Parallel::getAmountTaskNameExpectedToRun()
{
  return s_amount_task_name_expected_to_run;
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
int
Parallel::getNumThreads()
{
  return s_num_threads;
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
int
Parallel::getNumPartitions()
{
  return s_num_partitions;
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
Parallel::setKokkosInstancesPerTask( unsigned int num )
{
  s_kokkos_instances_per_task = num;
}

//_____________________________________________________________________________
//
unsigned int
Parallel::getKokkosInstancesPerTask()
{
  return s_kokkos_instances_per_task;
}

//_____________________________________________________________________________
//
void
Parallel::setKokkosLeaguesPerLoop( unsigned int num )
{
#if defined(KOKKOS_USING_GPU)
  s_kokkos_leagues_per_loop = num;
#endif
}

//_____________________________________________________________________________
//
unsigned int
Parallel::getKokkosLeaguesPerLoop()
{
  return s_kokkos_leagues_per_loop;
}

//_____________________________________________________________________________
//
void
Parallel::setKokkosTeamsPerLeague( unsigned int num )
{
  s_kokkos_teams_per_league = num;
}

//_____________________________________________________________________________
//
unsigned int
Parallel::getKokkosTeamsPerLeague()
{
  return s_kokkos_teams_per_league;
}

//_____________________________________________________________________________
//  Sets the Kokkos execution policy
void
Parallel::setKokkosPolicy( Parallel::Kokkos_Policy policy )
{
  s_kokkos_policy = policy;
}

Parallel::Kokkos_Policy
Parallel::getKokkosPolicy()
{
  return s_kokkos_policy;
}

//_____________________________________________________________________________
//  Sets/gets the Kokkos chuck size for Kokkos::TeamPolicy & Kokkos::RangePolicy
void
Parallel::setKokkosChunkSize( int size )
{
  s_kokkos_chunk_size = size;
}

int
Parallel::getKokkosChunkSize()
{
#if defined(KOKKOS_USING_GPU)
  if(s_kokkos_chunk_size < 0)
  {
    // Get the default chunk size using a dummy policy.
    if(s_kokkos_policy == Parallel::Kokkos_Team_Policy)
       s_kokkos_chunk_size =
         Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(128, 512).chunk_size();
    else if(s_kokkos_policy == Parallel::Kokkos_Range_Policy)
       s_kokkos_chunk_size =
         Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, int>(0, 512).chunk_size();
  }
#endif

  return s_kokkos_chunk_size;
}

//_____________________________________________________________________________
//  Sets/gets the Kokkos tile size for Kokkos::MDRangePolicy
void Parallel::setKokkosTileSize( int isize, int jsize, int ksize )
{
  s_kokkos_tile_i_size = isize;
  s_kokkos_tile_j_size = jsize;
  s_kokkos_tile_k_size = ksize;

  // Can not be used as Kokkos is not yet initialized.
// #if defined(KOKKOS_USING_GPU)
//   // Use the Kokkos::MDRangePolicy default tile size.
//   Kokkos::DefaultExecutionSpace execSpace;

//   int max_threads =
//     Kokkos::Impl::get_tile_size_properties(execSpace).max_threads;

//   if(isize * jsize * ksize > max_threads)
//   {
//     std::cerr << "The product of tile dimensions ("
//            << isize << "x" << jsize << "x" << ksize << ") "
//            << isize * jsize * ksize << " "
//            << "exceed maximum number of threads per block: " << max_threads
//               << std::endl;

//     throw InternalError("ExecSpace Error: "
//                      "MDRange tile dims exceed maximum number "
//                      "of threads per block - reduce the tile dims",
//                      __FILE__, __LINE__);
//   }
// #endif
}

void Parallel::getKokkosTileSize( int &isize, int &jsize, int &ksize )
{
#if defined(KOKKOS_USING_GPU)
  if(s_kokkos_tile_i_size < 0 ||
     s_kokkos_tile_j_size < 0 ||
     s_kokkos_tile_k_size < 0)
  {
    // Use the Kokkos::MDRangePolicy default tile size.
    Kokkos::DefaultExecutionSpace execSpace;

    // Get the cube root of the max_threads
    int tileSize =
      std::cbrt(Kokkos::Impl::get_tile_size_properties(execSpace).max_threads);

    // Find the largest exponent so the tile size is a power of two.
    unsigned int exp = 0;
    while (tileSize >>= 1)
      exp++;

    if( exp > 1 )
      tileSize = pow(2, exp-1);
    else
      tileSize = 1;

    s_kokkos_tile_i_size = tileSize;
    s_kokkos_tile_j_size = tileSize;
    s_kokkos_tile_k_size = tileSize;
  }
#endif

  isize = s_kokkos_tile_i_size;
  jsize = s_kokkos_tile_j_size;
  ksize = s_kokkos_tile_k_size;
}


//_____________________________________________________________________________
//
void
Parallel::initializeManager( int& argc , char**& argv )
{
  s_initialized = true;

  if(s_world_rank != -1) {  // IF ALREADY INITIALIZED, JUST RETURN...
    return;
    // If s_world_rank is not -1, then already been initialized..
    // This only happens (I think) if usage() is called (due to bad
    // input parameters (to sus)) and usage() needs to init MPI so
    // that it only displays the usage to the root process.
  }

  // Kokkos specific defaults.
#if defined(HAVE_KOKKOS)

#if defined(USE_KOKKOS_PARTITION_MASTER)
  if(s_num_partitions <= 0) {
    s_num_partitions = 1;
  }

  if(s_threads_per_partition <= 0) {
    s_threads_per_partition = 1;
  }
#endif

  // Set GPU parameters (NOTE: This could be autotuned if knowledge of
  // how many patches are assigned to this MPI rank and how many SMs
  // are on this particular machine.)

  // TODO, only display if gpu mode is turned on and if these values
  // weren't set.
#if defined(KOKKOS_USING_GPU)
  if(s_using_device) {
    if(s_kokkos_leagues_per_loop <= 0) {
      s_kokkos_leagues_per_loop = 1;
    }
    if(s_kokkos_teams_per_league <= 0) {
      s_kokkos_teams_per_league = 256;
    }
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  if(s_kokkos_leagues_per_loop <= 0) {
    s_kokkos_leagues_per_loop = 1;
  }
  if(s_kokkos_teams_per_league <= 0) {
    s_kokkos_teams_per_league = 16;
  }
#endif

#endif // Kokkos specific defaults.

#if(!defined( DISABLE_SCI_MALLOC))
  const char* oldtag = Uintah::AllocatorSetDefaultTagMalloc("MPI initialization");
#endif

#ifdef THREADED_MPI_AVAILABLE
  int provided = -1;
  int required = MPI_THREAD_SINGLE;
  if(s_num_threads > 0 || s_num_partitions > 0) {
    required = MPI_THREAD_MULTIPLE;
  }
  else {
    required = MPI_THREAD_SINGLE;
  }

  int status = Uintah::MPI::Init_thread(&argc, &argv, required, &provided);
  if(status != MPI_SUCCESS)
    MPIError(const_cast<char*>("Uinath::MPI::Init"), status);

  s_required = required;
  s_provided = provided;

  if(provided < required) {
    std::cerr << "Provided MPI parallel support of " << provided
              << " is not enough for the required level of " << required << "\n"
              << "To use multi-threaded scheduler, "
              << "your MPI implementation needs to support "
              << "MPI_THREAD_MULTIPLE (level-3)"
              << std::endl;
    throw InternalError("Bad MPI level", __FILE__, __LINE__);
  }
#else
  int status = Uintah::MPI::Init(&argc, &argv);
  if(status != MPI_SUCCESS)
    MPIError(const_cast<char*>("Uinath::MPI::Init"), status);
#endif

  Uintah::worldComm_ = MPI_COMM_WORLD;

  status = Uintah::MPI::Comm_size(Uintah::worldComm_, &s_world_size);
  if(status != MPI_SUCCESS)
    MPIError(const_cast<char*>("Uintah::MPI::Comm_size"), status);

  status = Uintah::MPI::Comm_rank(Uintah::worldComm_, &s_world_rank);
  if(status != MPI_SUCCESS)
    MPIError(const_cast<char*>("Uintah::MPI::Comm_rank"), status);

#if(!defined( DISABLE_SCI_MALLOC))
  Uintah::AllocatorSetDefaultTagMalloc(oldtag);
  Uintah::AllocatorMallocStatsAppendNumber( s_world_rank );
#endif

#if defined(USE_KOKKOS_PARTITION_MASTER)
  s_root_context = scinew ProcessorGroup(nullptr, Uintah::worldComm_, s_world_rank, s_world_size, s_num_partitions);
#else
  s_root_context = scinew ProcessorGroup(nullptr, Uintah::worldComm_, s_world_rank, s_world_size, s_num_threads);
#endif
}

//_____________________________________________________________________________
//
void
Parallel::printManager()
{
  if(s_root_context->myRank() == 0) {
    std::string plural = (s_root_context->nRanks() > 1) ? "es" : "";
    proc0cout << "Parallel CPU MPI process" << plural
              << " (using MPI): \t" << s_root_context->nRanks() << std::endl;

    proc0cout << "Parallel CPU MPI Level Required: " << s_required
              << ", Provided: " << s_provided << std::endl;

#if defined(THREADED_MPI_AVAILABLE)

#if defined(_OPENMP)

#if defined(HAVE_KOKKOS)
    if( s_using_cpu ) {
      if(s_num_threads > 0) { // Unified Scheduler
        proc0cout << "Parallel CPU std::threads per MPI process: \t"
                  << s_num_threads << std::endl;
      } else { // MPI Scheduler
        proc0cout << "Serial CPU execution" << std::endl;
      }

    } else { // Kokkos or KokkosOpenMP scheduler
#if defined(USE_KOKKOS_PARTITION_MASTER)
      if(s_num_partitions > 1 || s_threads_per_partition > 1)
      {
        proc0cout << "Kokkos::OpenMP::partition_master" << std::endl;
        if(s_num_partitions > 1) {
          proc0cout << "Kokkos::OpenMP::partition_master thread partitions per MPI process: \t"
                    << s_num_partitions
                    << << std::endl;
        }

        if(s_threads_per_partition > 1) {
          proc0cout << "Kokkos::OpenMP::partition_master threads per thread partition: \t\t"
                    << s_threads_per_partition
                    << << std::endl;
        }
      }
#else  // OMP Parallel
      if(s_num_threads > 0)
      {
#if _OPENMP >= 201511
        if(omp_get_max_active_levels() > 1)
#else
        if(omp_get_nested())
#endif
        {
          proc0cout << "OpenMP parallel execution" << std::endl;
	  proc0cout << "OpenMP threads per MPI process: \t" << s_num_threads
		    << std::endl;
        } else {
          proc0cout << "Serial CPU execution (OpenMP active levels is one)"
                    << std::endl;
	  proc0cout << "OpenMP threads per MPI process: \t" << s_num_threads
                    << " is set but will not be used!!!!!!!" << std::endl;
        }
      }
#endif  // OMP Parallel
      else
      {
        proc0cout << "Serial CPU execution" << std::endl;
      }
    }
#else  // !defined(HAVE_KOKKOS)
    if(s_num_threads > 0) { // Unified Scheduler
      proc0cout << "Parallel CPU std::threads per MPI process: \t"
                << s_num_threads << std::endl;
    } else { // Unified Scheduler
      proc0cout << "Serial CPU execution" << std::endl;
    }
#endif  // defined(HAVE_KOKKOS)

#else   // !defined(_OPENMP)
      proc0cout << "Serial CPU execution" << std::endl;
#endif  // !defined(_OPENMP)

#else  // !defined(THREADED_MPI_AVAILABLE)
      proc0cout << "Serial CPU execution" << std::endl;
#endif  // !defined(THREADED_MPI_AVAILABLE)
  }
//    Uintah::MPI::Errhandler_set(Uintah::worldComm_, MPI_ERRORS_RETURN);
}

//_____________________________________________________________________________
//
void
Parallel::finalizeManager( Circumstances circumstances /* = NormalShutdown */ )
{
  static bool finalized = false;

  if(finalized) {
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
  if(s_initialized == false) {
    throw InternalError("Trying to finalize without having MPI initialized",
                        __FILE__, __LINE__);
  }

  if(circumstances == Abort) {
    int errorcode = 1;
    if(s_world_rank == 0) {
      std::cout << "FinalizeManager() called... "
                << "Calling Uintah::MPI::Abort on rank "
                << s_world_rank << ".\n";
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
    if((status = Uintah::MPI::Finalize()) != MPI_SUCCESS) {
      MPIError(const_cast<char*>("Uintah::MPI::Finalize"), status);
    }
  }

  if(s_root_context != nullptr) {
    delete s_root_context;
    s_root_context = nullptr;
  }
}
