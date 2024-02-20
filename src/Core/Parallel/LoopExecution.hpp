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

// The purpose of this file is to provide portability between Kokkos
// and non-Kokkos builds.  The user should be able to #include this
// file, and then obtain all the tools needed.  For example, suppose a
// user calls a parallel_for loop but Kokkos is NOT provided, this
// will run the functor in a loop and also not use Kokkos views.  If
// Kokkos is provided, this creates a lambda expression and inside
// that it contains loops over the functor.  Kokkos Views are also
// used.  At the moment regular CPU code and Kokkos execution spaces
// are supported.

#ifndef UINTAH_LOOP_EXECUTION_HPP
#define UINTAH_LOOP_EXECUTION_HPP

#include <Core/Parallel/ExecutionObject.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InternalError.h>

#include <sci_defs/gpu_defs.h>

#include <algorithm>
#include <cstring>
#include <cxxabi.h>
#include <initializer_list>
#include <typeinfo>
#include <vector> //  Used to manage multiple streams in a task.
#include <cstddef> // TODO: What is this doing here?

#define ARRAY_SIZE 16

enum TASKGRAPH {
  DEFAULT = -1
};

// Macros don't like passing in data types that contain commas in
// them, such as two template arguments. This helps fix that.
#define COMMA ,

// Helps turn defines into usable strings (even if it has a comma in it)
#define STRV(...) #__VA_ARGS__
#define STRVX(...) STRV(__VA_ARGS__)

// Example of Kokkos options following Kokkos internal order:
//     https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_Core_fwd.hpp
//     https://github.com/kokkos/kokkos/wiki/Initialization

// Execution Space                            Memory Space
// -------------------------------------      --------------
// Kokkos::DefaultHostExecutionSpace          Kokkos::DefaultHostExecutionSpace::memory_space
// Kokkos::DefaultExecutionSpace              Kokkos::DefaultExecutionSpace::memory_space

// Kokkos chooses the two spaces using the following list:
//
// 1. Kokkos::Cuda                            Kokkos::CudaSpace
// 2. Kokkos::Experimental::OpenMPTarget      Kokkos::Experimental::OpenMPTargetSpace
// 3. Kokkos::Experimental::OpenACC           Kokkos::Experimental::OpenACC
// 4. Kokkos::Experimental::HIP               Kokkos::Experimental::HIPSpace
// 5. Kokkos::Experimental::SYCL              Kokkos::Experimental::SYCLDeviceUSMSpace
// 6. Kokkos::OpenMP                          Kokkos::HostSpace
// 7. Kokkos::Threads                         Kokkos::HostSpace
// 8. Kokkos::Experimental::HPX               Kokkos::HostSpace
// 9. Kokkos::Serial                          Kokkos::HostSpace
//

// The highest execution space in the list which is enabled is Kokkos'
// default execution space, and the highest enabled host execution
// space is Kokkos' default host execution space.

#if defined(HAVE_KOKKOS)
// Host side.
  #if defined(KOKKOS_ENABLE_OPENMP)
    #define UINTAH_CPU_TAG            Kokkos::OpenMP COMMA Kokkos::HostSpace
    #define KOKKOS_OPENMP_TAG         Kokkos::OpenMP COMMA Kokkos::HostSpace
    #define KOKKOS_DEFAULT_HOST_TAG   Kokkos::DefaultHostExecutionSpace COMMA Kokkos::DefaultHostExecutionSpace::memory_space
//  |- This should default into:      Kokkos::OpenMP COMMA Kokkos::HostSpace
  #else
    #define UINTAH_CPU_TAG            UintahSpaces::CPU COMMA UintahSpaces::HostSpace
    #define KOKKOS_OPENMP_TAG         UintahSpaces::CPU COMMA UintahSpaces::HostSpace
    #define KOKKOS_DEFAULT_HOST_TAG   UintahSpaces::CPU COMMA UintahSpaces::HostSpace
    #endif

// Device side
  #if defined(KOKKOS_USING_GPU)
    #define KOKKOS_DEFAULT_DEVICE_TAG   Kokkos::DefaultExecutionSpace COMMA Kokkos::DefaultExecutionSpace::memory_space

  #else // !defined(KOKKOS_USING_GPU)
    #if defined(KOKKOS_ENABLE_OPENMP)
      #define KOKKOS_DEFAULT_DEVICE_TAG Kokkos::DefaultExecutionSpace COMMA Kokkos::DefaultExecutionSpace::memory_space
//    |- This should default into:      Kokkos::OpenMP COMMA Kokkos::HostSpace

    #else
      #define KOKKOS_DEFAULT_DEVICE_TAG UintahSpaces::CPU COMMA UintahSpaces::HostSpace
    #endif
  #endif
#else // !defined(HAVE_KOKKOS)
  #define UINTAH_CPU_TAG              UintahSpaces::CPU COMMA UintahSpaces::HostSpace
  #define KOKKOS_OPENMP_TAG           UintahSpaces::CPU COMMA UintahSpaces::HostSpace
  #define KOKKOS_DEFAULT_HOST_TAG     UintahSpaces::CPU COMMA UintahSpaces::HostSpace
  #define KOKKOS_DEFAULT_DEVICE_TAG   UintahSpaces::CPU COMMA UintahSpaces::HostSpace
#endif

// #pragma message "The value of UINTAH_CPU_TAG: "            STRVX(UINTAH_CPU_TAG)
// #pragma message "The value of KOKKOS_OPENMP_TAG: "         STRVX(KOKKOS_OPENMP_TAG)
// #pragma message "The value of KOKKOS_DEFAULT_HOST_TAG: "   STRVX(KOKKOS_DEFAULT_HOST_TAG)
// #pragma message "The value of KOKKOS_DEFAULT_DEVICE_TAG: " STRVX(KOKKOS_DEFAULT_DEVICE_TAG)

namespace Uintah {

class BlockRange;

class BlockRange
{
public:

  enum { rank = 3 };

  BlockRange() {}

  template <typename ArrayType>
  void setValues(ArrayType const & c0, ArrayType const & c1)
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
      m_dim[i] =   (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
    }
  }

  template <typename ArrayType>
  BlockRange(ArrayType const & c0, ArrayType const & c1)
  {
    setValues(c0, c1);
  }

  BlockRange(const BlockRange& obj) {
    for (int i=0; i<rank; ++i) {
      this->m_offset[i] = obj.m_offset[i];
      this->m_dim[i] = obj.m_dim[i];
    }
  }

  int begin(int r) const { return m_offset[r]; }
  int   end(int r) const { return m_offset[r] + m_dim[r]; }

  size_t size() const
  {
    size_t result = 1u;
    for (int i=0; i<rank; ++i) {
      result *= m_dim[i];
    }
    return result;
  }

private:
  int m_offset[rank];
  int m_dim[rank];
};

// Lambda expressions for Kokkos cannot properly capture plain fixed
// sized arrays (it passes pointers, not the arrays) but they can
// properly capture and copy a struct of arrays.  These arrays have
// sizes known at compile time.  For Kokkos, this struct containing an
// array will do a full clone/by value copy as part of the lambda
// capture.  If you require a runtime/variable sized array, that
// requires a different mechanism involving pools and deep copies, and
// as of yet hasn't been implemented (Brad P.)
template <typename T, unsigned int CAPACITY>
struct struct1DArray
{
  unsigned short int runTime_size{CAPACITY};
  T arr[CAPACITY];
  GPU_INLINE_FUNCTION struct1DArray() {}

  // This constructor copies elements from one container into here.
  template <typename Container>
  GPU_INLINE_FUNCTION struct1DArray(const Container& container, unsigned int runTimeSize) : runTime_size(runTimeSize) {
// #ifndef NDEBUG
//     if(runTime_size > CAPACITY){
//       throw InternalError("ERROR. struct1DArray is not being used properly. The run-time size exceeds the compile time size (std::vector constructor).", __FILE__, __LINE__);
//     }
// #endif
    for (unsigned int i = 0; i < runTime_size; i++) {
      arr[i] = container[i];
    }
  }

// This constructor supports the initialization list interface
  struct1DArray(std::initializer_list<T> const myList)
    : runTime_size(myList.size()) {
// #ifndef NDEBUG
//     if(runTime_size > CAPACITY){
//       throw InternalError("ERROR. struct1DArray is not being used properly. The run-time size exceeds the compile time size (initializer_list constructor).", __FILE__, __LINE__);
//     }
// #endif
    std::copy(myList.begin(), myList.begin()+runTime_size,arr);
  }

// This constructor allows for only the runtime_size to be specified
  GPU_INLINE_FUNCTION struct1DArray(int runTimeSize) : runTime_size(runTimeSize) {
// #ifndef NDEBUG
//     if(runTime_size > CAPACITY){
//       throw InternalError("ERROR. struct1DArray is not being used properly. The run-time size exceeds the compile time size (int constructor).", __FILE__, __LINE__);
//     }
// #endif
  }

  GPU_INLINE_FUNCTION T& operator[](unsigned int index) {
    return arr[index];
  }

  GPU_INLINE_FUNCTION const T& operator[](unsigned int index) const {
    return arr[index];
  }

  GPU_INLINE_FUNCTION int mySize(){
    return CAPACITY;
  }

}; // end struct struct1DArray

template <typename T, unsigned int CAPACITY_FIRST_DIMENSION, unsigned int CAPACITY_SECOND_DIMENSION>
struct struct2DArray
{
  struct1DArray<T, CAPACITY_SECOND_DIMENSION> arr[CAPACITY_FIRST_DIMENSION];

  GPU_INLINE_FUNCTION struct2DArray() {}
  unsigned short int i_runTime_size{CAPACITY_FIRST_DIMENSION};
  unsigned short int j_runTime_size{CAPACITY_SECOND_DIMENSION};

  // This constructor copies elements from one container into here.
  template <typename Container>
  GPU_INLINE_FUNCTION struct2DArray(const Container& container,
                            int first_dim_runtimeSize = CAPACITY_FIRST_DIMENSION,
                            int second_dim_runtimeSize = CAPACITY_SECOND_DIMENSION) :
    i_runTime_size(first_dim_runtimeSize), j_runTime_size(second_dim_runtimeSize)
  {
    for (unsigned int i = 0; i < i_runTime_size; i++) {
      for (unsigned int j = 0; j < j_runTime_size; j++) {
        arr[i][j] = container[i][j];
      }
      arr[i].runTime_size=i_runTime_size;
    }
  }

  GPU_INLINE_FUNCTION struct1DArray<T, CAPACITY_SECOND_DIMENSION>& operator[](unsigned int index) {
    return arr[index];
  }

  GPU_INLINE_FUNCTION const struct1DArray<T, CAPACITY_SECOND_DIMENSION>& operator[](unsigned int index) const {
    return arr[index];
  }

}; // end struct struct2DArray

//----------------------------------------------------------------------------
// Start parallel loops
//----------------------------------------------------------------------------

#if defined(KOKKOS_ENABLE_OPENMP)
template <typename ExecSpace, typename MemSpace>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, ExecSpace>::type
getInstance(ExecutionObject<ExecSpace, MemSpace>& execObj, int index = 0)
{
  ExecSpace instanceObject;
  return instanceObject;
}
#endif

#if defined(KOKKOS_USING_GPU)
template <typename ExecSpace, typename MemSpace>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, ExecSpace>::type
getInstance(ExecutionObject<ExecSpace, MemSpace>& execObj, int index = 0)
{
  return execObj.getInstance(index);
}
#endif

//----------------------------------------------------------------------------
// Parallel_for loops
//
// CPU
// No ExecSpace/MemSpace
// OpenMP       - Range (default), MDRange and Team Policy
// GPU          - Team (default), Range and MDRange Policy
//----------------------------------------------------------------------------

// CPU - parallel_for
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor)
{
  int status;
  char *name(abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status));

  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  for (int k=rbegin2; k<rend2; ++k) {
    for (int j=rbegin1; j<rend1; ++j) {
      for (int i=rbegin0; i<rend0; ++i) {
        functor(i, j, k);
      }
    }
  }

  std::free( name );
}

// CPU - parallel_for - For legacy loops where no execution space was
// specified as a template parameter.
template <typename Functor>
inline void
parallel_for(BlockRange const & r, const Functor & functor)
{
  // Force users into using a single CPU thread if they didn't specify
  // OpenMP.

   // Make an empty object
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj;
  parallel_for<UintahSpaces::CPU>(execObj, r, functor);
}

// OpenMP - parallel_for
#if defined(KOKKOS_ENABLE_OPENMP__COMMENTED_OUT)

template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj,
             BlockRange const & r, const Functor & functor)
{
  int status;
  char *name(abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status));

  const int i_size = r.end(0) - r.begin(0);
  const int j_size = r.end(1) - r.begin(1);
  const int k_size = r.end(2) - r.begin(2);

  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  const unsigned int numItems = ((i_size > 0 ? i_size : 1) *
                                 (j_size > 0 ? j_size : 1) *
                                 (k_size > 0 ? k_size : 1));

  Parallel::Kokkos_Policy kokkos_policy = Parallel::getKokkosPolicy();

  // Range Policy
  if(kokkos_policy == Parallel::Kokkos_Range_Policy)
  {
    Kokkos::RangePolicy<ExecSpace, int> rangePolicy(0, numItems);

    int size = Parallel::getKokkosChunkSize();
    if(size > 0)
      rangePolicy.set_chunk_size(size);

    Kokkos::parallel_for(name, rangePolicy,
                         KOKKOS_LAMBDA(int n) {
                           const int k = n / (j_size * i_size) + rbegin2;
                           const int j = (n / i_size) % j_size + rbegin1;
                           const int i = n % i_size + rbegin0;

                           functor(i, j, k);
                         });
  }
  // MDRange Policy
  else if(kokkos_policy == Parallel::Kokkos_MDRange_Policy)
  {
    int i_tile, j_tile, k_tile;
    Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

    if(i_tile > 0 || j_tile > 0 || k_tile > 0)
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin0, rbegin1, rbegin2},
                      {rend0,   rend1,   rend2},
                      {i_tile,  j_tile,  k_tile});

      Kokkos::parallel_for(name, mdRangePolicy, functor);
    }
    else
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin0, rbegin1, rbegin2},
                      {rend0,   rend1,   rend2},
                      {i_size,  j_size,  k_size});

      Kokkos::parallel_for(name, mdRangePolicy, functor);
      }
  }
  // MDRange Policy - Reverse index
  else if(kokkos_policy == Parallel::Kokkos_MDRange_Reverse_Policy)
  {
    int i_tile, j_tile, k_tile;
    Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

    if(i_tile > 0 || j_tile > 0 || k_tile > 0)
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin2, rbegin1, rbegin0},
                      {rend2,   rend1,   rend0},
                      {k_tile,  j_tile,  i_tile});

        Kokkos::parallel_for(name, mdRangePolicy,
                             KOKKOS_LAMBDA(int k, int j, int i) {
                               functor(i, j, k);
                             });
    }
    else
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin2, rbegin1, rbegin0},
                      {rend2,   rend1,   rend0},
                      {k_size,  j_size,  i_size});

      Kokkos::parallel_for(name, mdRangePolicy,
                           KOKKOS_LAMBDA(int k, int j, int i) {
                             functor(i, j, k);
                           });
    }
  }
  // Team Policy
  else if(kokkos_policy == Parallel::Kokkos_Team_Policy)
  {
    // Assumption - there is only one league for OpenMP
    const int teams_per_league = Parallel::getKokkosTeamsPerLeague();
    const int team_range_size = numItems;

    const int actualTeams = teams_per_league < team_range_size ?
                            teams_per_league : team_range_size;

    Kokkos::TeamPolicy<ExecSpace> teamPolicy(1, actualTeams);
    typedef Kokkos::TeamPolicy<ExecSpace> policy_type;

    int size = Parallel::getKokkosChunkSize();
    if(size > 0)
      teamPolicy.set_chunk_size(size);

    Kokkos::parallel_for(name, teamPolicy,
                         KOKKOS_LAMBDA(typename policy_type::member_type thread) {
      // printf("i is %d\n", thread.team_rank());
      Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_range_size),
                           [&](const int& n) {
        const int i = n / (j_size * k_size) + rbegin0;
        const int j = (n / k_size) % j_size + rbegin1;
        const int k = n % k_size + rbegin2;

        functor(i, j, k);
      });
    });
  }

  std::free( name );
}

#endif  // #if defined(KOKKOS_ENABLE_OPENMP)

// GPU - parallel_for
#if defined(HAVE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename Functor>

inline typename std::enable_if<
#if defined(KOKKOS_USING_GPU)
  std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value
#endif
#if defined(KOKKOS_USING_GPU) && defined(KOKKOS_ENABLE_OPENMP)
  ||
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  std::is_same<ExecSpace, Kokkos::OpenMP>::value
#endif
  , void>::type

parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor)
{
  int status;
  char *name(abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status));

  const int i_size = r.end(0) - r.begin(0);
  const int j_size = r.end(1) - r.begin(1);
  const int k_size = r.end(2) - r.begin(2);

  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  const unsigned int numItems = ((i_size > 0 ? i_size : 1) *
                                 (j_size > 0 ? j_size : 1) *
                                 (k_size > 0 ? k_size : 1));

  Parallel::Kokkos_Policy kokkos_policy = Parallel::getKokkosPolicy();

  // Team Policy
  if(kokkos_policy == Parallel::Kokkos_Team_Policy)
  {
    // Team policy approach.

    // Overall goal, split a 3D range requested by the user into various
    // SMs on the GPU.  (In essence, this would be a Kokkos
    // MD_Team+Policy, if one existed) The process requires going from
    // 3D range to a 1D range, partitioning the 1D range into groups
    // that are multiples of 32, then converting that group of 32 range
    // back into a 3D (i,j,k) index.

    // The user has two partitions available.  1) One is the total
    // number of streaming multiprocessors.  2) The other is splitting a
    // task into multiple streams and execution units.

    // Get the requested amount of threads per streaming multiprocessor
    // (SM) and number of SMs totals.
    if(std::is_same<ExecSpace, Kokkos::OpenMP>::value)
    {
      // Assumption - there is only one league for OpenMP
      const int teams_per_league = Parallel::getKokkosTeamsPerLeague();
      const int team_range_size = numItems;

      const int actualTeams = teams_per_league < team_range_size ?
                              teams_per_league : team_range_size;

      Kokkos::TeamPolicy<ExecSpace> teamPolicy(1, actualTeams);
      typedef Kokkos::TeamPolicy<ExecSpace> policy_type;

      int size = Parallel::getKokkosChunkSize();
      if(size > 0)
        teamPolicy.set_chunk_size(size);

      Kokkos::parallel_for(name, teamPolicy,
                           KOKKOS_LAMBDA(typename policy_type::member_type thread) {
        // printf("i is %d\n", thread.team_rank());
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_range_size),
                             [&](const int& n) {
          const int i = n / (j_size * k_size) + rbegin0;
          const int j = (n / k_size) % j_size + rbegin1;
          const int k = n % k_size + rbegin2;

          functor(i, j, k);
        });
      });
    }
    else
    {
      const int kokkos_leagues_per_loop = Parallel::getKokkosLeaguesPerLoop();
      const int kokkos_teams_per_league = Parallel::getKokkosTeamsPerLeague();

      const int nPartitions = execObj.getNumInstances();

      // The requested range of data may not have enough work for the
      // requested command line arguments, so shrink them if necessary.
      int teams_per_loop = kokkos_teams_per_league * kokkos_leagues_per_loop;
      int team_range_size = numItems / nPartitions;

      const unsigned int actual_teams =
        teams_per_loop < team_range_size ?
        teams_per_loop : team_range_size;

      const unsigned int actual_teams_per_league =
        kokkos_teams_per_league < team_range_size ?
        kokkos_teams_per_league : team_range_size;

      const unsigned int actual_leagues_per_loop =
        (actual_teams - 1) / kokkos_teams_per_league + 1;

      for (int p = 0; p < nPartitions; p++) {

        ExecSpace instanceObject = getInstance(execObj, p);

        // Use a Team Policy, this allows us to control how many threads
        // per league and how many leagues are used.
        Kokkos::TeamPolicy< ExecSpace > teamPolicy(instanceObject, actual_leagues_per_loop, actual_teams_per_league);
        typedef Kokkos::TeamPolicy< ExecSpace > policy_type;

        int size = Parallel::getKokkosChunkSize();
        if(size > 0)
          teamPolicy.set_chunk_size(size);

        Kokkos::parallel_for(name, teamPolicy,
                             KOKKOS_LAMBDA(typename policy_type::member_type thread) {
          // We are within an SM, and all SMs share the same amount of
          // assigned Kokkos threads.  Figure out which range of N items
          // this SM should work on (as a multiple of 32).
          const unsigned int currentPartition = p * actual_leagues_per_loop + thread.league_rank();
          unsigned int estimatedThreadAmount = numItems * (currentPartition) / (actual_leagues_per_loop * nPartitions);
          const unsigned int startingN =  estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
          unsigned int endingN;

          // Check if this is the last partition
          if(currentPartition + 1 == actual_leagues_per_loop * nPartitions) {
            endingN = numItems;
          } else {
            estimatedThreadAmount = numItems * (currentPartition + 1) / (actual_leagues_per_loop * nPartitions);
            endingN = estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
          }

          const unsigned int totalN = endingN - startingN;
          // printf("league_rank: %d, team_size: %d, team_rank: %d, startingN: %d, endingN: %d, totalN: %d\n", thread.league_rank(), thread.team_size(), thread.team_rank(), startingN, endingN, totalN);

          Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, totalN),
                               [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2] (const int& N) {
            // Craft an i,j,k out of this range.  This approach works with
            // row-major layout so that consecutive Kokkos threads work
            // along consecutive slots in memory.

            // printf("parallel_for team demo - n is %d, league_rank is %d, true n is %d\n", N, thread.league_rank(), (startingN + N));

            const int i = ((startingN + N) % i_size)           + rbegin0;
            const int j = ((startingN + N) / i_size) % j_size  + rbegin1;
            const int k =  (startingN + N) / (i_size * j_size) + rbegin2;

            functor(i, j, k);
          });
        });
      }
    }
  }
  else
  {
    ExecSpace instanceObject = getInstance(execObj);

    // Range Policy
    if(kokkos_policy == Parallel::Kokkos_Range_Policy)
    {
      Kokkos::RangePolicy<ExecSpace> rangePolicy(instanceObject, 0, numItems);

      int size = Parallel::getKokkosChunkSize();
      if(size > 0)
        rangePolicy.set_chunk_size(size);

      Kokkos::parallel_for(name, rangePolicy,
                           KOKKOS_LAMBDA(int n) {
                             const int k = n / (j_size * i_size) + rbegin2;
                             const int j = (n / i_size) % j_size + rbegin1;
                             const int i = n % i_size + rbegin0;

                             functor(i, j, k);
                           });
    }
    // MDRange Policy
    else if(kokkos_policy == Parallel::Kokkos_MDRange_Policy)
    {
      int i_tile, j_tile, k_tile;
      Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

      if(i_tile > 0 || j_tile > 0 || k_tile > 0)
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin0, rbegin1, rbegin2},
                        {rend0,   rend1,   rend2},
                        {i_tile,  j_tile,  k_tile});

        Kokkos::parallel_for(name, mdRangePolicy, functor);
      }
      else
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin0, rbegin1, rbegin2},
                        {rend0,   rend1,   rend2},
                        {i_size,  j_size,  k_size});

        Kokkos::parallel_for(name, mdRangePolicy, functor);
      }
    }
    // MDRange Policy - Reverse index
    else if(kokkos_policy == Parallel::Kokkos_MDRange_Reverse_Policy)
    {
      int i_tile, j_tile, k_tile;
      Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

      if(i_tile > 0 || j_tile > 0 || k_tile > 0)
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin2, rbegin1, rbegin0},
                        {rend2,   rend1,   rend0},
                        {k_tile,  j_tile,  i_tile});

        Kokkos::parallel_for(name, mdRangePolicy,
                             KOKKOS_LAMBDA(int k, int j, int i) {
                               functor(i, j, k);
                             });
      }
      else
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin2, rbegin1, rbegin0},
                        {rend2,   rend1,   rend0},
                        {k_size,  j_size,  i_size});

        Kokkos::parallel_for(name, mdRangePolicy,
                             KOKKOS_LAMBDA(int k, int j, int i) {
                               functor(i, j, k);
                             });
      }
    }
  }

  std::free( name );
}
#endif  // #if defined(HAVE_KOKKOS)

//----------------------------------------------------------------------------
// Parallel_reduce_sum loops
//
// CPU
// No ExecSpace/MemSpace
// OpenMP       - Range (default), MDRange and Team Policy
// GPU          - Team (default), Range and MDRange Policy
//----------------------------------------------------------------------------

// CPU parallel_reduce_sum
template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_reduce_sum(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, ReductionType & red)
{
  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  for (int k=rbegin2; k<rend2; ++k) {
    for (int j=rbegin1; j<rend1; ++j) {
      for (int i=rbegin0; i<rend0; ++i) {

        ReductionType tmp = 0;
        functor(i, j, k, tmp);

        red += tmp;
      }
    }
  }
}

// CPU parallel_reduce_sum - For legacy loops where no execution space
// was specified as a template parameter.
template < typename Functor, typename ReductionType>
inline void
parallel_reduce_sum(BlockRange const & r, const Functor & functor, ReductionType & red)
{
  // Force users into using a single CPU thread if they didn't specify
  // OpenMP.

  // Make an empty object
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj;

  parallel_reduce_sum<UintahSpaces::CPU>(execObj, r, functor, red);
}

// OpenMP parallel_reduce_sum
#if defined(KOKKOS_ENABLE_OPENMP__COMMENTED_OUT)

template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_reduce_sum(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    BlockRange const & r, const Functor & functor, ReductionType & red)
{
  int status;
  char *name(abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status));

  const int i_size = r.end(0) - r.begin(0);
  const int j_size = r.end(1) - r.begin(1);
  const int k_size = r.end(2) - r.begin(2);

  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  const unsigned int numItems = ((i_size > 0 ? i_size : 1) *
                                 (j_size > 0 ? j_size : 1) *
                                 (k_size > 0 ? k_size : 1));

  Parallel::Kokkos_Policy kokkos_policy = Parallel::getKokkosPolicy();

  // Range Policy
  if(kokkos_policy == Parallel::Kokkos_Range_Policy)
  {
    Kokkos::RangePolicy<ExecSpace, int> rangePolicy(0, numItems);

    int size = Parallel::getKokkosChunkSize();
    if(size > 0)
      rangePolicy.set_chunk_size(size);

    Kokkos::parallel_reduce(name, rangePolicy,
                            KOKKOS_LAMBDA(const int& n, ReductionType & tmp) {
                              const int k = n / (j_size * i_size) + rbegin2;
                              const int j = (n / i_size) % j_size + rbegin1;
                              const int i = n % i_size + rbegin0;

                              functor(i, j, k, tmp);
                            }, red);
  }
  // MDRange Policy
  else if(kokkos_policy == Parallel::Kokkos_MDRange_Policy)
  {
    int i_tile, j_tile, k_tile;
    Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

    if(i_tile > 0 || j_tile > 0 || k_tile > 0)
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin0, rbegin1, rbegin2},
                      {rend0,   rend1,   rend2},
                      {i_tile,  j_tile,  k_tile});

      Kokkos::parallel_reduce(name, mdRangePolicy, functor, red);
    }
    else
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin0, rbegin1, rbegin2},
                      {rend0,   rend1,   rend2},
                      {i_size,  j_size,  k_size});

      Kokkos::parallel_reduce(name, mdRangePolicy, functor, red);
    }
  }
  // MDRange Policy - Reverse index
  else if(kokkos_policy == Parallel::Kokkos_MDRange_Reverse_Policy)
  {
    int i_tile, j_tile, k_tile;
    Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

    if(i_tile > 0 || j_tile > 0 || k_tile > 0)
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin2, rbegin1, rbegin0},
                      {rend2,   rend1,   rend0},
                      {k_tile,  j_tile,  i_tile});

      Kokkos::parallel_reduce(name, mdRangePolicy,
                              KOKKOS_LAMBDA(int k, int j, int i,
                                            ReductionType & tmp) {
                                functor(i, j, k, tmp);
                              }, red);
    }
    else
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin2, rbegin1, rbegin0},
                      {rend2,   rend1,   rend0},
                      {k_size,  j_size,  i_size});

      Kokkos::parallel_reduce(name, mdRangePolicy,
                              KOKKOS_LAMBDA(int k, int j, int i,
                                            ReductionType& tmp) {
                                functor(i, j, k, tmp);
                              }, red);
    }
  }
  // Team Policy
  else if(kokkos_policy == Parallel::Kokkos_Team_Policy)
  {
    // Assumption - there is only one league for the OpenMP
    const int teams_per_league = Parallel::getKokkosTeamsPerLeague();
    const int team_range_size = numItems;

    const int actualTeams = teams_per_league < team_range_size ?
                            teams_per_league : team_range_size;

    Kokkos::TeamPolicy<ExecSpace> teamPolicy(1, actualTeams);
    typedef Kokkos::TeamPolicy<ExecSpace> policy_type;

    int size = Parallel::getKokkosChunkSize();
    if(size > 0)
      teamPolicy.set_chunk_size(size);

    Kokkos::parallel_reduce(name, teamPolicy,
                            KOKKOS_LAMBDA(typename policy_type::member_type thread, ReductionType& inner_sum) {
      // printf("i is %d\n", thread.team_rank());
      Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_range_size),
                           [&](const int& n) {
        const int i = n / (j_size * k_size) + rbegin0;
        const int j = (n / k_size) % j_size + rbegin1;
        const int k = n % k_size + rbegin2;

        ReductionType tmp = 0;
        functor(i, j, k, tmp);

        inner_sum += tmp;
      });
    }, red);
  }

  std::free( name );
}

#endif  // #if defined(KOKKOS_ENABLE_OPENMP

// GPU parallel_reduce_sum
#if defined(HAVE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>

inline typename std::enable_if<
#if defined(KOKKOS_USING_GPU)
  std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value
#endif
#if defined(KOKKOS_USING_GPU) && defined(KOKKOS_ENABLE_OPENMP)
  ||
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  std::is_same<ExecSpace, Kokkos::OpenMP>::value
#endif
  , void>::type

parallel_reduce_sum(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, ReductionType & red)
{
  int status;
  char *name = abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status);

  // Overall goal, split a 3D range requested by the user into various
  // SMs on the GPU.  (In essence, this would be a Kokkos
  // MD_Team+Policy, if one existed) The process requires going from
  // 3D range to a 1D range, partitioning the 1D range into groups
  // that are multiples of 32, then converting that group of 32 range
  // back into a 3D (i,j,k) index.
  const int i_size = r.end(0) - r.begin(0);
  const int j_size = r.end(1) - r.begin(1);
  const int k_size = r.end(2) - r.begin(2);

  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  const unsigned int numItems = ((i_size > 0 ? i_size : 1) *
                                 (j_size > 0 ? j_size : 1) *
                                 (k_size > 0 ? k_size : 1));

  Parallel::Kokkos_Policy kokkos_policy = Parallel::getKokkosPolicy();

  // Team Policy
  if(kokkos_policy == Parallel::Kokkos_Team_Policy)
  {
    // Team policy approach.

    // Overall goal, split a 3D range requested by the user into various
    // SMs on the GPU.  (In essence, this would be a Kokkos
    // MD_Team+Policy, if one existed) The process requires going from
    // 3D range to a 1D range, partitioning the 1D range into groups
    // that are multiples of 32, then converting that group of 32 range
    // back into a 3D (i,j,k) index.

    // The user has two partitions available.  1) One is the total
    // number of streaming multiprocessors.  2) The other is splitting a
    // task into multiple streams and execution units.

    // Get the requested amount of threads per streaming multiprocessor
    // (SM) and number of SMs totals.
    if (std::is_same<ExecSpace, Kokkos::OpenMP>::value)
    {
      // Assumption - there is only one league for the OpenMP
      const int teams_per_league = Parallel::getKokkosTeamsPerLeague();
      const int team_range_size = numItems;

      const int actualTeams = teams_per_league < team_range_size ?
                              teams_per_league : team_range_size;

      Kokkos::TeamPolicy<ExecSpace> teamPolicy(1, actualTeams);
      typedef Kokkos::TeamPolicy<ExecSpace> policy_type;

      int size = Parallel::getKokkosChunkSize();
      if(size > 0)
        teamPolicy.set_chunk_size(size);

      Kokkos::parallel_reduce(name, teamPolicy,
                              KOKKOS_LAMBDA(typename policy_type::member_type thread, ReductionType& inner_sum) {
        // printf("i is %d\n", thread.team_rank());
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_range_size),
                             [&](const int& n) {
          const int i = n / (j_size * k_size) + rbegin0;
          const int j = (n / k_size) % j_size + rbegin1;
          const int k = n % k_size + rbegin2;

          ReductionType tmp = 0;
          functor(i, j, k, tmp);

          inner_sum += tmp;
        });
      }, red);
    }
    else
    {
      const int kokkos_leagues_per_loop = Parallel::getKokkosLeaguesPerLoop();
      const int kokkos_teams_per_league = Parallel::getKokkosTeamsPerLeague();

      const int nPartitions = execObj.getNumInstances();

      // The requested range of data may not have enough work for the
      // requested command line arguments, so shrink them if necessary.
      int teams_per_loop = kokkos_teams_per_league * kokkos_leagues_per_loop;
      int team_range_size = numItems / nPartitions;

      const unsigned int actual_teams =
        teams_per_loop < team_range_size ?
        teams_per_loop : team_range_size;

      const unsigned int actual_teams_per_league =
        kokkos_teams_per_league < team_range_size ?
        kokkos_teams_per_league : team_range_size;

      const unsigned int actual_leagues_per_loop =
        (actual_teams - 1) / kokkos_teams_per_league + 1;

      for (int p = 0; p < nPartitions; p++) {
        ReductionType tmp0 = 0;

        ExecSpace instanceObject = getInstance(execObj, p);

        // Use a Team Policy, this allows us to control how many threads
        // per league and how many leagues are used.
        Kokkos::TeamPolicy< ExecSpace > teamPolicy(instanceObject,
                                                   actual_leagues_per_loop,
                                                   actual_teams_per_league);

        typedef Kokkos::TeamPolicy< ExecSpace > policy_type;

        int size = Parallel::getKokkosChunkSize();
        if(size > 0)
          teamPolicy.set_chunk_size(size);

        Kokkos::parallel_reduce(name, teamPolicy,
                                KOKKOS_LAMBDA(typename policy_type::member_type thread, ReductionType& inner_sum) {

          // We are within an SM, and all SMs share the same amount of
          // assigned Kokkos threads.  Figure out which range of N items
          // this SM should work on (as a multiple of 32).
          const unsigned int currentPartition = p * actual_leagues_per_loop + thread.league_rank();
          unsigned int estimatedThreadAmount =
            numItems * (currentPartition) / (actual_leagues_per_loop * nPartitions);
          const unsigned int startingN =  estimatedThreadAmount +
            ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
          unsigned int endingN;

          // Check if this is the last partition
          if (currentPartition + 1 == actual_leagues_per_loop * nPartitions) {
            endingN = numItems;
          } else {
            estimatedThreadAmount = numItems * (currentPartition + 1) / (actual_leagues_per_loop * nPartitions);
            endingN = estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
          }

          const unsigned int totalN = endingN - startingN;
          // printf("league_rank: %d, team_size: %d, team_rank: %d, startingN: %d, endingN: %d, totalN: %d\n", thread.league_rank(), thread.team_size(), thread.team_rank(), startingN, endingN, totalN);

          Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, totalN),
                               [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2] (const int& N) {
            // Craft an i,j,k out of this range.  This approach works with
            // row-major layout so that consecutive Kokkos threads work
            // along consecutive slots in memory.

            // printf("parallel_reduce team demo - n is %d, league_rank is %d, true n is %d\n", N, thread.league_rank(), (startingN + N));

            const int i = ((startingN + N) % i_size)           + rbegin0;
            const int j = ((startingN + N) / i_size) % j_size  + rbegin1;
            const int k =  (startingN + N) / (j_size * i_size) + rbegin2;

            ReductionType tmp = 0;
            functor(i, j, k, tmp);

            inner_sum += tmp;
          });
        }, tmp0);

        red += tmp0;
      }
    }
  }
  else
  {
    ExecSpace instanceObject = getInstance(execObj);

    // Range Policy
    if(kokkos_policy == Parallel::Kokkos_Range_Policy)
    {
      Kokkos::RangePolicy<ExecSpace>
        rangePolicy(instanceObject, 0, numItems);

      int size = Parallel::getKokkosChunkSize();
      if(size > 0)
        rangePolicy.set_chunk_size(size);

      Kokkos::parallel_reduce(name, rangePolicy,
                              KOKKOS_LAMBDA(int n, ReductionType& tmp) {
          const int k = n / (j_size * i_size) + rbegin2;
          const int j = (n / i_size) % j_size + rbegin1;
          const int i = n % i_size + rbegin0;

          functor(i, j, k, tmp);

        }, red);
    }
    // MDRange Policy
    else if(kokkos_policy == Parallel::Kokkos_MDRange_Policy)
    {
      int i_tile, j_tile, k_tile;
      Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

      if(i_tile > 0 || j_tile > 0 || k_tile > 0)
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin0, rbegin1, rbegin2},
                        {rend0,   rend1,   rend2},
                        {i_tile,  j_tile,  k_tile});

        Kokkos::parallel_reduce(name, mdRangePolicy, functor, red);
      }
      else
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin0, rbegin1, rbegin2},
                        {rend0,   rend1,   rend2},
                        {i_size,  j_size,  k_size});

        Kokkos::parallel_reduce(name, mdRangePolicy, functor, red);
      }
    }
    // MDRange Policy - Reverse index
    else if(kokkos_policy == Parallel::Kokkos_MDRange_Reverse_Policy)
    {
      int i_tile, j_tile, k_tile;
      Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

      if(i_tile > 0 || j_tile > 0 || k_tile > 0)
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin2, rbegin1, rbegin0},
                        {rend2,   rend1,   rend0},
                        {k_tile,  j_tile,  i_tile});

        Kokkos::parallel_reduce(name, mdRangePolicy,
                                KOKKOS_LAMBDA(int k, int j, int i,
                                              ReductionType& tmp) {
                                  functor(i, j, k, tmp);
                                }, red);
      }
      else
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin2, rbegin1, rbegin0},
                        {rend2,   rend1,   rend0},
                        {k_size,  j_size,  i_size});

        Kokkos::parallel_reduce(name, mdRangePolicy,
                                KOKKOS_LAMBDA(int k, int j, int i,
                                              ReductionType& tmp) {
                                  functor(i, j, k, tmp);
                                }, red);
      }
    }
  }

  std::free( name );
}
#endif  // #if defined(HAVE_KOKKOS)

//----------------------------------------------------------------------------
// Parallel_reduce_min loops
//
// CPU
// No ExecSpace/MemSpace
// OpenMP       - Range (default), MDRange and Team Policy
// GPU          - Team (default), Range and MDRange Policy
//----------------------------------------------------------------------------

// CPU parallel_reduce_min
// TODO: This appears to not do any "min" on the reduction.
template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_reduce_min(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    BlockRange const & r, const Functor & functor, ReductionType & red)
{
  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  ReductionType tmp = red;

  for (int k=rbegin2; k<rend2; ++k) {
    for( int j=rbegin1; j<rend1; ++j) {
      for (int i=rbegin0; i<rend0; ++i) {
        functor(i, j, k, tmp);

        if(red > tmp)
          red = tmp;
      }
    }
  }
}

// CPU parallel_reduce_min - For legacy loops where no execution space
// was specified as a template parameter.
template < typename Functor, typename ReductionType>
void
parallel_reduce_min(BlockRange const & r, const Functor & functor, ReductionType & red)
{
  // Force users into using a single CPU thread if they didn't specify
  // OpenMP.

  // Make an empty object
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj;

  parallel_reduce_min<UintahSpaces::CPU>(execObj, r, functor, red);
}

// OpenMP parallel_reduce_min
#if defined(KOKKOS_ENABLE_OPENMP__COMMENTED_OUT)

template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_reduce_min(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    BlockRange const & r, const Functor & functor, ReductionType & red)
{
  int status;
  char *name(abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status));

  const int i_size = r.end(0) - r.begin(0);
  const int j_size = r.end(1) - r.begin(1);
  const int k_size = r.end(2) - r.begin(2);

  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  const unsigned int numItems = ((i_size > 0 ? i_size : 1) *
                                 (j_size > 0 ? j_size : 1) *
                                 (k_size > 0 ? k_size : 1));

  Parallel::Kokkos_Policy kokkos_policy = Parallel::getKokkosPolicy();

  // Range Policy
  if(kokkos_policy == Parallel::Kokkos_Range_Policy)
  {
    Kokkos::RangePolicy<ExecSpace, int> rangePolicy(0, numItems);

    int size = Parallel::getKokkosChunkSize();
    if(size > 0)
      rangePolicy.set_chunk_size(size);

    Kokkos::parallel_reduce(name, rangePolicy,
                            KOKKOS_LAMBDA(int n, ReductionType & tmp) {

                              const int k = n / (j_size * i_size) + rbegin2;
                              const int j = (n / i_size) % j_size + rbegin1;
                              const int i = n % i_size + rbegin0;

                              functor(i, j, k, tmp);

                            }, Kokkos::Min<ReductionType>(red));
  }

  // MDRange Policy
  else if(kokkos_policy == Parallel::Kokkos_MDRange_Policy)
  {
    int i_tile, j_tile, k_tile;
    Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

    if(i_tile > 0 || j_tile > 0 || k_tile > 0)
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin0, rbegin1, rbegin2},
                      {rend0,   rend1,   rend2},
                      {i_tile,  j_tile,  k_tile});

      Kokkos::parallel_reduce(name, mdRangePolicy, functor,
                              Kokkos::Min<ReductionType>(red));
    }
    else
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin0, rbegin1, rbegin2},
                      {rend0,   rend1,   rend2},
                      {i_size,  j_size,  k_size});

      Kokkos::parallel_reduce(name, mdRangePolicy, functor,
                              Kokkos::Min<ReductionType>(red));
    }
  }
  // MDRange Policy - Reverse index
  else if(kokkos_policy == Parallel::Kokkos_MDRange_Reverse_Policy)
  {
    int i_tile, j_tile, k_tile;
    Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

    if(i_tile > 0 || j_tile > 0 || k_tile > 0)
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin2, rbegin1, rbegin0},
                      {rend2,   rend1,   rend0},
                      {k_tile,  j_tile,  i_tile});

      Kokkos::parallel_reduce(name, mdRangePolicy,
                              KOKKOS_LAMBDA(int k, int j, int i,
                                            ReductionType& tmp) {
                                functor(i, j, k, tmp);
                              }, Kokkos::Min<ReductionType>(red));
    }
    else
    {
      Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
        mdRangePolicy({rbegin2, rbegin1, rbegin0},
                      {rend2,   rend1,   rend0},
                      {k_size,  j_size,  i_size});

      Kokkos::parallel_reduce(name, mdRangePolicy,
                              KOKKOS_LAMBDA(int k, int j, int i,
                                            ReductionType& tmp) {
                                functor(i, j, k, tmp);
                              }, Kokkos::Min<ReductionType>(red));
    }
  }
  // Team Policy
  else if(kokkos_policy == Parallel::Kokkos_Team_Policy)
  {
    // Assumption - there is only one league for OpenMP
    const int teams_per_league = Parallel::getKokkosTeamsPerLeague();
    const int team_range_size = numItems;

    const int actualTeams = teams_per_league < team_range_size ?
                            teams_per_league : team_range_size;

    Kokkos::TeamPolicy<ExecSpace> teamPolicy(1, actualTeams);
    typedef Kokkos::TeamPolicy<ExecSpace> policy_type;

    int size = Parallel::getKokkosChunkSize();
    if(size > 0)
      teamPolicy.set_chunk_size(size);

    Kokkos::parallel_reduce(name, teamPolicy,
                            KOKKOS_LAMBDA(typename policy_type::member_type thread, ReductionType& inner_min) {
      // printf("i is %d\n", thread.team_rank());
      Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_range_size),
                           [&](const int& n) {
        const int i = n / (j_size * k_size) + rbegin0;
        const int j = (n / k_size) % j_size + rbegin1;
        const int k = n % k_size + rbegin2;

        ReductionType tmp = 0;
        functor(i, j, k, tmp);

        if(inner_min > tmp)
          inner_min = tmp;
      });
    }, Kokkos::Min<ReductionType>(red));
  }

  std::free( name );
}
#endif  // #if defined(KOKKOS_ENABLE_OPENMP

// GPU parallel_reduce_min
#if defined(HAVE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>

inline typename std::enable_if<
#if defined(KOKKOS_USING_GPU)
  std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value
#endif
#if defined(KOKKOS_USING_GPU) && defined(KOKKOS_ENABLE_OPENMP)
  ||
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  std::is_same<ExecSpace, Kokkos::OpenMP>::value
#endif
  , void>::type

parallel_reduce_min(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    BlockRange const & r, const Functor & functor, ReductionType & red)
{
  int status;
  char *name = abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status);

  const int i_size = r.end(0) - r.begin(0);
  const int j_size = r.end(1) - r.begin(1);
  const int k_size = r.end(2) - r.begin(2);

  const int rbegin0 = r.begin(0);
  const int rbegin1 = r.begin(1);
  const int rbegin2 = r.begin(2);

  const int rend0 = r.end(0);
  const int rend1 = r.end(1);
  const int rend2 = r.end(2);

  const unsigned int numItems = ((i_size > 0 ? i_size : 1) *
                                 (j_size > 0 ? j_size : 1) *
                                 (k_size > 0 ? k_size : 1));

  Parallel::Kokkos_Policy kokkos_policy = Parallel::getKokkosPolicy();

  // Team Policy
  if(kokkos_policy == Parallel::Kokkos_Team_Policy)
  {
    // Team policy approach.

    // Overall goal, split a 3D range requested by the user into various
    // SMs on the GPU.  (In essence, this would be a Kokkos
    // MD_Team+Policy, if one existed) The process requires going from
    // 3D range to a 1D range, partitioning the 1D range into groups
    // that are multiples of 32, then converting that group of 32 range
    // back into a 3D (i,j,k) index.

    // The user has two partitions available.  1) One is the total
    // number of streaming multiprocessors.  2) The other is splitting a
    // task into multiple streams and execution units.

    // Get the requested amount of threads per streaming multiprocessor
    // (SM) and number of SMs totals.
    if (std::is_same<ExecSpace, Kokkos::OpenMP>::value)
    {
      // Assumption - there is only one league for OpenMP
      const int teams_per_league = Parallel::getKokkosTeamsPerLeague();
      const int team_range_size = numItems;

      const int actualTeams = teams_per_league < team_range_size ?
                              teams_per_league : team_range_size;

      Kokkos::TeamPolicy<ExecSpace> teamPolicy(1, actualTeams);
      typedef Kokkos::TeamPolicy<ExecSpace> policy_type;

      int size = Parallel::getKokkosChunkSize();
      if(size > 0)
        teamPolicy.set_chunk_size(size);

      Kokkos::parallel_reduce(name, teamPolicy,
                              KOKKOS_LAMBDA(typename policy_type::member_type thread, ReductionType& inner_min) {
        // printf("i is %d\n", thread.team_rank());
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_range_size),
                             [&](const int& n) {
          const int i = n / (j_size * k_size) + rbegin0;
          const int j = (n / k_size) % j_size + rbegin1;
          const int k = n % k_size + rbegin2;

          ReductionType tmp = 0;
          functor(i, j, k, tmp);

          if(inner_min > tmp)
            inner_min = tmp;
        });
      }, Kokkos::Min<ReductionType>(red));
    }
    else
    {
      const int kokkos_leagues_per_loop = Parallel::getKokkosLeaguesPerLoop();
      const int kokkos_teams_per_league = Parallel::getKokkosTeamsPerLeague();

      const int nPartitions = execObj.getNumInstances();

      // The requested range of data may not have enough work for the
      // requested command line arguments, so shrink them if necessary.
      int teams_per_loop = kokkos_teams_per_league * kokkos_leagues_per_loop;
      int team_range_size = numItems / nPartitions;

      const unsigned int actual_teams =
        teams_per_loop < team_range_size ?
        teams_per_loop : team_range_size;

      const unsigned int actual_teams_per_league =
        kokkos_teams_per_league < team_range_size ?
        kokkos_teams_per_league : team_range_size;

      const unsigned int actual_leagues_per_loop =
        (actual_teams - 1) / kokkos_teams_per_league + 1;

      for (int p = 0; p < nPartitions; p++) {
        ReductionType tmp0 = 0;

        ExecSpace instanceObject = getInstance(execObj, p);

        // Use a Team Policy, this allows us to control how many threads
        // per league and how many leagues are used.
        Kokkos::TeamPolicy< ExecSpace > teamPolicy(instanceObject,
                                                   actual_leagues_per_loop,
                                                   actual_teams_per_league);

        typedef Kokkos::TeamPolicy< ExecSpace > policy_type;

        int size = Parallel::getKokkosChunkSize();
        if(size > 0)
          teamPolicy.set_chunk_size(size);

        Kokkos::parallel_reduce(name, teamPolicy,
                                KOKKOS_LAMBDA(typename policy_type::member_type thread, ReductionType& inner_min) {

          // We are within an SM, and all SMs share the same amount of
          // assigned Kokkos threads.  Figure out which range of N items
          // this SM should work on (as a multiple of 32).
          const unsigned int currentPartition = p * actual_leagues_per_loop + thread.league_rank();
          unsigned int estimatedThreadAmount =
            numItems * (currentPartition) / (actual_leagues_per_loop * nPartitions);
          const unsigned int startingN =  estimatedThreadAmount +
            ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
          unsigned int endingN;

          // Check if this is the last partition
          if (currentPartition + 1 == actual_leagues_per_loop * nPartitions) {
            endingN = numItems;
          } else {
            estimatedThreadAmount = numItems * (currentPartition + 1) / (actual_leagues_per_loop * nPartitions);
            endingN = estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
          }

          const unsigned int totalN = endingN - startingN;
          // printf("league_rank: %d, team_size: %d, team_rank: %d, startingN: %d, endingN: %d, totalN: %d\n", thread.league_rank(), thread.team_size(), thread.team_rank(), startingN, endingN, totalN);

          Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, totalN),
                               [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2] (const int& N) {
            // Craft an i,j,k out of this range.  This approach works with
            // row-major layout so that consecutive Kokkos threads work
            // along consecutive slots in memory.

            // printf("parallel_reduce team demo - n is %d, league_rank is %d, true n is %d\n", N, thread.league_rank(), (startingN + N));

            const int i = ((startingN + N) % i_size)           + rbegin0;
            const int j = ((startingN + N) / i_size) % j_size  + rbegin1;
            const int k =  (startingN + N) / (j_size * i_size) + rbegin2;

            ReductionType tmp;
            functor(i, j, k, tmp);

            if(inner_min > tmp)
              inner_min = tmp;
          });
        }, Kokkos::Min<ReductionType>(tmp0));

        if(red > tmp0)
          red = tmp0;
      }
    }
  }
  else
  {
    ReductionType tmp = red;

    ExecSpace instanceObject = getInstance(execObj);

    // Range Policy
    if(kokkos_policy == Parallel::Kokkos_Range_Policy)
    {
      Kokkos::RangePolicy<ExecSpace>
        rangePolicy(instanceObject, 0, numItems);

      int size = Parallel::getKokkosChunkSize();
      if(size > 0)
        rangePolicy.set_chunk_size(size);

      Kokkos::parallel_reduce(name, rangePolicy,
                              KOKKOS_LAMBDA(int n, ReductionType& tmp) {
          const int k = n / (j_size * i_size) + rbegin2;
          const int j = (n / i_size) % j_size + rbegin1;
          const int i = n % i_size + rbegin0;

          functor(i, j, k, tmp);

        }, Kokkos::Min<ReductionType>(red));
    }
    // MDRange Policy
    else if(kokkos_policy == Parallel::Kokkos_MDRange_Policy)
    {
      int i_tile, j_tile, k_tile;
      Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

      if(i_tile > 0 || j_tile > 0 || k_tile > 0)
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin0, rbegin1, rbegin2},
                        {rend0,   rend1,   rend2},
                        {i_tile,  j_tile,  k_tile});

        Kokkos::parallel_reduce(name, mdRangePolicy, functor,
                                Kokkos::Min<ReductionType>(red));
      }
      else
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin0, rbegin1, rbegin2},
                        {rend0,   rend1,   rend2},
                        {i_size,  j_size,  k_size});

        Kokkos::parallel_reduce(name, mdRangePolicy, functor,
                                Kokkos::Min<ReductionType>(red));
      }
    }
    // MDRange Policy - Reverse index
    else if(kokkos_policy == Parallel::Kokkos_MDRange_Reverse_Policy)
    {
      int i_tile, j_tile, k_tile;
      Parallel::getKokkosTileSize(i_tile, j_tile, k_tile);

      if(i_tile > 0 || j_tile > 0 || k_tile > 0)
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin2, rbegin1, rbegin0},
                        {rend2,   rend1,   rend0},
                        {k_tile,  j_tile,  i_tile});

        Kokkos::parallel_reduce(name, mdRangePolicy,
                                KOKKOS_LAMBDA(int k, int j, int i,
                                              ReductionType& tmp) {
                                  functor(i, j, k, tmp);
                                }, Kokkos::Min<ReductionType>(red));
      }
      else
      {
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
          mdRangePolicy({rbegin2, rbegin1, rbegin0},
                        {rend2,   rend1,   rend0},
                        {k_size,  j_size,  i_size});

        Kokkos::parallel_reduce(name, mdRangePolicy,
                                KOKKOS_LAMBDA(int k, int j, int i,
                                              ReductionType& tmp) {
                                  functor(i, j, k, tmp);
                                }, Kokkos::Min<ReductionType>(red));
      }
    }
  }

  std::free( name );
}

#endif  // #if defined(HAVE_KOKKOS)

//----------------------------------------------------------------------------
// Sweeping_parallel_for loops

// CPU
// OpenMP
// GPU - not implemented
//----------------------------------------------------------------------------

// CPU sweeping_parallel_for
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU >::value, void>::type
sweeping_parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, const bool plusX, const bool plusY, const bool plusZ , const int npart)
{
  const int idir = plusX ? 1 : -1;
  const int jdir = plusY ? 1 : -1;
  const int kdir = plusZ ? 1 : -1;

  const int start_x = plusX ? r.begin(0) : r.end(0)-1;
  const int start_y = plusY ? r.begin(1) : r.end(1)-1;
  const int start_z = plusZ ? r.begin(2) : r.end(2)-1;

  const int end_x = plusX ? r.end(0) : -r.begin(0)+1;
  const int end_y = plusY ? r.end(1) : -r.begin(1)+1;
  const int end_z = plusZ ? r.end(2) : -r.begin(2)+1;

  for (int k=start_z; k*kdir<end_z; k=k+kdir) {
    for (int j=start_y; j*jdir<end_y; j=j+jdir) {
      for (int i=start_x ; i*idir<end_x; i=i+idir) {
        functor(i, j, k);
      }
    }
  }
}

// OpenMP sweeping_parallel_for
#if defined(KOKKOS_ENABLE_OPENMP)
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
sweeping_parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, const bool plusX, const bool plusY, const bool plusZ , const int npart)
{
  int status;
  char *name(abi::__cxa_demangle(typeid(Functor).name(), 0, 0, &status));

  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  /////////// CUBIC BLOCKS SUPPORTED ONLY /////////////////
  // RECTANGLES ARE HARD BUT POSSIBLY MORE EFFICIENT //////
  /////////////////////////////////////////////////////////
  const int nPartitionsx = npart; // try to break domain into nxnxn block
  const int nPartitionsy = npart;
  const int nPartitionsz = npart;
  const int dx = ie - ib;
  const int dy = je - jb;
  const int dz = ke - kb;
  const int sdx = dx / nPartitionsx;
  const int sdy = dy / nPartitionsy;
  const int sdz = dz / nPartitionsz;
  const int rdx = dx - sdx * nPartitionsx;
  const int rdy = dy - sdy * nPartitionsy;
  const int rdz = dz - sdz * nPartitionsz;

  const int nphase = nPartitionsx + nPartitionsy + nPartitionsz - 2;
  int tpp=0; //  Total parallel processes/blocks

  int concurrentBlocksArray[nphase/2+1]; // +1 needed for odd values,
                                         // use symmetry

  for (int iphase=0; iphase<nphase; iphase++) {
    if ((nphase-iphase-1) >= iphase) {
      tpp =(iphase+2)*(iphase+1)/2;

      tpp -= std::max(iphase-nPartitionsx+1,0)*(iphase-nPartitionsx+2)/2;
      tpp -= std::max(iphase-nPartitionsy+1,0)*(iphase-nPartitionsy+2)/2;
      tpp -= std::max(iphase-nPartitionsz+1,0)*(iphase-nPartitionsz+2)/2;

      concurrentBlocksArray[iphase]=tpp;
    } else {
      tpp=concurrentBlocksArray[nphase-iphase-1];
    }

    Kokkos::View<int*, Kokkos::HostSpace> xblock("xblock", tpp);
    Kokkos::View<int*, Kokkos::HostSpace> yblock("yblock", tpp);
    Kokkos::View<int*, Kokkos::HostSpace> zblock("zblock", tpp);

    int icount = 0 ;
    // Attempts to iterate over k j i , despite  spatial dependencies.
    for (int k=0; k< std::min(iphase+1,nPartitionsz); k++) {
      for (int j=0; j< std::min(iphase-k+1,nPartitionsy); j++) {
        if ((iphase -k-j) <nPartitionsx){
          xblock(icount) = iphase-k-j;
          yblock(icount) = j;
          zblock(icount) = k;
          icount++;
        }
      }
    }

    // Multidirectional parameters
    const int idir = plusX ? 1 : -1;
    const int jdir = plusY ? 1 : -1;
    const int kdir = plusZ ? 1 : -1;

    Kokkos::RangePolicy<ExecSpace, int> rangePolicy(0, tpp);

    int size = Parallel::getKokkosChunkSize();
    if(size > 0)
      rangePolicy.set_chunk_size(size);

    Kokkos::parallel_for(name, rangePolicy,
                         KOKKOS_LAMBDA(int iblock) {
      const int  xiBlock = plusX ? xblock(iblock) : nPartitionsx-xblock(iblock)-1;
      const int  yiBlock = plusY ? yblock(iblock) : nPartitionsx-yblock(iblock)-1;
      const int  ziBlock = plusZ ? zblock(iblock) : nPartitionsx-zblock(iblock)-1;

      const int blockx_start=ib+xiBlock *sdx;
      const int blocky_start=jb+yiBlock *sdy;
      const int blockz_start=kb+ziBlock *sdz;

      const int blockx_end= ib+ (xiBlock+1) * sdx + (xiBlock+1 == nPartitionsx ? rdx:0);
      const int blocky_end= jb+ (yiBlock+1) * sdy + (yiBlock+1 == nPartitionsy ? rdy:0);
      const int blockz_end= kb+ (ziBlock+1) * sdz + (ziBlock+1 == nPartitionsz ? rdz:0);

      const int blockx_end_dir= plusX ? blockx_end :-blockx_start+1 ;
      const int blocky_end_dir= plusY ? blocky_end :-blocky_start+1 ;
      const int blockz_end_dir= plusZ ? blockz_end :-blockz_start+1 ;

      for (int k=plusZ ? blockz_start : blockz_end-1; k*kdir<blockz_end_dir; k=k+kdir) {
        for (int j=plusY ? blocky_start : blocky_end-1; j*jdir<blocky_end_dir; j=j+jdir) {
          for (int i=plusX ? blockx_start : blockx_end-1; i*idir<blockx_end_dir; i=i+idir) {
            functor(i, j, k);
          }
        }
      }
    });
  } // end for (int iphase = 0; iphase < nphase; iphase++)

  std::free( name );
}
#endif  // #if defined(KOKKOS_ENABLE_OPENMP


// GPU sweeping_parallel_for
#if defined(KOKKOS_USING_GPU)

template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
sweeping_parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, const bool plusX, const bool plusY, const bool plusZ, const int npart)
{
    SCI_THROW(InternalError("Error: sweeps on GPU has not been implimented .", __FILE__, __LINE__));
}

#endif  // #if defined(KOKKOS_USING_GPU)

//----------------------------------------------------------------------------
// Parallel_for_unstructured loops
//
// CPU - Kokkos View version
// CPU - std::vector version
// OpenMP
// GPU
//----------------------------------------------------------------------------

// Allows the user to specify a vector (or view) of indices that
// require an operation, often needed for boundary conditions and
// possibly structured grids.

#if defined(HAVE_KOKKOS)

// CPU parallel_for_unstructured - Kokkos::View
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
                          Kokkos::View<IntVector*, Kokkos::HostSpace> iterSpace,
                          const unsigned int list_size , const Functor & functor)
{
  for (unsigned int iblock=0; iblock<list_size; ++iblock) {
    functor(iterSpace[iblock][0], iterSpace[iblock][1], iterSpace[iblock][2]);
  };
}

#else

// CPU parallel_for_unstructured - std::vector
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
                          std::vector<IntVector> &iterSpace,
                          const unsigned int list_size, const Functor & functor)
{
  for (unsigned int iblock=0; iblock<list_size; ++iblock) {
    functor(iterSpace[iblock][0], iterSpace[iblock][1], iterSpace[iblock][2]);
  };
}

#endif  // #if defined(HAVE_KOKKOS)

// OpenMP - parallel_for_unstructured
#if defined(KOKKOS_ENABLE_OPENMP)
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
                          Kokkos::View<IntVector*, Kokkos::HostSpace> iterSpace,
                          const unsigned int list_size, const Functor & functor)
{
  // int status;
  // char *name(abi::__cxa_demangle(typeid(ExecSpace).name(), 0, 0, &status));

  Kokkos::RangePolicy<ExecSpace, int> rangePolicy(0, list_size);

  int size = Parallel::getKokkosChunkSize();
  if(size > 0)
    rangePolicy.set_chunk_size(size);

  Kokkos::parallel_for("OpenMP Unstructured", rangePolicy,
                       [=](const unsigned int & iblock) {
    functor(iterSpace[iblock][0], iterSpace[iblock][1], iterSpace[iblock][2]);
  });

  // std::free( name );
}

#endif  // #if defined(KOKKOS_ENABLE_OPENMP)

// GPU parallel_for_unstructured
#if defined(KOKKOS_USING_GPU)

template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
                          Kokkos::View<IntVector*, MemSpace> iterSpace,
                          const unsigned int list_size, const Functor & functor)
{
  ExecSpace instanceObject = getInstance(execObj);

  Kokkos::RangePolicy< ExecSpace > rangePolicy(instanceObject, 0, list_size);

  int size = Parallel::getKokkosChunkSize();
  if(size > 0)
    rangePolicy.set_chunk_size(size);

  Kokkos::parallel_for("GPU Unstructured", rangePolicy,
                       KOKKOS_LAMBDA(const unsigned int & iblock) {

      // const IntVector & is = iterSpace[iblock];
      // functor(is.m_value[0], is.m_value[1], is.m_value[2]);

      functor(iterSpace[iblock][0], iterSpace[iblock][1], iterSpace[iblock][2]);
    });

  /*
  const int leagues_per_loop = Parallel::getKokkosLeaguesPerLoop();
  const int teams_per_league = Parallel::getKokkosTeamsPerLeague();

  const int actualTeams = teams_per_league < list_size ?
                          teams_per_league : list_size;

  ExecSpace instanceObject = getInstance(execObj);

  Kokkos::TeamPolicy< ExecSpace > teamPolicy(instanceObject,
                                             leagues_per_loop,
                                             actualTeams);
  typedef Kokkos::TeamPolicy< ExecSpace > policy_type;

  Kokkos::parallel_for("GPU Unstructured", teamPolicy,
                       KOKKOS_LAMBDA(typename policy_type::member_type thread) {

    const unsigned int currentBlock = thread.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, list_size),
                        [&,iterSpace] (const unsigned int& iblock) {
    functor(iterSpace[iblock][0], iterSpace[iblock][1], iterSpace[iblock][2]);
      });
    });
  */

  // std::free( name );
}

#endif  // #if defined(KOKKOS_USING_GPU)

//----------------------------------------------------------------------------
// Parallel_for_initialize
//
// GPU
//----------------------------------------------------------------------------

// GPU parallel_for_initialize
#if defined(KOKKOS_USING_GPU)

template <typename ExecSpace, typename MemSpace, typename T2, typename T3>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
parallel_for_initialize(ExecutionObject<ExecSpace, MemSpace>& execObj,
                        T2 KV3, const T3 init_val)
{
  int leagues_per_loop = Parallel::getKokkosLeaguesPerLoop();
  int teams_per_league = Parallel::getKokkosTeamsPerLeague();

  const int num_cells = KV3.m_view.size();
  const int actualTeams = teams_per_league < num_cells ?
                          teams_per_league : num_cells;

  typedef Kokkos::TeamPolicy<ExecSpace> policy_type;
  Kokkos::TeamPolicy<ExecSpace> teamPolicy(leagues_per_loop, actualTeams);

  Kokkos::parallel_for("GPU Initialized", teamPolicy,
                       KOKKOS_LAMBDA(typename policy_type::member_type thread) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, num_cells),
                           [=](const int& i) {
          KV3(i) = init_val;
      });
    });
}
#endif  // #if defined(KOKKOS_USING_GPU)

// Initialization API that takes both KokkosView3 arguments and
// View<KokkosView3> arguments.  If compiling w/o kokkos it takes CC
// NC SFCX SFCY SFCZ arguments and std::vector<T> arguments


//----------------------------------------------------------------------------
// Parallel_initialize_grouped
//
// OpenMP
// GPU
//----------------------------------------------------------------------------

// OpenMP - parallel_for_grouped
#if defined(KOKKOS_ENABLE_OPENMP)

template <typename ExecSpace, typename MemSpace, typename T, typename ValueType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_initialize_grouped(ExecutionObject<ExecSpace, MemSpace>& execObj,
                            const struct1DArray<T, ARRAY_SIZE>& KKV3, const ValueType& init_val) {

  // int status;
  // char *name(abi::__cxa_demangle(typeid(ExecSpace).name(), 0, 0, &status));

  // TODO: This should probably be serialized and not use a
  // Kokkos::parallel_for?

  unsigned int n_cells = 0;
  for (unsigned int j = 0; j < KKV3.runTime_size; j++){
    n_cells += KKV3[j].m_view.size();
  }

  Kokkos::RangePolicy<ExecSpace, int> rangePolicy(0, n_cells);

  int size = Parallel::getKokkosChunkSize();
  if(size > 0)
    rangePolicy.set_chunk_size(size);

  Kokkos::parallel_for("OpenMP Initialize Grouped", rangePolicy,
                       KOKKOS_LAMBDA(unsigned int i_tot) {
      // TODO: Find a more efficient way of doing this.
      int i = i_tot;
      int j = 0;
      while (i-(int) KKV3[j].m_view.size() > -1) {
        i -= KKV3[j].m_view.size();
        j++;
      }

      KKV3[j](i) = init_val;
    });

  // std::free( name );
}

// template <typename ExecSpace, typename MemSpace, typename T2, typename T3>
// inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
// parallel_initialize_single(ExecutionObject& execObj, T2 KKV3, const T3 init_val){
//  for (unsigned int j=0; j < KKV3.size(); j++){
//    Kokkos::RangePolicy<ExecSpace, int> rangePolicy(0, KKV3(j).m_view.size());
  // int size = Parallel::getKokkosChunkSize();
  // if(size > 0)
  //   rangePolicy.set_chunk_size(size);

//    Kokkos::parallel_for("OpenMP Initialize Grouped", rangePolicy,
//                         KOKKOS_LAMBDA(int i) {
//        KKV3(j)(i) = init_val;
//      });
//  }
// }

#endif  // #if defined(KOKKOS_ENABLE_OPENMP

// GPU parallel_for_grouped
#if defined(KOKKOS_USING_GPU)

/* DS 11052019: Wrote alternative (and simpler) version of
 * parallel_initialize_grouped for Kokkos The previous version seems
 * to produce error in parallel_initialize.  This version finds the
 * max numb of cells among variables and uses it as an iteration
 * count. Using simpler RangePolicy instead of TeamPolicy. All
 * computations to find out index in TeamPolicy - especially divides
 * and mods do not seem worth for simple init code. Secondly iterating
 * over all variables within struct1DArray manually rather than
 * spawning extra threads. This reduces degreee of parallelism, but
 * produces correct result. Revisit later if it becomes a bottleneck.
 */
template <typename ExecSpace, typename MemSpace, typename T, typename ValueType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
parallel_initialize_grouped(ExecutionObject<ExecSpace, MemSpace>& execObj,
    const struct1DArray<T, ARRAY_SIZE>& KKV3, const ValueType& init_val) {

  // int status;
  // char *name(abi::__cxa_demangle(typeid(ExecSpace).name(), 0, 0, &status));

  // n_cells is the max of cells total to process among
  // collection of vars (the view of Kokkos views).  For
  // example, if this were being used to if one var had 4096
  // cells and another var had 5832 cells, n_cells would become
  // 5832
  unsigned int n_cells = 0;
  for (size_t j = 0; j < KKV3.runTime_size; j++){
    n_cells = KKV3[j].m_view.size() > n_cells ? KKV3[j].m_view.size() : n_cells;
  }

  ExecSpace instanceObject = getInstance(execObj);

  Kokkos::RangePolicy< ExecSpace > rangePolicy(instanceObject, 0, n_cells);

  Kokkos::parallel_for("GPU Initialize Grouped", rangePolicy,
                       KOKKOS_LAMBDA(size_t i) {
      for(size_t j=0; j<KKV3.runTime_size; j++){
        if(i<KKV3[j].m_view.size())
          KKV3[j](i) = init_val;
      }
    });

  // std::free( name );
}

/*
template <typename ExecSpace, typename MemSpace, typename T, typename ValueType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
parallel_initialize_grouped(ExecutionObject<ExecSpace, MemSpace>& execObj,
                            const struct1DArray<T, ARRAY_SIZE>& KKV3, const ValueType& init_val) {

  // n_cells is the amount of cells total to process among collection of vars (the view of Kokkos views)
  // For example, if this were being used to  if one var had 4096 cells and another var had 5832 cells, n_cells would become 4096+5832=

  unsigned int n_cells = 0;
  for (unsigned int j = 0; j < KKV3.runTime_size; j++){
    n_cells += KKV3[j].m_view.size();
  }

  const int leagues_per_loop = Parallel::getKokkosLeaguesPerLoop();
  const int teams_per_league = Parallel::getKokkosTeamsPerLeague();

  const int actualTeams = teams_per_league < n_cells ?
                          teams_per_league : n_cells;

  ExecSpace instanceObject = getInstance(execObj);

  Kokkos::TeamPolicy< ExecSpace > teamPolicy(instanceObject,
                                             leagues_per_loop,
                                             actualTeams);

  typedef Kokkos::TeamPolicy< ExecSpace > policy_type;

  Kokkos::parallel_for("GPU Initialize Grouped", teamPolicy,
                       KOKKOS_LAMBDA(typename policy_type::member_type thread) {

      // i_tot will come in as a number between 0 and actualTeams.  Suppose actualTeams is 256.
      // Thread 0 should work on cell 0, thread 1 should work on cell 1, ... thread 255 should work on cell 255
      // then they all advanced forward by actualTeams.
      // Thread 0 works on cell 256, thread 1 works on cell 257... thread 511 works on cell 511.
      // This should continue until all cells are completed.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, actualTeams),
                         [&, n_cells, actualTeams, KKV3] (const unsigned int& i_tot) {
      const unsigned int n_iter = n_cells / actualTeams + (n_cells % actualTeams > 0 ? 1 : 0); // round up (more efficient to compute this outside parallel_for?)
      unsigned int  j = 0;
      unsigned int old_i = 0;
      for (unsigned int i = 0; i < n_iter; i++) {
         // use a while for small data sets or massive streaming multiprocessors
         while (i * actualTeams + i_tot - old_i >= KKV3[j].m_view.size()) {
           old_i += KKV3[j].m_view.size();
           j++;
           if(KKV3.runTime_size <= j) {
             return; // do nothing
           }
         }
         KKV3[j](i * actualTeams + i_tot - old_i) = init_val;
      }
    });
  });
}
*/

#endif  // #if defined(KOKKOS_USING_GPU)

//----------------------------------------------------------------------------
// Parallel_initialize loops
//
// legacy_initialize
// legacy_initialize
// CPU
// OpenMP
// GPU
//----------------------------------------------------------------------------

template <class TTT> // Needed for the casting inside of the Variadic
                     // template, also allows for nested templating
using Alias = TTT;

// CPU legacy_initialize
template< typename T, typename T2, const unsigned int T3>
inline void legacy_initialize(T inside_value, struct1DArray<T2,T3>& data_fields) {  // for vectors
  int nfields=data_fields.runTime_size;
  for(int i=0;  i< nfields; i++){
    data_fields[i].initialize(inside_value);
  }
  return;
}

// CPU legacy_initialize
template< typename T, typename T2>
inline void legacy_initialize(T inside_value, T2& data_fields) {  // for stand alone data fields
  data_fields.initialize(inside_value);
  return;
}

// CPU parallel_initialize
template <typename ExecSpace, typename MemSpace, typename T, class ...Ts>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_initialize(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    const T& initializationValue, Ts & ... inputs) {
  Alias<int[]>{( // first part of magic unpacker
                 // inputs.initialize (inside_value)
      legacy_initialize(initializationValue, inputs)
      ,0)...,0}; //second part of magic unpacker
}

#if defined(HAVE_KOKKOS)

// Forward Declaration of KokkosView3
template <typename T, typename MemSpace>
class KokkosView3;

// For an array of Views
template<typename T, typename MemSpace, unsigned int Capacity>
inline void setValueAndReturnView(struct1DArray<T, Capacity>* V, const T& x, int &index){
  V[index / ARRAY_SIZE][index % ARRAY_SIZE] = x;
  index++;
  return;
}

// For an array of Views
template<typename T, typename MemSpace, unsigned int Capacity1, unsigned int Capacity2>
inline void setValueAndReturnView(struct1DArray<T, Capacity1>* V, const struct1DArray<T, Capacity2>& small_v, int &index){
  int extra_i = small_v.runTime_size;
  for(int i = 0; i < extra_i; i++){
    V[(index+i) / ARRAY_SIZE][(index+i) % ARRAY_SIZE] = small_v[i];
  }
  index += extra_i;
  return;
}

template <typename T, typename MemSpace>
inline void sumViewSize(const T& x, int &index){
  index++;
  return;
}

//template <typename T, typename MemSpace>
//inline void sumViewSize(const Kokkos::View<T*, MemSpace>& small_v, int &index){
  //index += small_v.runTime_size;
  //return ;
//}

template <typename T, typename MemSpace, unsigned int Capacity>
inline void sumViewSize(const struct1DArray< T, Capacity> & small_v, int &index){
  index += small_v.runTime_size;
  return ;
}

#if defined(KOKKOS_ENABLE_OPENMP)
template <typename ExecSpace, typename MemSpace, typename T, class ...Ts>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_initialize(ExecutionObject<ExecSpace, MemSpace>& execObj, const T& initializationValue, Ts & ... inputs) {
  // Could this be modified to accept grid variables AND containers of
  // grid variables?

  // Count the number of views used (sometimes they may be views of views)
  int n = 0 ; // Get number of variadic arguments
  Alias<int[]>{( // First part of magic unpacker
      sumViewSize<KokkosView3< T, MemSpace>, MemSpace>(inputs, n)
      ,0)...,0}; // Second part of magic unpacker

  // Allocate space in host memory to track n total views.
  const int n_init_groups = ((n-1) / ARRAY_SIZE) + 1;
  struct1DArray< KokkosView3< T, MemSpace>, ARRAY_SIZE > hostArrayOfViews[n_init_groups];

  // Copy over the views one by one into this view of views.
  int i = 0; // Iterator counter
  Alias<int[]>{( // First part of magic unpacker
      setValueAndReturnView< KokkosView3< T, MemSpace>, MemSpace>(hostArrayOfViews, inputs, i)
      ,0)...,0}; // Second part of magic unpacker

  for (int j=0; j<n_init_groups; j++){
    hostArrayOfViews[j].runTime_size = n >ARRAY_SIZE * (j+1) ? ARRAY_SIZE : n % ARRAY_SIZE+1;
    parallel_initialize_grouped<ExecSpace, MemSpace, KokkosView3< T, MemSpace> >(execObj, hostArrayOfViews[j], initializationValue);
    //parallel_initialize_single<ExecSpace>(execObj, inputs_, inside_value); // safer version, less ambitious
  }
}
#endif

// GPU parallel_initialize
#if defined(KOKKOS_USING_GPU)
// This function is the same as above, but appears to be necessary due
// to CCVariable support.....
template <typename ExecSpace, typename MemSpace, typename T, class ...Ts>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::DefaultExecutionSpace>::value, void>::type
parallel_initialize(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    const T & initializationValue, Ts & ... inputs) {
  // Could this be modified to accept grid variables AND containers of
  // grid variables?

  // Count the number of views used (sometimes they may be views of views)
  int n = 0 ; // Get number of variadic arguments
  Alias<int[]>{( // First part of magic unpacker
      sumViewSize<KokkosView3< T, MemSpace>, Kokkos::HostSpace >(inputs, n)
      ,0)...,0}; // Second part of magic unpacker

  const int n_init_groups = ((n-1)/ARRAY_SIZE) + 1;
  struct1DArray< KokkosView3< T, MemSpace>, ARRAY_SIZE > hostArrayOfViews[n_init_groups];

  // Copy over the views one by one into this view of views.
  int i=0; // Iterator counter
  Alias<int[]>{( // First part of magic unpacker
      setValueAndReturnView< KokkosView3< T, MemSpace>, Kokkos::HostSpace >(hostArrayOfViews, inputs, i)
      ,0)...,0}; // Second part of magic unpacker

  for (int j=0; j<n_init_groups; j++){
    // DS 11052019: setting else part to n % ARRAY_SIZE instead of n %
    // ARRAY_SIZE+1. Why +1? It adds one extra variable, which does
    // not exist.  At least matches with the alternative
    // implementation
    //hostArrayOfViews[j].runTime_size = n >ARRAY_SIZE * (j+1) ? ARRAY_SIZE : n % ARRAY_SIZE+1;
    hostArrayOfViews[j].runTime_size = n >ARRAY_SIZE * (j+1) ? ARRAY_SIZE : n % ARRAY_SIZE;
    parallel_initialize_grouped<ExecSpace, MemSpace, KokkosView3< T, MemSpace> >(execObj, hostArrayOfViews[j], initializationValue);
    //parallel_initialize_single<ExecSpace>(execObj, inputs_, inside_value); // safer version, less ambitious
  }
}
#endif  // #if defined(KOKKOS_USING_GPU)
#endif  // #if defined(HAVE_KOKKOS)

//----------------------------------------------------------------------------
// Other loops that should get cleaned up
//----------------------------------------------------------------------------

template <typename Functor>
inline void serial_for(BlockRange const & r, const Functor & functor)
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
    for (int j=jb; j<je; ++j) {
      for (int i=ib; i<ie; ++i) {
        functor(i, j, k);
      }
    }
  }
}

template <typename Functor, typename Option>
inline void parallel_for(BlockRange const & r, const Functor & functor, const Option& op)
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
    for (int j=jb; j<je; ++j) {
      for (int i=ib; i<ie; ++i) {
        functor(op, i, j, k);
      }
    }
  }
};

} // namespace Uintah

#endif // UINTAH_LOOP_EXECUTION_HPP
