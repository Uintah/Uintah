/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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

// The purpose of this file is to provide portability between Kokkos and non-Kokkos builds.
// The user should be able to #include this file, and then obtain all the tools needed.
// For example, suppose a user calls a parallel_for loop but Kokkos is NOT provided, this will run the
// functor in a loop and also not use Kokkos views.  If Kokkos is provided, this creates a
// lambda expression and inside that it contains loops over the functor.  Kokkos Views are also used.
// At the moment we seek to only support regular CPU code, Kokkos OpenMP, and CUDA execution spaces,
// though it shouldn't be too difficult to expand it to others.  This doesn't extend it
// to CUDA kernels (without Kokkos), and that can get trickier (block/dim parameters) especially with
// regard to a parallel_reduce (many ways to "reduce" a value and return it back to host memory)

#ifndef UINTAH_HOMEBREW_LOOP_EXECUTION_HPP
#define UINTAH_HOMEBREW_LOOP_EXECUTION_HPP

#include <Core/Parallel/ExecutionObject.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InternalError.h>

#include <cstring>
#include <cstddef> // TODO: What is this doing here?
#include <vector> //  Used to manage multiple streams in a task.
#include <algorithm>
#include <initializer_list>

#include <sci_defs/cuda_defs.h>
#include <sci_defs/kokkos_defs.h>

#define ARRAY_SIZE 16

using std::max;
using std::min;

#if defined( UINTAH_ENABLE_KOKKOS )
#include <Kokkos_Core.hpp>
  const bool HAVE_KOKKOS = true;
#if !defined( KOKKOS_ENABLE_CUDA )  //This define comes from Kokkos itself.
  //Kokkos GPU is not included in this build.  Create some stub types so these types at least exist.
  namespace Kokkos {
    class Cuda {};
    class CudaSpace {};
  }
#elif !defined( KOKKOS_ENABLE_OPENMP )
  // For the Unified Scheduler + Kokkos GPU but not Kokkos OpenMP.
  // This logic may be temporary if all GPU functionality is merged into the Kokkos scheduler
  // and the Unified Scheduler is no longer used for GPU logic.  Brad P Jun 2018
  namespace Kokkos {
    class OpenMP {};
  }
#endif
#else //if defined( UINTAH_ENABLE_KOKKOS )

  //Kokkos not included in this build.  Create some stub types so these types at least exist.
  namespace Kokkos {
    class OpenMP {};
    class HostSpace {};
    class Cuda {};
    class CudaSpace {};
  }
  const bool HAVE_KOKKOS = false;
#define KOKKOS_LAMBDA [&]

#endif //UINTAH_ENABLE_KOKKOS

// Create some data types for non-Kokkos CPU runs.
namespace UintahSpaces{
  class CPU {};          // These are used for legacy Uintah tasks (e.g. no Kokkos)
  class HostSpace {};    // And also to refer to any data in host memory
#if defined( HAVE_CUDA )
  class GPU {};          // At the moment, this is only used to describe data in Cuda Memory
  class CudaSpace {};    // and not to manage non-Kokkos GPU tasks.
#endif //HAVE_CUDA
}

enum TASKGRAPH {
  DEFAULT = -1
};
// Macros don't like passing in data types that contain commas in them,
// such as two template arguments. This helps fix that.
#define COMMA ,

// Helps turn defines into usable strings (even if it has a comma in it)
#define STRV(...) #__VA_ARGS__
#define STRVX(...) STRV(__VA_ARGS__)

// Boilerplate alert.  This can be made much cleaner in C++17 with compile time if statements (if constexpr() logic.)
// We would be able to use the following "using" statements instead of the "#defines", and do compile time comparisons like so:
// if constexpr  (std::is_same< Kokkos::Cuda,      TAG1 >::value) {
// For C++11, the process is doable but just requires a muck of boilerplating to get there, instead of the above if
// statement, I instead opted for a weirder system where I compared the tags as strings
// if (strcmp(STRVX(ORIGINAL_KOKKOS_CUDA_TAG), STRVX(TAG1)) == 0) {
// Further, the C++11 way ended up defining a tag as more of a string of code rather than an actual data type.

//using UINTAH_CPU_TAG = UintahSpaces::CPU;
//using KOKKOS_OPENMP_TAG = Kokkos::OpenMP;
//using KOKKOS_CUDA_TAG = Kokkos::Cuda;

// Main concept of the below tags: Whatever tags the user supplies is directly compiled into the Uintah binary build.
// In case of a situation where a user supplies a tag that isn't valid for that build, such as KOKKOS_CUDA_TAG in a non-CUDA build,
// the tag is "downgraded" to one that is valid.  So in a non-CUDA build, KOKKOS_CUDA_TAG gets associated with
// Kokkos::OpenMP or UintahSpaces::CPU.  This helps ensure that the compiler never attempts to compile anything with a
// Kokkos::Cuda data type in a non-GPU build
#define ORIGINAL_UINTAH_CPU_TAG     UintahSpaces::CPU COMMA UintahSpaces::HostSpace
#define ORIGINAL_KOKKOS_OPENMP_TAG  Kokkos::OpenMP COMMA Kokkos::HostSpace
#define ORIGINAL_KOKKOS_CUDA_TAG    Kokkos::Cuda COMMA Kokkos::CudaSpace

#if defined(UINTAH_ENABLE_KOKKOS) && defined(HAVE_CUDA)
  #if defined(KOKKOS_ENABLE_OPENMP)
    #define UINTAH_CPU_TAG            Kokkos::OpenMP COMMA Kokkos::HostSpace
    #define KOKKOS_OPENMP_TAG         Kokkos::OpenMP COMMA Kokkos::HostSpace
  #else
    #define UINTAH_CPU_TAG            UintahSpaces::CPU COMMA UintahSpaces::HostSpace
    #define KOKKOS_OPENMP_TAG         UintahSpaces::CPU COMMA UintahSpaces::HostSpace
  #endif
  #define KOKKOS_CUDA_TAG             Kokkos::Cuda COMMA Kokkos::CudaSpace
#elif defined(UINTAH_ENABLE_KOKKOS) && !defined(HAVE_CUDA)
  #if defined(KOKKOS_ENABLE_OPENMP)
    #define UINTAH_CPU_TAG            Kokkos::OpenMP COMMA Kokkos::HostSpace
    #define KOKKOS_OPENMP_TAG         Kokkos::OpenMP COMMA Kokkos::HostSpace
    #define KOKKOS_CUDA_TAG           Kokkos::OpenMP COMMA Kokkos::HostSpace
  #else
    #define UINTAH_CPU_TAG            UintahSpaces::CPU COMMA UintahSpaces::HostSpace
    #define KOKKOS_OPENMP_TAG         UintahSpaces::CPU COMMA UintahSpaces::HostSpace
    #define KOKKOS_CUDA_TAG           UintahSpaces::CPU COMMA UintahSpaces::HostSpace
  #endif
#elif !defined(UINTAH_ENABLE_KOKKOS)
  #define UINTAH_CPU_TAG              UintahSpaces::CPU COMMA UintahSpaces::HostSpace
  #define KOKKOS_OPENMP_TAG           UintahSpaces::CPU COMMA UintahSpaces::HostSpace
  #define KOKKOS_CUDA_TAG             UintahSpaces::CPU COMMA UintahSpaces::HostSpace
#endif

namespace Uintah {

class BlockRange;

class BlockRange
{
public:

  enum { rank = 3 };

  BlockRange(){}

  template <typename ArrayType>
  void setValues( ArrayType const & c0, ArrayType const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
      m_dim[i] =   (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
    }
  }

  template <typename ArrayType>
  BlockRange( ArrayType const & c0, ArrayType const & c1 )
  {
    setValues( c0, c1 );
  }

  BlockRange( const BlockRange& obj ) {
    for (int i=0; i<rank; ++i) {
      this->m_offset[i] = obj.m_offset[i];
      this->m_dim[i] = obj.m_dim[i];
    }
  }

  int begin( int r ) const { return m_offset[r]; }
  int   end( int r ) const { return m_offset[r] + m_dim[r]; }

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

// Lambda expressions for CUDA cannot properly capture plain fixed sized arrays (it passes pointers, not the arrays)
// but they can properly capture and copy a struct of arrays.  These arrays have sizes known at compile time.
// For CUDA, this struct containing an array will do a full clone/by value copy as part of the lambda capture.
// If you require a runtime/variable sized array, that requires a different mechanism involving pools and deep copies,
// and as of yet hasn't been implemented (Brad P.)
template <typename T, unsigned int CAPACITY>
struct struct1DArray
{
  unsigned short int runTime_size{CAPACITY};
  T arr[CAPACITY];
  struct1DArray(){}

  // This constructor copies elements from one container into here.
  template <typename Container>
  struct1DArray(const Container& container, unsigned int runTimeSize) : runTime_size(runTimeSize) {
#ifndef NDEBUG
    if(runTime_size > CAPACITY){
      throw InternalError("ERROR. struct1DArray is not being used properly. The run-time size exceeds the compile time size (std::vector constructor).", __FILE__, __LINE__);
    }
#endif
    for (unsigned int i = 0; i < runTime_size; i++) {
      arr[i] = container[i];
    }
  }

// This constructor supports the initialization list interface
  struct1DArray(  std::initializer_list<T> const myList) : runTime_size(myList.size()) {
#ifndef NDEBUG
    if(runTime_size > CAPACITY){
      throw InternalError("ERROR. struct1DArray is not being used properly. The run-time size exceeds the compile time size (initializer_list constructor).", __FILE__, __LINE__);
    }
#endif
    std::copy(myList.begin(), myList.begin()+runTime_size,arr);
  }

// This constructor allows for only the runtime_size to be specified
  struct1DArray(int  runTimeSize) : runTime_size(runTimeSize) {
#ifndef NDEBUG
    if(runTime_size > CAPACITY){
      throw InternalError("ERROR. struct1DArray is not being used properly. The run-time size exceeds the compile time size (int constructor).", __FILE__, __LINE__);
    }
#endif
  }

  inline T& operator[](unsigned int index) {
    return arr[index];
  }

  inline const T& operator[](unsigned int index) const {
    return arr[index];
  }

  int mySize(){
    return CAPACITY;
  }

}; // end struct struct1DArray

template <typename T, unsigned int CAPACITY_FIRST_DIMENSION, unsigned int CAPACITY_SECOND_DIMENSION>
struct struct2DArray
{
  struct1DArray<T, CAPACITY_SECOND_DIMENSION> arr[CAPACITY_FIRST_DIMENSION];

  struct2DArray(){}
  unsigned short int i_runTime_size{CAPACITY_FIRST_DIMENSION};
  unsigned short int j_runTime_size{CAPACITY_SECOND_DIMENSION};

  // This constructor copies elements from one container into here.
  template <typename Container>
  struct2DArray(const Container& container, int first_dim_runtimeSize=CAPACITY_FIRST_DIMENSION, int second_dim_runtimeSize=CAPACITY_SECOND_DIMENSION) : i_runTime_size(first_dim_runtimeSize) ,  j_runTime_size(second_dim_runtimeSize) {
    for (unsigned int i = 0; i < i_runTime_size; i++) {
      for (unsigned int j = 0; j < j_runTime_size; j++) {
        arr[i][j] = container[i][j];
      }
      arr[i].runTime_size=i_runTime_size;
    }
  }

  inline struct1DArray<T, CAPACITY_SECOND_DIMENSION>& operator[](unsigned int index) {
    return arr[index];
  }

  inline const struct1DArray<T, CAPACITY_SECOND_DIMENSION>& operator[](unsigned int index) const {
    return arr[index];
  }

}; // end struct struct2DArray

struct int_3
{
  int dim[3]; // indices for x y z dimensions

  int_3(){}

  int_3( const int i, const int j, const int k ) {
    dim[0]=i;
    dim[1]=j;
    dim[2]=k;
  }

  inline const int& operator[]( unsigned int index) const {
    return dim[index];
  }

}; // end struct int_3

//----------------------------------------------------------------------------
// Start parallel loops
//----------------------------------------------------------------------------

// -------------------------------------  parallel_for loops  ---------------------------------------------
#if defined(UINTAH_ENABLE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_for( ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor )
{

  const unsigned int i_size = r.end(0) - r.begin(0);
  const unsigned int j_size = r.end(1) - r.begin(1);
  const unsigned int k_size = r.end(2) - r.begin(2);
  const unsigned int rbegin0 = r.begin(0);
  const unsigned int rbegin1 = r.begin(1);
  const unsigned int rbegin2 = r.begin(2);
  const unsigned int numItems = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);

  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP, int>(0, numItems).set_chunk_size(1), [&, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2](int n) {
    const int k = n / (j_size * i_size) + rbegin2;
    const int j = (n / i_size) % j_size + rbegin1;
    const int i = n % i_size + rbegin0;
    functor( i, j, k );
  });
}

#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

#if defined(UINTAH_ENABLE_KOKKOS)
#if defined(HAVE_CUDA)

template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_for( ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor )
{

  // Team policy approach (reuses CUDA threads)

  // Overall goal, split a 3D range requested by the user into various SMs on the GPU.  (In essence, this would be a Kokkos MD_Team+Policy, if one existed)
  // The process requires going from 3D range to a 1D range, partitioning the 1D range into groups that are multiples of 32,
  // then converting that group of 32 range back into a 3D (i,j,k) index.
  unsigned int i_size = r.end(0) - r.begin(0);
  unsigned int j_size = r.end(1) - r.begin(1);
  unsigned int k_size = r.end(2) - r.begin(2);
  unsigned int rbegin0 = r.begin(0);
  unsigned int rbegin1 = r.begin(1);
  unsigned int rbegin2 = r.begin(2);

  // The user has two partitions available.  1) One is the total number of streaming multiprocessors.  2) The other is
  // splitting a task into multiple streams and execution units.

  const unsigned int numItems = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);

  // Get the requested amount of threads per streaming multiprocessor (SM) and number of SMs totals.
  const unsigned int cuda_threads_per_block   = execObj.getCudaThreadsPerBlock();
  const unsigned int cuda_blocks_per_loop     = execObj.getCudaBlocksPerLoop();
  const unsigned int streamPartitions = execObj.getNumStreams();

  // The requested range of data may not have enough work for the requested command line arguments, so shrink them if necessary.
  const unsigned int actual_threads = (numItems / streamPartitions) > (cuda_threads_per_block * cuda_blocks_per_loop)
                                    ? (cuda_threads_per_block * cuda_blocks_per_loop) : (numItems / streamPartitions);
  const unsigned int actual_threads_per_block = (numItems / streamPartitions) > cuda_threads_per_block ? cuda_threads_per_block : (numItems / streamPartitions);
  const unsigned int actual_cuda_blocks_per_loop = (actual_threads - 1) / cuda_threads_per_block + 1;

  for (unsigned int i = 0; i < streamPartitions; i++) {
    
#if defined(NO_STREAM)
    Kokkos::Cuda instanceObject();
    Kokkos::TeamPolicy< Kokkos::Cuda > tp( actual_cuda_blocks_per_loop, actual_threads_per_block );
#else
    void* stream = execObj.getStream(i);
    if (!stream) {
      std::cout << "Error, the CUDA stream must not be nullptr\n" << std::endl;
      SCI_THROW(InternalError("Error, the CUDA stream must not be nullptr.", __FILE__, __LINE__));
    }
    Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
    //Kokkos::TeamPolicy< Kokkos::Cuda, Kokkos::LaunchBounds<640,1> > tp( instanceObject, actual_cuda_blocks_per_loop, actual_threads_per_block );
    Kokkos::TeamPolicy< Kokkos::Cuda > tp( instanceObject, actual_cuda_blocks_per_loop, actual_threads_per_block );
#endif
    
    // Use a Team Policy, this allows us to control how many threads per block and how many blocks are used.
    typedef Kokkos::TeamPolicy< Kokkos::Cuda > policy_type;
    Kokkos::parallel_for ( tp, [=] __device__ ( typename policy_type::member_type thread ) {

      // We are within an SM, and all SMs share the same amount of assigned CUDA threads.
      // Figure out which range of N items this SM should work on (as a multiple of 32).
      const unsigned int currentPartition = i * actual_cuda_blocks_per_loop + thread.league_rank();
      unsigned int estimatedThreadAmount = numItems * (currentPartition) / ( actual_cuda_blocks_per_loop * streamPartitions );
      const unsigned int startingN =  estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
      unsigned int endingN;
      // Check if this is the last partition
      if ( currentPartition + 1 == actual_cuda_blocks_per_loop * streamPartitions ) {
        endingN = numItems;
      } else {
        estimatedThreadAmount = numItems * ( currentPartition + 1 ) / ( actual_cuda_blocks_per_loop * streamPartitions );
        endingN = estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
      }
      const unsigned int totalN = endingN - startingN;
      //printf("league_rank: %d, team_size: %d, team_rank: %d, startingN: %d, endingN: %d, totalN: %d\n", thread.league_rank(), thread.team_size(), thread.team_rank(), startingN, endingN, totalN);

      Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, totalN), [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2] (const int& N) {
        // Craft an i,j,k out of this range
        // This approach works with row-major layout so that consecutive Cuda threads work along consecutive slots in memory.
        //printf("parallel_for team demo - n is %d, league_rank is %d, true n is %d\n", N, thread.league_rank(), (startingN + N));
        const int k = (startingN + N) / (j_size * i_size) + rbegin2;
        const int j = ((startingN + N) / i_size) % j_size + rbegin1;
        const int i = (startingN + N) % i_size + rbegin0;
        // Actually run the functor.
        functor( i, j, k );
      });
    });
  }

#if defined(NO_STREAM)
  cudaDeviceSynchronize();
#endif

// ----------------- Team policy but with one stream -----------------
//  unsigned int i_size = r.end(0) - r.begin(0);
//  unsigned int j_size = r.end(1) - r.begin(1);
//  unsigned int k_size = r.end(2) - r.begin(2);
//  unsigned int rbegin0 = r.begin(0);
//  unsigned int rbegin1 = r.begin(1);
//  unsigned int rbegin2 = r.begin(2);
//
//
//  int cuda_threads_per_block = execObj.getCudaThreadsPerBlock();
//  int cuda_blocks_per_loop   = execObj.getCudaBlocksPerLoop();
//
//  //If 256 threads aren't needed, use less.
//  //But cap at 256 threads total, as this will correspond to 256 threads in a block.
//  //Later the TeamThreadRange will reuse those 256 threads.  For example, if teamThreadRangeSize is 800, then
//  //Cuda thread 0 will be assigned to n = 0, n = 256, n = 512, and n = 768,
//  //Cuda thread 1 will be assigned to n = 1, n = 257, n = 513, and n = 769...
//
//  const int teamThreadRangeSize = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);
//  const int actualThreads = teamThreadRangeSize > cuda_threads_per_block ? cuda_threads_per_block : teamThreadRangeSize;
//
//  typedef Kokkos::TeamPolicy<ExecSpace> policy_type;
//
//  Kokkos::parallel_for (Kokkos::TeamPolicy<ExecSpace>( cuda_blocks_per_loop, actualThreads ),
//                           [=] __device__ ( typename policy_type::member_type thread ) {
//    Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, teamThreadRangeSize), [=] (const int& n) {
//
//      const int i = n / (j_size * k_size) + rbegin0;
//      const int j = (n / k_size) % j_size + rbegin1;
//      const int k = n % k_size + rbegin2;
//      functor( i, j, k );
//    });
//  });

  // ----------------- Range policy with one stream -----------------
//  const int teamThreadRangeSize = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);
//  const int threadsPerGroup = 256;
//  const int actualThreads = teamThreadRangeSize > threadsPerGroup ? threadsPerGroup : teamThreadRangeSize;
//
//  void* stream = r.getStream();
//
//  if (!stream) {
//    std::cout << "Error, the CUDA stream must not be nullptr\n" << std::endl;
//    exit(-1);
//  }
//  Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
//  Kokkos::RangePolicy<Kokkos::Cuda> rangepolicy( instanceObject, 0, actualThreads);
//
//  Kokkos::parallel_for( rangepolicy, KOKKOS_LAMBDA ( const int& n ) {
//    int threadNum = n;
//    while ( threadNum < teamThreadRangeSize ) {
//      const int i = threadNum / (j_size * k_size) + rbegin0;
//      const int j = (threadNum / k_size) % j_size + rbegin1;
//      const int k = threadNum % k_size + rbegin2;
//      functor( i, j, k );
//      threadNum += threadsPerGroup;
//    }
//  });
}
#endif  //#if defined(HAVE_CUDA)
#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    functor(i,j,k);
  }}}
}

//For legacy loops where no execution space was specified as a template parameter.
template < typename Functor>
void
parallel_for( BlockRange const & r, const Functor & functor )
{
  //Force users into using a single CPU thread if they didn't specify OpenMP
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj; // Make an empty object
  parallel_for<UintahSpaces::CPU>( execObj, r, functor );
}

// -------------------------------------  parallel_reduce_sum loops  ---------------------------------------------

#if defined(UINTAH_ENABLE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_reduce_sum(ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, ReductionType & red  )
{
  ReductionType tmp = red;
  const unsigned int i_size = r.end(0) - r.begin(0);
  const unsigned int j_size = r.end(1) - r.begin(1);
  const unsigned int k_size = r.end(2) - r.begin(2);
  const unsigned int rbegin0 = r.begin(0);
  const unsigned int rbegin1 = r.begin(1);
  const unsigned int rbegin2 = r.begin(2);

  // MDRange approach
  //    typedef typename Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3,
  //                                                         Kokkos::Iterate::Left,
  //                                                         Kokkos::Iterate::Left>
  //                                                         > MDPolicyType_3D;
  //
  //    MDPolicyType_3D mdpolicy_3d( {{r.begin(0),r.begin(1),r.begin(2)}}, {{r.end(0),r.end(1),r.end(2)}} );
  //
  //    Kokkos::parallel_reduce( mdpolicy_3d, f, tmp );


  // Manual approach
  //    const int ib = r.begin(0); const int ie = r.end(0);
  //    const int jb = r.begin(1); const int je = r.end(1);
  //    const int kb = r.begin(2); const int ke = r.end(2);
  //
  //    Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecSpace, int>(kb, ke).set_chunk_size(2), KOKKOS_LAMBDA(int k, ReductionType & tmp) {
  //      for (int j=jb; j<je; ++j) {
  //      for (int i=ib; i<ie; ++i) {
  //        f(i,j,k,tmp);
  //      }}
  //    });

  // Team Policy approach
//  const unsigned int teamThreadRangeSize = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);
//
//  const unsigned int actualThreads = teamThreadRangeSize > 16 ? 16 : teamThreadRangeSize;
//
//  typedef Kokkos::TeamPolicy< Kokkos::OpenMP > policy_type;
//
//  Kokkos::parallel_reduce (Kokkos::TeamPolicy< Kokkos::OpenMP >( 1, actualThreads ),
//                           [&] ( typename policy_type::member_type thread, ReductionType& inner_sum ) {
//    //printf("i is %d\n", thread.team_rank());
//
//    Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, teamThreadRangeSize), [&] (const int& n) {
//      const int i = n / (j_size * k_size) + rbegin0;
//      const int j = (n / k_size) % j_size + rbegin1;
//      const int k = n % k_size + rbegin2;
//      functor(i, j, k, inner_sum);
//    });
//  }, tmp);

  //Range policy manual approach:

  const unsigned int numItems = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);

  Kokkos::parallel_reduce( Kokkos::RangePolicy<Kokkos::OpenMP, int>(0, numItems).set_chunk_size(1), [&, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2](const int& n, ReductionType & tmp) {
    const int k = n / (j_size * i_size) + rbegin2;
    const int j = (n / i_size) % j_size + rbegin1;
    const int i = n % i_size + rbegin0;
    ReductionType tmp2=0;
    functor( i, j, k, tmp2 );
    tmp+=tmp2;
  }, red);

}

#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

#if defined(UINTAH_ENABLE_KOKKOS)
#if defined(HAVE_CUDA)
template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_reduce_sum( ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, ReductionType & red )
{
  // Overall goal, split a 3D range requested by the user into various SMs on the GPU.  (In essence, this would be a Kokkos MD_Team+Policy, if one existed)
  // The process requires going from 3D range to a 1D range, partitioning the 1D range into groups that are multiples of 32,
  // then converting that group of 32 range back into a 3D (i,j,k) index.
  ReductionType tmp = red;
  unsigned int i_size = r.end(0) - r.begin(0);
  unsigned int j_size = r.end(1) - r.begin(1);
  unsigned int k_size = r.end(2) - r.begin(2);
  unsigned int rbegin0 = r.begin(0);
  unsigned int rbegin1 = r.begin(1);
  unsigned int rbegin2 = r.begin(2);

  // The user has two partitions available.  1) One is the total number of streaming multiprocessors.  2) The other is
  // splitting a task into multiple streams and execution units.

  const unsigned int numItems = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);

  // Get the requested amount of threads per streaming multiprocessor (SM) and number of SMs totals.
  const unsigned int cuda_threads_per_block = execObj.getCudaThreadsPerBlock();
  const unsigned int cuda_blocks_per_loop     = execObj.getCudaBlocksPerLoop();
  const unsigned int streamPartitions = execObj.getNumStreams();

  // The requested range of data may not have enough work for the requested command line arguments, so shrink them if necessary.
  const unsigned int actual_threads = (numItems / streamPartitions) > (cuda_threads_per_block * cuda_blocks_per_loop)
                                    ? (cuda_threads_per_block * cuda_blocks_per_loop) : (numItems / streamPartitions);
  const unsigned int actual_threads_per_block = (numItems / streamPartitions) > cuda_threads_per_block ? cuda_threads_per_block : (numItems / streamPartitions);
  const unsigned int actual_cuda_blocks_per_loop = (actual_threads - 1) / cuda_threads_per_block + 1;
  for (unsigned int i = 0; i < streamPartitions; i++) {

#if defined(NO_STREAM)
    Kokkos::Cuda instanceObject();
    Kokkos::TeamPolicy< Kokkos::Cuda > reduce_tp( actual_cuda_blocks_per_loop, actual_threads_per_block );
#else
    void* stream = execObj.getStream(i);
    if (!stream) {
      std::cout << "Error, the CUDA stream must not be nullptr\n" << std::endl;
      SCI_THROW(InternalError("Error, the CUDA stream must not be nullptr.", __FILE__, __LINE__));
    }
    Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
    //Kokkos::TeamPolicy< Kokkos::Cuda, Kokkos::LaunchBounds<640,1>  > reduce_tp( instanceObject, actual_cuda_blocks_per_loop, actual_threads_per_block );
    Kokkos::TeamPolicy< Kokkos::Cuda > reduce_tp( instanceObject, actual_cuda_blocks_per_loop, actual_threads_per_block );
#endif
    
    // Use a Team Policy, this allows us to control how many threads per block and how many blocks are used.
    typedef Kokkos::TeamPolicy< Kokkos::Cuda > policy_type;
    Kokkos::parallel_reduce ( reduce_tp, [=] __device__ ( typename policy_type::member_type thread, ReductionType& inner_sum ) {

      // We are within an SM, and all SMs share the same amount of assigned CUDA threads.
      // Figure out which range of N items this SM should work on (as a multiple of 32).
      const unsigned int currentPartition = i * actual_cuda_blocks_per_loop + thread.league_rank();
      unsigned int estimatedThreadAmount = numItems * (currentPartition) / ( actual_cuda_blocks_per_loop * streamPartitions );
      const unsigned int startingN =  estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
      unsigned int endingN;
      // Check if this is the last partition
      if ( currentPartition + 1 == actual_cuda_blocks_per_loop * streamPartitions ) {
        endingN = numItems;
      } else {
        estimatedThreadAmount = numItems * ( currentPartition + 1 ) / ( actual_cuda_blocks_per_loop * streamPartitions );
        endingN = estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
      }
      const unsigned int totalN = endingN - startingN;
      //printf("league_rank: %d, team_size: %d, team_rank: %d, startingN: %d, endingN: %d, totalN: %d\n", thread.league_rank(), thread.team_size(), thread.team_rank(), startingN, endingN, totalN);

      Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, totalN), [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2] (const int& N) {
        // Craft an i,j,k out of this range. 
        // This approach works with row-major layout so that consecutive Cuda threads work along consecutive slots in memory.
        //printf("reduce team demo - n is %d, league_rank is %d, true n is %d\n", N, thread.league_rank(), (startingN + N));
        const int k = (startingN + N) / (j_size * i_size) + rbegin2;
        const int j = ((startingN + N) / i_size) % j_size + rbegin1;
        const int i = (startingN + N) % i_size + rbegin0;
        // Actually run the functor.
        ReductionType tmp2=0;
        functor(i,j,k, tmp2);
        inner_sum+=tmp2;
      });
    }, tmp);

    red = tmp;
  }

#if defined(NO_STREAM)
  cudaDeviceSynchronize();
#endif

  //  Manual approach using range policy that shares threads.
//  const int teamThreadRangeSize = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);
//  const int threadsPerGroup = 256;
//  const int actualThreads = teamThreadRangeSize > threadsPerGroup ? threadsPerGroup : teamThreadRangeSize;
//
//  void* stream = r.getStream();
//
//  if (!stream) {
//    std::cout << "Error, the CUDA stream must not be nullptr\n" << std::endl;
//    exit(-1);
//  }
//  Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
//  Kokkos::RangePolicy<Kokkos::Cuda> rangepolicy( instanceObject, 0, actualThreads);
//
//  Kokkos::parallel_reduce( rangepolicy, KOKKOS_LAMBDA ( const int& n, ReductionType & inner_tmp ) {
//    int threadNum = n;
//    while ( threadNum < teamThreadRangeSize ) {
//      const int i = threadNum / (j_size * k_size) + rbegin0;
//      const int j = (threadNum / k_size) % j_size + rbegin1;
//      const int k = threadNum % k_size + rbegin2;
//      functor( i, j, k, inner_tmp );
//      threadNum += threadsPerGroup;
//    }
//  }, tmp);
//
//  red = tmp;

}
#endif  //#if defined(HAVE_CUDA)
#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_reduce_sum( ExecutionObject<ExecSpace, MemSpace>& execObj, BlockRange const & r, const Functor & functor, ReductionType & red  )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    ReductionType tmp = 0;
    functor(i,j,k,tmp);
    red+=tmp;
  }}}
}

//For legacy loops where no execution space was specified as a template parameter.
template < typename Functor, typename ReductionType>
void
parallel_reduce_sum( BlockRange const & r, const Functor & functor, ReductionType & red )
{
  //Force users into using a single CPU thread if they didn't specify OpenMP
  ExecutionObject<UintahSpaces::CPU, UintahSpaces::HostSpace> execObj; // Make an empty object
  parallel_reduce_sum<UintahSpaces::CPU>( execObj, r, functor, red );
}

// -------------------------------------  parallel_reduce_min loops  ---------------------------------------------


#if defined(UINTAH_ENABLE_KOKKOS)
template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_reduce_min( ExecutionObject<ExecSpace, MemSpace>& execObj,
                     BlockRange const & r, const Functor & functor, ReductionType & red  )
{
  ReductionType tmp0 = red;

  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  // Manual approach
  Kokkos::parallel_reduce( Kokkos::RangePolicy<Kokkos::OpenMP, int>(kb, ke).set_chunk_size(1), [=](int k, ReductionType & tmp1) {
    ReductionType tmp2;
    for (int j=jb; j<je; ++j) {
    for (int i=ib; i<ie; ++i) {
      functor(i,j,k,tmp2);
      tmp1=min(tmp2,tmp1);
    }}
  }, Kokkos::Min<ReductionType>(tmp0));

  red = min(tmp0,red); 

}
#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

#if defined(UINTAH_ENABLE_KOKKOS)
#if defined(HAVE_CUDA)
template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_reduce_min( ExecutionObject<ExecSpace, MemSpace>& execObj,  
                     BlockRange const & r, const Functor & functor, ReductionType & red )
{
  ReductionType tmp = red;
  unsigned int i_size = r.end(0) - r.begin(0);
  unsigned int j_size = r.end(1) - r.begin(1);
  unsigned int k_size = r.end(2) - r.begin(2);
  unsigned int rbegin0 = r.begin(0);
  unsigned int rbegin1 = r.begin(1);
  unsigned int rbegin2 = r.begin(2);

  // The user has two partitions available.  1) One is the total number of streaming multiprocessors.  2) The other is
  // splitting a task into multiple streams and execution units.

  const unsigned int numItems = (i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) * (k_size > 0 ? k_size : 1);

  // Get the requested amount of threads per streaming multiprocessor (SM) and number of SMs totals.
  const unsigned int cuda_threads_per_block = execObj.getCudaThreadsPerBlock();
  const unsigned int cuda_blocks_per_loop     = execObj.getCudaBlocksPerLoop();
  const unsigned int streamPartitions = execObj.getNumStreams();

  // The requested range of data may not have enough work for the requested command line arguments, so shrink them if necessary.
  const unsigned int actual_threads = (numItems / streamPartitions) > (cuda_threads_per_block * cuda_blocks_per_loop)
                                    ? (cuda_threads_per_block * cuda_blocks_per_loop) : (numItems / streamPartitions);
  const unsigned int actual_threads_per_block = (numItems / streamPartitions) > cuda_threads_per_block ? cuda_threads_per_block : (numItems / streamPartitions);
  const unsigned int actual_cuda_blocks_per_loop = (actual_threads - 1) / cuda_threads_per_block + 1;
  for (unsigned int i = 0; i < streamPartitions; i++) {

#if defined(NO_STREAM)
    Kokkos::Cuda instanceObject();
    Kokkos::TeamPolicy< Kokkos::Cuda > reduce_tp( actual_cuda_blocks_per_loop, actual_threads_per_block );
#else
    void* stream = execObj.getStream(i);
    if (!stream) {
      std::cout << "Error, the CUDA stream must not be nullptr\n" << std::endl;
      SCI_THROW(InternalError("Error, the CUDA stream must not be nullptr.", __FILE__, __LINE__));
    }
    Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
    //Kokkos::TeamPolicy< Kokkos::Cuda, Kokkos::LaunchBounds<640,1>  > reduce_tp( instanceObject, actual_cuda_blocks_per_loop, actual_threads_per_block );
    Kokkos::TeamPolicy< Kokkos::Cuda > reduce_tp( instanceObject, actual_cuda_blocks_per_loop, actual_threads_per_block );
#endif
    
    // Use a Team Policy, this allows us to control how many threads per block and how many blocks are used.
    typedef Kokkos::TeamPolicy< Kokkos::Cuda > policy_type;
    Kokkos::parallel_reduce ( reduce_tp, [=] __device__ ( typename policy_type::member_type thread, ReductionType& inner_min ) {


      // We are within an SM, and all SMs share the same amount of assigned CUDA threads.
      // Figure out which range of N items this SM should work on (as a multiple of 32).
      const unsigned int currentPartition = i * actual_cuda_blocks_per_loop + thread.league_rank();
      unsigned int estimatedThreadAmount = numItems * (currentPartition) / ( actual_cuda_blocks_per_loop * streamPartitions );
      const unsigned int startingN =  estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
      unsigned int endingN;
      // Check if this is the last partition
      if ( currentPartition + 1 == actual_cuda_blocks_per_loop * streamPartitions ) {
        endingN = numItems;
      } else {
        estimatedThreadAmount = numItems * ( currentPartition + 1 ) / ( actual_cuda_blocks_per_loop * streamPartitions );
        endingN = estimatedThreadAmount + ((estimatedThreadAmount % 32 == 0) ? 0 : (32-estimatedThreadAmount % 32));
      }
      const unsigned int totalN = endingN - startingN;
      //printf("league_rank: %d, team_size: %d, team_rank: %d, startingN: %d, endingN: %d, totalN: %d\n", thread.league_rank(), thread.team_size(), thread.team_rank(), startingN, endingN, totalN);

      Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, totalN), [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1, rbegin2] (const int& N) {
        // Craft an i,j,k out of this range. 
        // This approach works with row-major layout so that consecutive Cuda threads work along consecutive slots in memory.
        //printf("reduce team demo - n is %d, league_rank is %d, true n is %d\n", N, thread.league_rank(), (startingN + N));
        int k = (startingN + N) / (j_size * i_size) + rbegin2;
        int j = ((startingN + N) / i_size) % j_size + rbegin1;
        int i = (startingN + N) % i_size + rbegin0;
        // Actually run the functor.
        ReductionType tmp2;
        functor(i,j,k, tmp2 );
        inner_min= inner_min > tmp2 ? tmp2 : inner_min;
      });
    }, Kokkos::Min<ReductionType>(tmp));

    red = tmp;
  }

#if defined(NO_STREAM)
  cudaDeviceSynchronize();
#endif


}
#endif  //#if defined(HAVE_CUDA)
#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

//TODO: This appears to not do any "min" on the reduction.
template <typename ExecSpace, typename MemSpace, typename Functor, typename ReductionType>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_reduce_min( ExecutionObject<ExecSpace, MemSpace>& execObj,
                     BlockRange const & r, const Functor & functor, ReductionType & red  )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  ReductionType tmp = red;
  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    functor(i,j,k,tmp);
    red=min(tmp,red);
  }}}
}

template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU >::value, void>::type
sweeping_parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj,  BlockRange const & r, const Functor & functor, const bool plusX, const bool plusY, const bool plusZ , const int npart)
{

    const int   idir= plusX ? 1 : -1; 
    const int   jdir= plusY ? 1 : -1; 
    const int   kdir= plusZ ? 1 : -1; 

    const int start_x= plusX ? r.begin(0) : r.end(0)-1;
    const int start_y= plusY ? r.begin(1) : r.end(1)-1;
    const int start_z= plusZ ? r.begin(2) : r.end(2)-1;

    const int end_x= plusX ? r.end(0) : -r.begin(0)+1; 
    const int end_y= plusY ? r.end(1) : -r.begin(1)+1;
    const int end_z= plusZ ? r.end(2) : -r.begin(2)+1; 

    for (int k=start_z; k*kdir<end_z; k=k+kdir) {
      for (int j=start_y; j*jdir<end_y; j=j+jdir) {
        for (int i=start_x ; i*idir<end_x; i=i+idir) {
          functor(i,j,k);
        }}}

}

// -------------------------------------  sweeping_parallel_for loops  ---------------------------------------------
#if defined(UINTAH_ENABLE_KOKKOS)
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
sweeping_parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj,  BlockRange const & r, const Functor & functor, const bool plusX, const bool plusY, const bool plusZ , const int npart)
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  ////////////CUBIC BLOCKS SUPPORTED ONLY/////////////////// 
  // RECTANGLES ARE HARD BUT POSSIBLYE MORE EFFICIENT //////
  // ///////////////////////////////////////////////////////
  const int nPartitionsx=npart; // try to break domain into nxnxn block
  const int nPartitionsy=npart; 
  const int nPartitionsz=npart; 
  const int  dx=ie-ib;
  const int  dy=je-jb;
  const int  dz=ke-kb;
  const int  sdx=dx/nPartitionsx;
  const int  sdy=dy/nPartitionsy;
  const int  sdz=dz/nPartitionsz;
  const int  rdx=dx-sdx*nPartitionsx;
  const int  rdy=dy-sdy*nPartitionsy;
  const int  rdz=dz-sdz*nPartitionsz;

  const int nphase=nPartitionsx+nPartitionsy+nPartitionsz-2;
  int tpp=0; //  Total parallel processes/blocks

  int concurrentBlocksArray[nphase/2+1]; // +1 needed for odd values, use symmetry

  for (int iphase=0; iphase <nphase; iphase++ ){
    if  ((nphase-iphase -1)>= iphase){
      tpp=(iphase+2)*(iphase+1)/2;

      tpp-=max(iphase-nPartitionsx+1,0)*(iphase-nPartitionsx+2)/2;
      tpp-=max(iphase-nPartitionsy+1,0)*(iphase-nPartitionsy+2)/2;
      tpp-=max(iphase-nPartitionsz+1,0)*(iphase-nPartitionsz+2)/2;

      concurrentBlocksArray[iphase]=tpp;
    }else{
      tpp=concurrentBlocksArray[nphase-iphase-1];
    }

    Kokkos::View<int*, Kokkos::HostSpace> xblock("xblock",tpp) ;
    Kokkos::View<int*, Kokkos::HostSpace> yblock("yblock",tpp) ;
    Kokkos::View<int*, Kokkos::HostSpace> zblock("zblock",tpp) ;

    int icount = 0 ;
    for (int k=0;  k< min(iphase+1,nPartitionsz);  k++ ){ // attempts to iterate over k j i , despite  spatial dependencies
      for (int j=0;  j< min(iphase-k+1,nPartitionsy);  j++ ){
        if ((iphase -k-j) <nPartitionsx){
          xblock(icount)=iphase-k-j;
          yblock(icount)=j;
          zblock(icount)=k;
          icount++;
        }
      }
    }

    ///////// Multidirectional parameters
    const int   idir= plusX ? 1 : -1; 
    const int   jdir= plusY ? 1 : -1; 
    const int   kdir= plusZ ? 1 : -1; 

    Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP, int>(0, tpp).set_chunk_size(2), [=](int iblock) {
      const int  xiBlock = plusX ? xblock(iblock) : nPartitionsx-xblock(iblock)-1;
      const int  yiBlock = plusY ? yblock(iblock) : nPartitionsx-yblock(iblock)-1;
      const int  ziBlock = plusZ ? zblock(iblock) : nPartitionsx-zblock(iblock)-1;

      const int blockx_start=ib+xiBlock *sdx;
      const int blocky_start=jb+yiBlock *sdy;
      const int blockz_start=kb+ziBlock *sdz;

      const int blockx_end= ib+ (xiBlock+1)*sdx +(xiBlock+1 ==nPartitionsx ?  rdx:0 );
      const int blocky_end= jb+ (yiBlock+1)*sdy +(yiBlock+1 ==nPartitionsy ?  rdy:0 );
      const int blockz_end= kb+ (ziBlock+1)*sdz +(ziBlock+1 ==nPartitionsz ?  rdz:0 );

      const int blockx_end_dir= plusX ? blockx_end :-blockx_start+1 ;
      const int blocky_end_dir= plusY ? blocky_end :-blocky_start+1 ;
      const int blockz_end_dir= plusZ ? blockz_end :-blockz_start+1 ;

      for (int k=plusZ? blockz_start : blockz_end-1; k*kdir<blockz_end_dir; k=k+kdir) {
        for (int j=plusY? blocky_start : blocky_end-1; j*jdir<blocky_end_dir; j=j+jdir) {
          for (int i=plusX? blockx_start : blockx_end-1; i*idir<blockx_end_dir; i=i+idir) {
            functor(i,j,k);
          }
        }
      }
    }); // end Kokkos::parallel_for
  } // end for ( int iphase = 0; iphase < nphase; iphase++ )
}
#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

#if defined(UINTAH_ENABLE_KOKKOS)
#if defined(HAVE_CUDA)
template <typename ExecSpace, typename MemSpace, typename Functor>
inline typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
sweeping_parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj,  BlockRange const & r, const Functor & functor, const bool plusX, const bool plusY, const bool plusZ , const int npart)
{

    SCI_THROW(InternalError("Error: sweeps on cuda has not been implimented .", __FILE__, __LINE__));
}
#endif
#endif

// Allows the user to specify a vector (or view) of indices that require an operation,
// often needed for boundary conditions and possibly structured grids
// TODO: Can this be called parallel_for_unstructured?
#if defined(UINTAH_ENABLE_KOKKOS)
template <typename ExecSpace, typename MemSpace, typename Functor>
typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
             Kokkos::View<int_3*, Kokkos::HostSpace> iterSpace ,const unsigned int list_size , const Functor & functor )
{
  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP, int>(0, list_size).set_chunk_size(1), [=](const unsigned int & iblock) {
    functor(iterSpace[iblock][0],iterSpace[iblock][1],iterSpace[iblock][2]);
  });
}
#endif  //#if defined(UINTAH_ENABLE_KOKKOS)

//Allows the user to specify a vector (or view) of indices that require an operation, often needed for boundary conditions and possibly structured grids
//This GPU version is mostly a copy of the original GPU version
// TODO: Can this be called parallel_for_unstructured?
// TODO: Make streamable.
#if defined(UINTAH_ENABLE_KOKKOS)
#if defined(HAVE_CUDA)
template <typename ExecSpace, typename MemSpace, typename Functor>
typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
             Kokkos::View<int_3*, Kokkos::CudaSpace> iterSpace ,const unsigned int list_size , const Functor & functor )
{

  void* stream = execObj.getStream();
  Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
  Kokkos::RangePolicy< Kokkos::Cuda > policy(instanceObject, 0, list_size);

  Kokkos::parallel_for (policy, KOKKOS_LAMBDA (const unsigned int& iblock) {
    functor(iterSpace[iblock][0],iterSpace[iblock][1],iterSpace[iblock][2]);
  });

  /*
  unsigned int cudaThreadsPerBlock = execObj.getCudaThreadsPerBlock();
  unsigned int cudaBlocksPerLoop   = execObj.getCudaBlocksPerLoop();

  const unsigned int actualThreads = list_size > cudaThreadsPerBlock ? cudaThreadsPerBlock : list_size;

#if defined(NO_STREAM)
  Kokkos::Cuda instanceObject();
  Kokkos::TeamPolicy< Kokkos::Cuda > teamPolicy( cudaBlocksPerLoop, actualThreads );
#else
  void* stream = execObj.getStream();
  if (!stream) {
    std::cout << "Error, the CUDA stream must not be nullptr\n" << std::endl;
    SCI_THROW(InternalError("Error, the CUDA stream must not be nullptr.", __FILE__, __LINE__));
  }
  Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
  Kokkos::TeamPolicy< Kokkos::Cuda > teamPolicy( instanceObject, cudaBlocksPerLoop, actualThreads );
#endif

  typedef Kokkos::TeamPolicy< Kokkos::Cuda > policy_type;

  Kokkos::parallel_for (teamPolicy, KOKKOS_LAMBDA ( typename policy_type::member_type thread ) {

    const unsigned int currentBlock = thread.league_rank();
    Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, list_size), [&,iterSpace] (const unsigned int& iblock) {
    functor(iterSpace[iblock][0],iterSpace[iblock][1],iterSpace[iblock][2]);
      });
    });
  */

#if defined(NO_STREAM)
  cudaDeviceSynchronize();
#endif

}
#endif //HAVE_CUDA
#endif //UINTAH_ENABLE_KOKKOS

#if defined(UINTAH_ENABLE_KOKKOS)


#if defined(HAVE_CUDA)
// TODO: What is this?  It needs a better name
template <typename ExecSpace, typename MemSpace, typename T2, typename T3>
typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_for(ExecutionObject<ExecSpace, MemSpace>& execObj, T2 KV3, const T3 init_val)
{
  int cuda_threads_per_block = execObj.getCudaThreadsPerBlock();
  int cuda_blocks_per_loop   = execObj.getCudaBlocksPerLoop();

  const int num_cells=KV3.m_view.size();
  const int actualThreads = num_cells > cuda_threads_per_block ? cuda_threads_per_block : num_cells;

  typedef Kokkos::TeamPolicy<ExecSpace> policy_type;

  Kokkos::parallel_for (Kokkos::TeamPolicy<ExecSpace>( cuda_blocks_per_loop, actualThreads ),
                        KOKKOS_LAMBDA ( typename policy_type::member_type thread ) {
    Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, num_cells), [=] (const int& i) {
       KV3(i)=init_val;
    });
  });
}
#endif  //defined(HAVE_CUDA)
#endif  //defined(UINTAH_ENABLE_KOKKOS)

// ------------------------------  parallel_initialize loops and its helper functions  ------------------------------
// Initialization API that takes both KokkosView3 arguments and View<KokkosView3> arguments.
// If compiling w/o kokkos it takes CC NC SFCX SFCY SFCZ arguments and std::vector<T> arguments
// ------------------------------------------------------------------------------------------------------------------

#if defined(UINTAH_ENABLE_KOKKOS)

template <typename ExecSpace, typename MemSpace, typename T, typename ValueType>
typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_initialize_grouped(ExecutionObject<ExecSpace, MemSpace>& execObj,
                            const struct1DArray<T, ARRAY_SIZE>& KKV3,  const ValueType& init_val ){

  //TODO: This should probably be serialized and not use a Kokkos::parallel_for?

  unsigned int n_cells = 0;
  for (unsigned int j = 0; j < KKV3.runTime_size; j++){
    n_cells += KKV3[j].m_view.size();
  }

  Kokkos::parallel_for( Kokkos::RangePolicy<ExecSpace, int>(0, n_cells).set_chunk_size(2), [=](unsigned int i_tot) {
    // TODO: Find a more efficient way of doing this.
    int i = i_tot;
    int j = 0;
    while ( i-(int) KKV3[j].m_view.size() > -1 ){
      i -= KKV3[j].m_view.size();
      j++;
    }
    KKV3[j](i)=init_val;
  });
}

//template <typename ExecSpace, typename MemSpace, typename T2, typename T3>
//typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
//parallel_initialize_single(ExecutionObject& execObj, T2 KKV3, const T3 init_val ){
//  for (unsigned int j=0; j < KKV3.size(); j++){
//    Kokkos::parallel_for( Kokkos::RangePolicy<ExecSpace, int>(0,KKV3(j).m_view.size() ).set_chunk_size(2), [=](int i) {
//      KKV3(j)(i)=init_val;
//    });
//  }
//}

#endif //UINTAH_ENABLE_KOKKOS

template <class TTT> // Needed for the casting inside of the Variadic template, also allows for nested templating
using Alias = TTT;

#if defined(UINTAH_ENABLE_KOKKOS)

template <typename T, typename MemSpace>   //Forward Declaration of KokkosView3
class KokkosView3;


#if defined(HAVE_CUDA)

/* DS 11052019: Wrote alternative (and simpler) version of parallel_initialize_grouped for cuda
 * The previous version seems to produce error in parallel_initialize. 
 * This version finds the max numb of cells among variables and uses it as an
 * iteration count. Using simpler RangePolicy instead of TeamPolicy. All computations
 * to find out index in TeamPolicy - especially divides and mods do not seem worth for 
 * simple init code. Secondly iterating over all variables within struct1DArray manually
 * rather than spawning extra threads. This reduces degreee of parallelism, but produces 
 * correct result. Revisit later if it becomes a bottleneck. 
 */
template <typename ExecSpace, typename MemSpace, typename T, typename ValueType>
typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_initialize_grouped(ExecutionObject<ExecSpace, MemSpace>& execObj,
		const struct1DArray<T, ARRAY_SIZE>& KKV3, const ValueType& init_val ){

	// n_cells is the max of cells total to process among collection of vars (the view of Kokkos views)
	// For example, if this were being used to  if one var had 4096 cells and another var had 5832 cells, n_cells would become 5832

	unsigned int n_cells = 0;
	for (unsigned int j = 0; j < KKV3.runTime_size; j++){
		n_cells = KKV3[j].m_view.size() > n_cells ? KKV3[j].m_view.size() : n_cells;
	}
#if defined(NO_STREAM)
	Kokkos::RangePolicy< Kokkos::Cuda > policy(0, n_cells);
#else
	void* stream = execObj.getStream();
	Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
	Kokkos::RangePolicy< Kokkos::Cuda > policy(instanceObject, 0, n_cells);
#endif

	Kokkos::parallel_for (policy, KOKKOS_LAMBDA (int i){
		for(int j=0; j<KKV3.runTime_size; j++){
			if(i<KKV3[j].m_view.size())
				KKV3[j](i) = init_val;
		}
	});
}

/*
template <typename ExecSpace, typename MemSpace, typename T, typename ValueType>
typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_initialize_grouped(ExecutionObject<ExecSpace, MemSpace>& execObj,
                            const struct1DArray<T, ARRAY_SIZE>& KKV3, const ValueType& init_val ){

  // n_cells is the amount of cells total to process among collection of vars (the view of Kokkos views)
  // For example, if this were being used to  if one var had 4096 cells and another var had 5832 cells, n_cells would become 4096+5832=

  unsigned int n_cells = 0;
  for (unsigned int j = 0; j < KKV3.runTime_size; j++){
    n_cells += KKV3[j].m_view.size();
  }

  unsigned int cudaThreadsPerBlock = execObj.getCudaThreadsPerBlock();
  unsigned int cudaBlocksPerLoop   = execObj.getCudaBlocksPerLoop();

  const unsigned int actualThreads = n_cells > cudaThreadsPerBlock ? cudaThreadsPerBlock : n_cells;

#if defined(NO_STREAM)
  Kokkos::Cuda instanceObject();
  Kokkos::TeamPolicy< Kokkos::Cuda > teamPolicy( cudaBlocksPerLoop, actualThreads );
#else
  void* stream = execObj.getStream();
  if (!stream) {
    std::cout << "Error, the CUDA stream must not be nullptr\n" << std::endl;
    SCI_THROW(InternalError("Error, the CUDA stream must not be nullptr.", __FILE__, __LINE__));
  }
  Kokkos::Cuda instanceObject(*(static_cast<cudaStream_t*>(stream)));
  Kokkos::TeamPolicy< Kokkos::Cuda > teamPolicy( instanceObject, cudaBlocksPerLoop, actualThreads );
#endif

  typedef Kokkos::TeamPolicy< Kokkos::Cuda > policy_type;

  Kokkos::parallel_for (teamPolicy, KOKKOS_LAMBDA ( typename policy_type::member_type thread ) {

      // i_tot will come in as a number between 0 and actualThreads.  Suppose actualThreads is 256.
      // Thread 0 should work on cell 0, thread 1 should work on cell 1, ... thread 255 should work on cell 255
      // then they all advanced forward by actualThreads.
      // Thread 0 works on cell 256, thread 1 works on cell 257... thread 511 works on cell 511.
      // This should continue until all cells are completed.
    Kokkos::parallel_for (Kokkos::TeamThreadRange(thread, actualThreads), [&, n_cells, actualThreads, KKV3] (const unsigned int& i_tot) {
      const unsigned int n_iter = n_cells / actualThreads  + (n_cells % actualThreads > 0 ? 1 : 0); // round up (more efficient to compute this outside parallel_for?)
      unsigned int  j = 0;
      unsigned int old_i = 0;
      for (unsigned int i = 0; i < n_iter; i++) {
         while ( i * actualThreads + i_tot - old_i >= KKV3[j].m_view.size() ) { // using a while for small data sets or massive streaming multiprocessors
           old_i += KKV3[j].m_view.size();
           j++;
           if ( KKV3.runTime_size <= j ){
             return; // do nothing
           }
         }
         KKV3[j]( i * actualThreads + i_tot - old_i ) = init_val;
      }
    });
  });

#if defined(NO_STREAM)
  cudaDeviceSynchronize();
#endif
}
*/
#endif //HAVE_CUDA

//For array of Views
template<typename T, typename MemSpace, unsigned int Capacity>
inline void setValueAndReturnView(struct1DArray<T, Capacity>* V, const T& x, int &index){
  V[index / ARRAY_SIZE][index % ARRAY_SIZE] = x;
  index++;
  return;
}

//For array of Views
template<typename T, typename MemSpace, unsigned int Capacity1, unsigned int Capacity2>
inline void setValueAndReturnView(struct1DArray<T, Capacity1>* V,  const struct1DArray<T, Capacity2>& small_v, int &index){
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

template <typename ExecSpace, typename MemSpace, typename T, class ...Ts>  // Could this be modified to accept grid variables AND containers of grid variables?
typename std::enable_if<std::is_same<ExecSpace, Kokkos::OpenMP>::value, void>::type
parallel_initialize(ExecutionObject<ExecSpace, MemSpace>& execObj, const T& initializationValue,  Ts & ... inputs) {

  // Count the number of views used (sometimes they may be views of views)
  int n = 0 ; // Get number of variadic arguments
  Alias<int[]>{( //first part of magic unpacker
      sumViewSize<KokkosView3< T, MemSpace>, MemSpace>(inputs, n)
      ,0)...,0}; //second part of magic unpacker

  // Allocate space in host memory to track n total views.
  const int n_init_groups = ((n-1) / ARRAY_SIZE) + 1;
  struct1DArray< KokkosView3< T, MemSpace>, ARRAY_SIZE > hostArrayOfViews[n_init_groups];

  // Copy over the views one by one into this view of views.
  int i = 0; //iterator counter
  Alias<int[]>{( //first part of magic unpacker
      setValueAndReturnView< KokkosView3< T, MemSpace>, MemSpace>(hostArrayOfViews, inputs, i)
      ,0)...,0}; //second part of magic unpacker

  for (int j=0; j<n_init_groups; j++){
    hostArrayOfViews[j].runTime_size=n >ARRAY_SIZE * (j+1) ? ARRAY_SIZE : n % ARRAY_SIZE+1;
    parallel_initialize_grouped<ExecSpace, MemSpace, KokkosView3< T, MemSpace> >(execObj, hostArrayOfViews[j], initializationValue );
    //parallel_initialize_single<ExecSpace>(execObj, inputs_, inside_value ); // safer version, less ambitious
  }
}

////This function is the same as above,but appears to be necessary due to CCVariable support.....
template <typename ExecSpace, typename MemSpace, typename T, class ...Ts>  // Could this be modified to accept grid variables AND containers of grid variables?
typename std::enable_if<std::is_same<ExecSpace, Kokkos::Cuda>::value, void>::type
parallel_initialize(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    const T & initializationValue,  Ts & ... inputs) {

  // Count the number of views used (sometimes they may be views of views)
  int n = 0 ; // Get number of variadic arguments
  Alias<int[]>{( //first part of magic unpacker
      sumViewSize< KokkosView3< T, MemSpace>, Kokkos::HostSpace >(inputs, n)
      ,0)...,0}; //second part of magic unpacker

  const int n_init_groups = ((n-1)/ARRAY_SIZE) + 1;
  struct1DArray< KokkosView3< T, MemSpace>, ARRAY_SIZE > hostArrayOfViews[n_init_groups];

  // Copy over the views one by one into this view of views.
  int i=0; //iterator counter
  Alias<int[]>{( //first part of magic unpacker
      setValueAndReturnView< KokkosView3< T, MemSpace>, Kokkos::HostSpace >(hostArrayOfViews, inputs, i)
      ,0)...,0}; //second part of magic unpacker

  for (int j=0; j<n_init_groups; j++){
    //DS 11052019: setting else part to n % ARRAY_SIZE instead of n % ARRAY_SIZE+1. Why +1? It adds one extra variable, which does not exist
    //At least matches with the alternative implementation
    //hostArrayOfViews[j].runTime_size=n >ARRAY_SIZE * (j+1) ? ARRAY_SIZE : n % ARRAY_SIZE+1;    
	hostArrayOfViews[j].runTime_size=n >ARRAY_SIZE * (j+1) ? ARRAY_SIZE : n % ARRAY_SIZE;
    parallel_initialize_grouped<ExecSpace, MemSpace, KokkosView3< T, MemSpace> >( execObj, hostArrayOfViews[j], initializationValue );
    //parallel_initialize_single<ExecSpace>(execObj, inputs_, inside_value ); // safer version, less ambitious
  }
}

#endif // defined(UINTAH_ENABLE_KOKKOS)

template< typename T, typename T2, const unsigned int T3>
void legacy_initialize(T inside_value, struct1DArray<T2,T3>& data_fields) {  // for vectors
  int nfields=data_fields.runTime_size;
  for(int i=0;  i< nfields; i++){
    data_fields[i].initialize(inside_value);
  }
  return;
}

template< typename T, typename T2>
void legacy_initialize(T inside_value, T2& data_fields) {  // for stand alone data fields
  data_fields.initialize(inside_value);
  return;
}

template <typename ExecSpace, typename MemSpace, typename T, class ...Ts>
typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_initialize(ExecutionObject<ExecSpace, MemSpace>& execObj,
                    const T& initializationValue,  Ts & ... inputs) {
  Alias<int[]>{( //first part of magic unpacker
             //inputs.initialize (inside_value)
      legacy_initialize(initializationValue, inputs)
      ,0)...,0}; //second part of magic unpacker
}

// --------------------------------------  Other loops that should get cleaned up ------------------------------

template <typename Functor>
void serial_for( BlockRange const & r, const Functor & functor )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    functor(i,j,k);
  }}}
}

template <typename Functor, typename Option>
void parallel_for( BlockRange const & r, const Functor & f, const Option& op )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(op,i,j,k);
  }}}
};

#if defined( UINTAH_ENABLE_KOKKOS )
template <typename ExecSpace, typename MemSpace, typename Functor>
typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
             Kokkos::View<int_3*, Kokkos::HostSpace> iterSpace ,const unsigned int list_size , const Functor & functor )
{
  for (unsigned int iblock=0; iblock<list_size; ++iblock) {
    functor(iterSpace[iblock][0],iterSpace[iblock][1],iterSpace[iblock][2]);
  };
}
#else
template <typename ExecSpace, typename MemSpace, typename Functor>
typename std::enable_if<std::is_same<ExecSpace, UintahSpaces::CPU>::value, void>::type
parallel_for_unstructured(ExecutionObject<ExecSpace, MemSpace>& execObj,
             std::vector<int_3> &iterSpace ,const unsigned int list_size , const Functor & functor )
{
  for (unsigned int iblock=0; iblock<list_size; ++iblock) {
    functor(iterSpace[iblock][0],iterSpace[iblock][1],iterSpace[iblock][2]);
  };
}
#endif

#if defined( UINTAH_ENABLE_KOKKOS )

//This FunctorBuilder exists because I couldn't go the lambda approach.
//I was running into some conflict with Uintah/nvcc_wrapper/Kokkos/CUDA somewhere.
//So I went the alternative route and built a functor instead of building a lambda.
#if defined( HAVE_CUDA )
template < typename Functor, typename ReductionType >
struct FunctorBuilderReduce {
  //Functor& f = nullptr;

  //std::function is probably a wrong idea, CUDA doesn't support these.
  //std::function<void(int i, int j, int k, ReductionType & red)> f;
  int ib{0};
  int ie{0};
  int jb{0};
  int je{0};

  FunctorBuilderReduce(const BlockRange & r, const Functor & f) {
    ib = r.begin(0);
    ie = r.end(0);
    jb = r.begin(1);
    je = r.end(1);
  }
  void operator()(int k,  ReductionType & red) const {
    //const int ib = r.begin(0); const int ie = r.end(0);
    //const int jb = r.begin(1); const int je = r.end(1);

    for (int j=jb; j<je; ++j) {
      for (int i=ib; i<ie; ++i) {
        f(i,j,k,red);
      }
    }
  }
};
#endif //defined(HAVE_CUDA)
#endif //if defined( UINTAH_ENABLE_KOKKOS )

} // namespace Uintah

#endif // UINTAH_HOMEBREW_LOOP_EXECUTION_HPP
