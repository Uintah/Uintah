#ifndef CORE_PARALLEL_UINTAHMPI_H
#define CORE_PARALLEL_UINTAHMPI_H

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

#include <sci_defs/mpi_defs.h>

#include <mpi.h>

#include <cstdio> // printf
#include <chrono>
#include <limits>
#include <cstdint>
#include <vector>
#include <atomic>

#ifndef NDEBUG
#  include <stdexcept> // std::runtime_error
#endif

namespace Uintah { 
namespace MPI {

//------------------------------------------------------------------------------

namespace Impl {

inline int psize( MPI_Comm c ) {
  int s;
  MPI_Comm_size( c, &s );
  return s;
}
inline int prank( MPI_Comm c ) {
  int r;
  MPI_Comm_rank( c, &r );
  return r;
}
inline int num_neighbors( MPI_Comm c ) {
  int r = prank(c);
  int n = MPI_Graph_neighbors_count(c, r, &n);
  return n;
}

//------------------------------------------------------------------------------

inline int mpi_check_err( const int err )
{
  if (err != MPI_SUCCESS) {
    char err_str[MPI_MAX_ERROR_STRING] = {};
    int err_len = 0;
    MPI_Error_string(err, err_str, &err_len);
    printf( "ERROR: MPI call returned with error: %s (err code %i)\n", err_str, err );
#ifndef NDEBUG
    throw std::runtime_error(err_str);
#endif
  }
  return err;
}

//------------------------------------------------------------------------------

inline int tid()
{
  static std::atomic<int> count{0};
  const static thread_local int id = count.fetch_add( 1, std::memory_order_relaxed );
  return id;
}
// simple RAII timer
template <typename Tag>
struct Timer
{
  using clock_type  = std::chrono::high_resolution_clock;
  using time_point  = clock_type::time_point;

  static constexpr int64_t one = 1;
  static constexpr int64_t zero = 0;

  Timer() = default;

  ~Timer()
  {
    const int t = tid() % MPI_MAX_THREADS;
    int64_t tmp = std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - m_start).count();
    s_total[t] += (tmp > 0) ? tmp : 0;
    ++s_count[t];
    s_used[t] = true;
  }

  // NOT thread safe
  static void reset_tag()
  {
    for (int i=0; i<MPI_MAX_THREADS; ++i) {
      s_total[i] = 0;
      s_count[i] = 0;
      s_used[i] = false;
    }
  }

  // total time among all threads since last reset
  static int64_t total()
  {
    int64_t result = 0;
    for (int i=0; i<MPI_MAX_THREADS; ++i) {
      result += (s_used[i]) ? s_total[i] : 0 ;
    }
    return result;
  }

  // min time among all threads since last reset
  static int64_t min()
  {
    int64_t result = std::numeric_limits<int64_t>::max();
    for (int i=0; i<MPI_MAX_THREADS; ++i) {
      result = (s_used[i] && s_total[i] < result) ? s_total[i] : result ;
    }
    return result != std::numeric_limits<int64_t>::max() ? result : 0 ;
  }

  // max time among all threads since last reset
  static int64_t max()
  {
    int64_t result = std::numeric_limits<int64_t>::min();
    for (int i=0; i<MPI_MAX_THREADS; ++i) {
      result = (s_used[i] && result < s_total[i]) ? s_total[i] : result ;
    }
    return result != std::numeric_limits<int64_t>::min() ? result : 0 ;
  }

  // time on given thread since last reset
  static int64_t thread(int t)  { return s_total[ t % MPI_MAX_THREADS ]; }

  // number of threads that used the timer since last reset
  static int num_threads()
  {
    int result = 0;
    for (int i=0; i<MPI_MAX_THREADS; ++i) {
      result += s_used[i] ? 1 : 0;
    }
    return result;
  }

  static int64_t count()
  {
    int64_t result = 0;
    for (int i=0; i<MPI_MAX_THREADS; ++i) {
      result = (s_used[i] && result < s_count[i]) ? s_count[i] : result ;
    }
    return result;
  }

private:
  time_point m_start {clock_type::now()};

  static int64_t s_total[MPI_MAX_THREADS];
  static int64_t s_count[MPI_MAX_THREADS];
  static bool     s_used[MPI_MAX_THREADS];

  // disable copy, assignment
  Timer( const Timer & )             = delete;
  Timer( Timer && )                  = delete;
  Timer & operator=( const Timer & ) = delete;
  Timer & operator=( Timer && )      = delete;
};

template <typename Tag> int64_t  Timer<Tag>::s_total[MPI_MAX_THREADS] = {};
template <typename Tag> int64_t  Timer<Tag>::s_count[MPI_MAX_THREADS] = {};
template <typename Tag> bool     Timer<Tag>::s_used[ MPI_MAX_THREADS] = {};

//------------------------------------------------------------------------------

enum CommStatsType {
      COMM_COUNT
    , COMM_SIZE
    , COMM_HISTOGRAM_0
    , COMM_HISTOGRAM_1
    , COMM_HISTOGRAM_2
    , COMM_HISTOGRAM_3
    , SIZE
};

template < typename Tag >
struct CommStats
{
  static constexpr int64_t one  = 1;
  static constexpr int64_t zero = 0;

  static constexpr int64_t Cacheline = 64;       // cacheline
  static constexpr int64_t Page =      4096;     // page
  static constexpr int64_t HugePage =  2 << 20;  // huge page

  CommStats( int64_t count, MPI_Datatype d ) noexcept
  {
    apply(count, d);
  }

  CommStats( const int * counts, MPI_Datatype d, MPI_Comm c ) noexcept
  {
    apply(counts, d, c);
  }

  static void clear()
  {
    for (auto & v : s_values ) {
      v = 0;
    }
  }

  static int64_t get(CommStatsType i)
  {
    int64_t result = 0;
    const int begin = i * MPI_MAX_THREADS;
    const int end = (i+1) * MPI_MAX_THREADS;
    for (int j=begin; j<end; ++j) {
      result += s_values[j];
    }
    return result;
  }

private:
  static void apply( int64_t count, MPI_Datatype d ) noexcept
  {
    int s;
    MPI_Type_size( d, &s );

    const int64_t n = count * s;

    const int i = n <= Cacheline ? 0 :
                  n <= Page ? 1 :
                  n <= HugePage ? 2 : 3;

    const int t = tid();


    ++s_values[COMM_COUNT*MPI_MAX_THREADS + t];
    ++s_values[(COMM_HISTOGRAM_0 + i)*MPI_MAX_THREADS + t];
    s_values[COMM_SIZE*MPI_MAX_THREADS + t] += n;
  }

  static void apply( const int * counts, MPI_Datatype d, MPI_Comm c ) noexcept
  {
    int64_t count = 0;
    const int parallel_size = psize( c );
    for (int i=0; i<parallel_size; ++i) {
      count += counts[i];
    }

    apply( count, d );
  }

  static int64_t s_values[SIZE * MPI_MAX_THREADS];

};

template<typename Tag>
int64_t CommStats<Tag>::s_values[SIZE * MPI_MAX_THREADS]{};

//------------------------------------------------------------------------------

struct AlltoAllTag {};
struct BcastTag    {};
struct GatherTag   {};
struct ReduceTag   {};
struct ScanTag     {};
struct ScatterTag  {};
struct RecvTag     {};
struct SendTag     {};
struct TestTag     {};
struct WaitTag     {};
struct CommTag     {};
struct OneSidedTag {};

using AlltoallTimer = Timer<AlltoAllTag>;
using BcastTimer    = Timer<BcastTag>;
using GatherTimer   = Timer<GatherTag>;
using ReduceTimer   = Timer<ReduceTag>;
using ScanTimer     = Timer<ScanTag>;
using ScatterTimer  = Timer<ScatterTag>;
using CommTimer     = Timer<CommTag>;
using OneSidedTimer = Timer<OneSidedTag>;
using RecvTimer     = Timer<RecvTag>;
using SendTimer     = Timer<SendTag>;
using TestTimer     = Timer<TestTag>;
using WaitTimer     = Timer<WaitTag>;

using OneVolumeStats =  CommStats<OneSidedTag>;
using SendVolumeStats = CommStats<SendTag>;
using RecvVolumeStats = CommStats<RecvTag>;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
} // end namespace Impl


#if UINTAH_ENABLE_MPI3
  inline MPI_Aint Aint_add( MPI_Aint base , MPI_Aint disp )    { return MPI_Aint_add( base, disp ); }
  inline MPI_Aint Aint_diff( MPI_Aint addr1 , MPI_Aint addr2 ) { return MPI_Aint_diff( addr1, addr2 ); }
#endif

inline double Wtick( void ) { return MPI_Wtick(); }
inline double Wtime( void ) { return MPI_Wtime(); }

inline int Abort( MPI_Comm comm , int errorcode )
{
  return Impl::mpi_check_err(MPI_Abort( comm , errorcode ));
}
inline int Accumulate( MPICONST void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp,
                       int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err( MPI_Accumulate( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
                                              target_datatype, op, win ) );
}
inline int Add_error_class( int *errorclass )
{
  return Impl::mpi_check_err(MPI_Add_error_class( errorclass ));
}
inline int Add_error_code( int errorclass , int *errorcode )
{
  return Impl::mpi_check_err(MPI_Add_error_code( errorclass , errorcode ));
}
inline int Add_error_string( int errorcode , MPICONST char *string )
{
  return Impl::mpi_check_err(MPI_Add_error_string( errorcode , string ));
}
inline int Allgather( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcount, recvtype );
  return Impl::mpi_check_err(MPI_Allgather( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm ));
}
inline int Allgatherv( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , MPICONST int *recvcounts , MPICONST int *displs , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcounts, recvtype, comm );
  return Impl::mpi_check_err(MPI_Allgatherv( sendbuf , sendcount , sendtype , recvbuf , recvcounts , displs , recvtype , comm ));
}
inline int Alloc_mem( MPI_Aint size , MPI_Info info , void *baseptr )
{
  return Impl::mpi_check_err(MPI_Alloc_mem( size , info , baseptr ));
}
inline int Allreduce( MPICONST void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm )
{
  Impl::ReduceTimer timer;
  Impl::SendVolumeStats( count, datatype );
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Allreduce( sendbuf , recvbuf , count , datatype , op , comm ));
}
inline int Alltoall( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::AlltoallTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcount, recvtype );
  return Impl::mpi_check_err(MPI_Alltoall( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm ));
}
inline int Alltoallv( MPICONST void *sendbuf , MPICONST int *sendcounts , MPICONST int *sdispls , MPI_Datatype sendtype , void *recvbuf , MPICONST int *recvcounts , MPICONST int *rdispls , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::AlltoallTimer timer;
  Impl::SendVolumeStats( sendcounts, sendtype, comm );
  Impl::RecvVolumeStats( recvcounts, recvtype, comm );
  return Impl::mpi_check_err(MPI_Alltoallv( sendbuf , sendcounts , sdispls , sendtype , recvbuf , recvcounts , rdispls , recvtype , comm ));
}
inline int Alltoallw( MPICONST void *sendbuf , MPICONST int sendcounts[] , MPICONST int sdispls[] , MPICONST MPI_Datatype sendtypes[] , void *recvbuf , MPICONST int recvcounts[] , MPICONST int rdispls[] , MPICONST MPI_Datatype recvtypes[] , MPI_Comm comm )
{
  // TODO handle multiple send and recv types
  Impl::AlltoallTimer timer;
  const int psize = Impl::psize(comm);
  int64_t rcounts = 0;
  int64_t scounts = 0;
  for (int i=0; i<psize; ++i) {
    scounts += sendcounts[i];
    rcounts += recvcounts[i];
  }
  Impl::SendVolumeStats( scounts, sendtypes[0] );
  Impl::RecvVolumeStats( rcounts, recvtypes[0] );
  return Impl::mpi_check_err(MPI_Alltoallw( sendbuf , sendcounts , sdispls , sendtypes , recvbuf , recvcounts , rdispls , recvtypes , comm ));
}
inline int Barrier( MPI_Comm comm )
{
  Impl::WaitTimer timer;
  return Impl::mpi_check_err(MPI_Barrier( comm ));
}
inline int Bcast( void *buffer , int count , MPI_Datatype datatype , int root , MPI_Comm comm )
{
  Impl::BcastTimer timer;
  if (Impl::prank(comm) == root) {
    Impl::SendVolumeStats( count, datatype );
  }
  else {
    Impl::RecvVolumeStats( count, datatype );
  }
  return Impl::mpi_check_err(MPI_Bcast( buffer , count , datatype , root , comm ));
}
inline int Bsend( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Bsend( buf , count , datatype , dest , tag , comm ));
}
inline int Bsend_init( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  Impl::SendTimer timer;
  return Impl::mpi_check_err(MPI_Bsend_init( buf , count , datatype , dest , tag , comm , request ));
}
inline int Buffer_attach( void *buffer , int size )
{
  return Impl::mpi_check_err(MPI_Buffer_attach( buffer , size ));
}
inline int Buffer_detach( void *buffer_addr , int *size )
{
  return Impl::mpi_check_err(MPI_Buffer_detach( buffer_addr , size ));
}
inline int Cancel( MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Cancel( request ));
}
inline int Cart_coords( MPI_Comm comm , int rank , int maxdims , int coords[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cart_coords( comm , rank , maxdims , coords ));
}
inline int Cart_create( MPI_Comm comm_old , int ndims , MPICONST int dims[] , MPICONST int periods[] , int reorder , MPI_Comm *comm_cart )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cart_create( comm_old , ndims , dims , periods , reorder , comm_cart ));
}
inline int Cart_get( MPI_Comm comm , int maxdims , int dims[] , int periods[] , int coords[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cart_get( comm , maxdims , dims , periods , coords ));
}
inline int Cart_map( MPI_Comm comm , int ndims , MPICONST int dims[] , MPICONST int periods[] , int *newrank )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cart_map( comm , ndims , dims , periods , newrank ));
}
inline int Cart_rank( MPI_Comm comm , MPICONST int coords[] , int *rank )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cart_rank( comm , coords , rank ));
}
inline int Cart_shift( MPI_Comm comm , int direction , int disp , int *rank_source , int *rank_dest )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cart_shift( comm , direction , disp , rank_source , rank_dest ));
}
inline int Cart_sub( MPI_Comm comm , MPICONST int remain_dims[] , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cart_sub( comm , remain_dims , newcomm ));
}
inline int Cartdim_get( MPI_Comm comm , int *ndims )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Cartdim_get( comm , ndims ));
}
inline int Close_port( MPICONST char *port_name )
{
  return Impl::mpi_check_err(MPI_Close_port( port_name ));
}
inline int Comm_accept( MPICONST char *port_name , MPI_Info info , int root , MPI_Comm comm , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_accept( port_name , info , root , comm , newcomm ));
}
inline int Comm_call_errhandler( MPI_Comm comm , int errorcode )
{
  return Impl::mpi_check_err(MPI_Comm_call_errhandler( comm , errorcode ));
}
inline int Comm_compare( MPI_Comm comm1 , MPI_Comm comm2 , int *result )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_compare( comm1 , comm2 , result ));
}
inline int Comm_connect( MPICONST char *port_name , MPI_Info info , int root , MPI_Comm comm , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_connect( port_name , info , root , comm , newcomm ));
}
inline int Comm_create( MPI_Comm comm , MPI_Group group , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_create( comm , group , newcomm ));
}
inline int Comm_create_errhandler( MPI_Comm_errhandler_function *comm_errhandler_fn , MPI_Errhandler *errhandler )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_create_errhandler( comm_errhandler_fn , errhandler ));
}
#if UINTAH_ENABLE_MPI3
inline int Comm_create_group( MPI_Comm comm , MPI_Group group , int tag , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_create_group( comm , group , tag , newcomm ));
}
#endif
inline int Comm_create_keyval( MPI_Comm_copy_attr_function *comm_copy_attr_fn , MPI_Comm_delete_attr_function *comm_delete_attr_fn , int *comm_keyval , void *extra_state )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_create_keyval( comm_copy_attr_fn , comm_delete_attr_fn , comm_keyval , extra_state ));
}
inline int Comm_delete_attr( MPI_Comm comm , int comm_keyval )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_delete_attr( comm , comm_keyval ));
}
inline int Comm_disconnect( MPI_Comm *comm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_disconnect( comm ));
}
inline int Comm_dup( MPI_Comm comm , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_dup( comm , newcomm ));
}
#if UINTAH_ENABLE_MPI3
inline int Comm_dup_with_info( MPI_Comm comm , MPI_Info info , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_dup_with_info( comm , info , newcomm ));
}
#endif
inline int Comm_free( MPI_Comm *comm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_free( comm ));
}
inline int Comm_free_keyval( int *comm_keyval )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_free_keyval( comm_keyval ));
}
inline int Comm_get_attr( MPI_Comm comm , int comm_keyval , void *attribute_val , int *flag )
{
  return Impl::mpi_check_err(MPI_Comm_get_attr( comm , comm_keyval , attribute_val , flag ));
}
inline int Comm_get_errhandler( MPI_Comm comm , MPI_Errhandler *errhandler )
{
  return Impl::mpi_check_err(MPI_Comm_get_errhandler( comm , errhandler ));
}
#if UINTAH_ENABLE_MPI3
inline int Comm_get_info( MPI_Comm comm , MPI_Info *info )
{
  return Impl::mpi_check_err(MPI_Comm_get_info( comm , info ));
}
#endif
inline int Comm_get_name( MPI_Comm comm , char *comm_name , int *resultlen )
{
  return Impl::mpi_check_err(MPI_Comm_get_name( comm , comm_name , resultlen ));
}
inline int Comm_get_parent( MPI_Comm *parent )
{
  return Impl::mpi_check_err(MPI_Comm_get_parent( parent ));
}
inline int Comm_group( MPI_Comm comm , MPI_Group *group )
{
  return Impl::mpi_check_err(MPI_Comm_group( comm , group ));
}
#if UINTAH_ENABLE_MPI3
inline int Comm_idup( MPI_Comm comm , MPI_Comm *newcomm , MPI_Request *request )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_idup( comm , newcomm , request ));
}
#endif
inline int Comm_join( int fd , MPI_Comm *intercomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_join( fd , intercomm ));
}
inline int Comm_rank( MPI_Comm comm , int *rank )
{
  return Impl::mpi_check_err(MPI_Comm_rank( comm , rank ));
}
inline int Comm_remote_group( MPI_Comm comm , MPI_Group *group )
{
  return Impl::mpi_check_err(MPI_Comm_remote_group( comm , group ));
}
inline int Comm_remote_size( MPI_Comm comm , int *size )
{
  return Impl::mpi_check_err(MPI_Comm_remote_size( comm , size ));
}
inline int Comm_set_attr( MPI_Comm comm , int comm_keyval , void *attribute_val )
{
  return Impl::mpi_check_err(MPI_Comm_set_attr( comm , comm_keyval , attribute_val ));
}
inline int Comm_set_errhandler( MPI_Comm comm , MPI_Errhandler errhandler )
{
  return Impl::mpi_check_err(MPI_Comm_set_errhandler( comm , errhandler ));
}
#if UINTAH_ENABLE_MPI3
inline int Comm_set_info( MPI_Comm comm , MPI_Info info )
{
  return Impl::mpi_check_err(MPI_Comm_set_info( comm , info ));
}
#endif
inline int Comm_set_name( MPI_Comm comm , MPICONST char *comm_name )
{
  return Impl::mpi_check_err(MPI_Comm_set_name( comm , comm_name ));
}
inline int Comm_size( MPI_Comm comm , int *size )
{
  return Impl::mpi_check_err(MPI_Comm_size( comm , size ));
}
inline int Comm_spawn( MPICONST char *command , char *argv[] , int maxprocs , MPI_Info info , int root , MPI_Comm comm , MPI_Comm *intercomm , int array_of_errcodes[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_spawn( command , argv , maxprocs , info , root , comm , intercomm , array_of_errcodes ));
}
inline int Comm_spawn_multiple( int count , char *array_of_commands[] , char **array_of_argv[] , MPICONST int array_of_maxprocs[] , MPICONST MPI_Info array_of_info[] , int root , MPI_Comm comm , MPI_Comm *intercomm , int array_of_errcodes[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_spawn_multiple( count , array_of_commands , array_of_argv , array_of_maxprocs , array_of_info , root , comm , intercomm , array_of_errcodes ));
}
inline int Comm_split( MPI_Comm comm , int color , int key , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_split( comm , color , key , newcomm ));
}
#if UINTAH_ENABLE_MPI3
inline int Comm_split_type( MPI_Comm comm , int split_type , int key , MPI_Info info , MPI_Comm *newcomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_split_type( comm , split_type , key , info , newcomm ));
}
#endif
inline int Comm_test_inter( MPI_Comm comm , int *flag )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Comm_test_inter( comm , flag ));
}
#if UINTAH_ENABLE_MPI3
inline int Compare_and_swap( const void *origin_addr , const void *compare_addr , void *result_addr , MPI_Datatype datatype , int target_rank , MPI_Aint target_disp , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( 1, datatype);
  return Impl::mpi_check_err(MPI_Compare_and_swap( origin_addr , compare_addr , result_addr , datatype , target_rank , target_disp , win ));
}
#endif
inline int Dims_create( int nnodes , int ndims , int dims[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Dims_create( nnodes , ndims , dims ));
}
#if UINTAH_ENABLE_MPI3
inline int Dist_graph_create( MPI_Comm comm_old , int n , MPICONST int sources[] , MPICONST int degrees[] , MPICONST int destinations[] , MPICONST int weights[] , MPI_Info info , int reorder , MPI_Comm *comm_dist_graph )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Dist_graph_create( comm_old , n , sources , degrees , destinations , weights , info , reorder , comm_dist_graph ));
}
#endif
#if UINTAH_ENABLE_MPI3
inline int Dist_graph_create_adjacent( MPI_Comm comm_old , int indegree , const int sources[] , const int sourceweights[] , int outdegree , const int destinations[] , const int destweights[] , MPI_Info info , int reorder , MPI_Comm *comm_dist_graph )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Dist_graph_create_adjacent( comm_old , indegree , sources , sourceweights , outdegree , destinations , destweights , info , reorder , comm_dist_graph ));
}
inline int Dist_graph_neighbors( MPI_Comm comm , int maxindegree , int sources[] , int sourceweights[] , int maxoutdegree , int destinations[] , int destweights[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Dist_graph_neighbors( comm , maxindegree , sources , sourceweights , maxoutdegree , destinations , destweights ));
}
inline int Dist_graph_neighbors_count( MPI_Comm comm , int *indegree , int *outdegree , int *weighted )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Dist_graph_neighbors_count( comm , indegree , outdegree , weighted ));
}
#endif
inline int Errhandler_free( MPI_Errhandler *errhandler )
{
  return Impl::mpi_check_err(MPI_Errhandler_free( errhandler ));
}
inline int Error_class( int errorcode , int *errorclass )
{
  return Impl::mpi_check_err(MPI_Error_class( errorcode , errorclass ));
}
inline int Error_string( int errorcode , char *string , int *resultlen )
{
  return Impl::mpi_check_err(MPI_Error_string( errorcode , string , resultlen ));
}
inline int Exscan( MPICONST void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm )
{
  Impl::ScanTimer timer;
  Impl::SendVolumeStats( count, datatype );
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Exscan( sendbuf , recvbuf , count , datatype , op , comm ));
}
#if UINTAH_ENABLE_MPI3
inline int Fetch_and_op( const void *origin_addr , void *result_addr , MPI_Datatype datatype , int target_rank , MPI_Aint target_disp , MPI_Op op , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( 1, datatype);
  return Impl::mpi_check_err(MPI_Fetch_and_op( origin_addr , result_addr , datatype , target_rank , target_disp , op , win ));
}
#endif
inline int File_call_errhandler( MPI_File fh , int errorcode )
{
  return Impl::mpi_check_err(MPI_File_call_errhandler( fh , errorcode ));
}
inline int File_create_errhandler( MPI_File_errhandler_function *file_errhandler_fn , MPI_Errhandler *errhandler )
{
  return Impl::mpi_check_err(MPI_File_create_errhandler( file_errhandler_fn , errhandler ));
}
inline int File_get_errhandler( MPI_File file , MPI_Errhandler *errhandler )
{
  return Impl::mpi_check_err(MPI_File_get_errhandler( file , errhandler ));
}
inline int File_set_errhandler( MPI_File file , MPI_Errhandler errhandler )
{
  return Impl::mpi_check_err(MPI_File_set_errhandler( file , errhandler ));
}
inline int Finalize( void )
{
  return Impl::mpi_check_err(MPI_Finalize( ));
}
inline int Finalized( int *flag )
{
  return Impl::mpi_check_err(MPI_Finalized( flag ));
}
inline int Free_mem( void *base )
{
  return Impl::mpi_check_err(MPI_Free_mem( base ));
}
inline int Gather( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  if ( root == Impl::prank(comm) ) {
    Impl::RecvVolumeStats( recvcount, recvtype );
  }
  return Impl::mpi_check_err(MPI_Gather( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , root , comm ));
}
inline int Gatherv( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , MPICONST int *recvcounts , MPICONST int *displs , MPI_Datatype recvtype , int root , MPI_Comm comm )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  if ( root == Impl::prank(comm) ) {
    Impl::RecvVolumeStats(recvcounts, recvtype, comm );
  }
  return Impl::mpi_check_err(MPI_Gatherv( sendbuf , sendcount , sendtype , recvbuf , recvcounts , displs , recvtype , root , comm ));
}
inline int Get( void *origin_addr , int origin_count , MPI_Datatype origin_datatype , int target_rank , MPI_Aint target_disp , int target_count , MPI_Datatype target_datatype , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err(MPI_Get( origin_addr , origin_count , origin_datatype , target_rank , target_disp , target_count , target_datatype , win ));
}
#if UINTAH_ENABLE_MPI3
inline int Get_accumulate( const void *origin_addr , int origin_count , MPI_Datatype origin_datatype , void *result_addr , int result_count , MPI_Datatype result_datatype , int target_rank , MPI_Aint target_disp , int target_count , MPI_Datatype target_datatype , MPI_Op op , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err(MPI_Get_accumulate( origin_addr , origin_count , origin_datatype , result_addr , result_count , result_datatype , target_rank , target_disp , target_count , target_datatype , op , win ));
}
#endif
inline int Get_address( MPICONST void *location , MPI_Aint *address )
{
  return Impl::mpi_check_err(MPI_Get_address( location , address ));
}
inline int Get_count( MPICONST MPI_Status *status , MPI_Datatype datatype , int *count )
{
  return Impl::mpi_check_err(MPI_Get_count( status , datatype , count ));
}
inline int Get_elements( MPICONST MPI_Status *status , MPI_Datatype datatype , int *count )
{
  return Impl::mpi_check_err(MPI_Get_elements( status , datatype , count ));
}
#if UINTAH_ENABLE_MPI3
inline int Get_elements_x( const MPI_Status *status , MPI_Datatype datatype , MPI_Count *count )
{
  return Impl::mpi_check_err(MPI_Get_elements_x( status , datatype , count ));
}
inline int Get_library_version( char *version , int *resultlen )
{
  return Impl::mpi_check_err(MPI_Get_library_version( version , resultlen ));
}
#endif
inline int Get_processor_name( char *name , int *resultlen )
{
  return Impl::mpi_check_err(MPI_Get_processor_name( name , resultlen ));
}
inline int Get_version( int *version , int *subversion )
{
  return Impl::mpi_check_err(MPI_Get_version( version , subversion ));
}
inline int Graph_create( MPI_Comm comm_old , int nnodes , MPICONST int indx[] , MPICONST int edges[] , int reorder , MPI_Comm *comm_graph )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Graph_create( comm_old , nnodes , indx , edges , reorder , comm_graph ));
}
inline int Graph_get( MPI_Comm comm , int maxindex , int maxedges , int indx[] , int edges[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Graph_get( comm , maxindex , maxedges , indx , edges ));
}
inline int Graph_map( MPI_Comm comm , int nnodes , MPICONST int indx[] , MPICONST int edges[] , int *newrank )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Graph_map( comm , nnodes , indx , edges , newrank ));
}
inline int Graph_neighbors( MPI_Comm comm , int rank , int maxneighbors , int neighbors[] )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Graph_neighbors( comm , rank , maxneighbors , neighbors ));
}
inline int Graph_neighbors_count( MPI_Comm comm , int rank , int *nneighbors )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Graph_neighbors_count( comm , rank , nneighbors ));
}
inline int Graphdims_get( MPI_Comm comm , int *nnodes , int *nedges )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Graphdims_get( comm , nnodes , nedges ));
}
inline int Grequest_complete( MPI_Request request )
{
  return Impl::mpi_check_err(MPI_Grequest_complete( request ));
}
inline int Grequest_start( MPI_Grequest_query_function *query_fn , MPI_Grequest_free_function *free_fn , MPI_Grequest_cancel_function *cancel_fn , void *extra_state , MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Grequest_start( query_fn , free_fn , cancel_fn , extra_state , request ));
}
inline int Group_compare( MPI_Group group1 , MPI_Group group2 , int *result )
{
  return Impl::mpi_check_err(MPI_Group_compare( group1 , group2 , result ));
}
inline int Group_difference( MPI_Group group1 , MPI_Group group2 , MPI_Group *newgroup )
{
  return Impl::mpi_check_err(MPI_Group_difference( group1 , group2 , newgroup ));
}
inline int Group_excl( MPI_Group group , int n , MPICONST int ranks[] , MPI_Group *newgroup )
{
  return Impl::mpi_check_err(MPI_Group_excl( group , n , ranks , newgroup ));
}
inline int Group_free( MPI_Group *group )
{
  return Impl::mpi_check_err(MPI_Group_free( group ));
}
inline int Group_incl( MPI_Group group , int n , MPICONST int ranks[] , MPI_Group *newgroup )
{
  return Impl::mpi_check_err(MPI_Group_incl( group , n , ranks , newgroup ));
}
inline int Group_intersection( MPI_Group group1 , MPI_Group group2 , MPI_Group *newgroup )
{
  return Impl::mpi_check_err(MPI_Group_intersection( group1 , group2 , newgroup ));
}
inline int Group_range_excl( MPI_Group group , int n , int ranges[][3] , MPI_Group *newgroup )
{
  return Impl::mpi_check_err(MPI_Group_range_excl( group , n , ranges , newgroup ));
}
inline int Group_range_incl( MPI_Group group , int n , int ranges[][3] , MPI_Group *newgroup )
{
  return Impl::mpi_check_err(MPI_Group_range_incl( group , n , ranges , newgroup ));
}
inline int Group_rank( MPI_Group group , int *rank )
{
  return Impl::mpi_check_err(MPI_Group_rank( group , rank ));
}
inline int Group_size( MPI_Group group , int *size )
{
  return Impl::mpi_check_err(MPI_Group_size( group , size ));
}
inline int Group_translate_ranks( MPI_Group group1 , int n , MPICONST int ranks1[] , MPI_Group group2 , int ranks2[] )
{
  return Impl::mpi_check_err(MPI_Group_translate_ranks( group1 , n , ranks1 , group2 , ranks2 ));
}
inline int Group_union( MPI_Group group1 , MPI_Group group2 , MPI_Group *newgroup )
{
  return Impl::mpi_check_err(MPI_Group_union( group1 , group2 , newgroup ));
}
#if UINTAH_ENABLE_MPI3
inline int Iallgather( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcount, recvtype );
  return Impl::mpi_check_err(MPI_Iallgather( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm , request ));
}
inline int Iallgatherv( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , const int recvcounts[] , const int displs[] , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcounts, recvtype, comm );
  return Impl::mpi_check_err(MPI_Iallgatherv( sendbuf , sendcount , sendtype , recvbuf , recvcounts , displs , recvtype , comm , request ));
}
inline int Iallreduce( const void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm , MPI_Request *request )
{
  Impl::ReduceTimer timer;
  Impl::SendVolumeStats( count, datatype );
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Iallreduce( sendbuf , recvbuf , count , datatype , op , comm , request ));
}
inline int Ialltoall( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::AlltoallTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcount, recvtype );
  return Impl::mpi_check_err(MPI_Ialltoall( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm , request ));
}
inline int Ialltoallv( const void *sendbuf , const int sendcounts[] , const int sdispls[] , MPI_Datatype sendtype , void *recvbuf , const int recvcounts[] , const int rdispls[] , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::AlltoallTimer timer;
  Impl::SendVolumeStats( sendcounts, sendtype, comm );
  Impl::RecvVolumeStats( recvcounts, recvtype, comm );
  return Impl::mpi_check_err(MPI_Ialltoallv( sendbuf , sendcounts , sdispls , sendtype , recvbuf , recvcounts , rdispls , recvtype , comm , request ));
}
inline int Ialltoallw( MPICONST void *sendbuf , const int sendcounts[] , const int sdispls[] , const MPI_Datatype sendtypes[] , void *recvbuf , const int recvcounts[] , const int rdispls[] , const MPI_Datatype recvtypes[] , MPI_Comm comm , MPI_Request *request )
{
  // TODO handle multiple send and recv types
  Impl::AlltoallTimer timer;
  const int psize = Impl::psize(comm);
  int64_t rcounts = 0;
  int64_t scounts = 0;
  for (int i=0; i<psize; ++i) {
    scounts += sendcounts[i];
    rcounts += recvcounts[i];
  }
  Impl::SendVolumeStats( scounts, sendtypes[0] );
  Impl::RecvVolumeStats( rcounts, recvtypes[0] );
  return Impl::mpi_check_err(MPI_Ialltoallw( sendbuf , sendcounts , sdispls , sendtypes , recvbuf , recvcounts , rdispls , recvtypes , comm , request ));
}
inline int Ibarrier( MPI_Comm comm , MPI_Request *request )
{
  Impl::WaitTimer timer;
  return Impl::mpi_check_err(MPI_Ibarrier( comm , request ));
}
inline int Ibcast( void *buffer , int count , MPI_Datatype datatype , int root , MPI_Comm comm , MPI_Request *request )
{
  Impl::BcastTimer timer;
  if (Impl::prank(comm) == root) {
    Impl::SendVolumeStats( count, datatype );
  }
  else {
    Impl::RecvVolumeStats( count, datatype );
  }
  return Impl::mpi_check_err(MPI_Ibcast( buffer , count , datatype , root , comm , request ));
}
#endif
inline int Ibsend( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Ibsend( buf , count , datatype , dest , tag , comm , request ));
}
#if UINTAH_ENABLE_MPI3
inline int Iexscan( const void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm , MPI_Request *request )
{
  Impl::ScanTimer timer;
  Impl::SendVolumeStats( count, datatype );
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Iexscan( sendbuf , recvbuf , count , datatype , op , comm , request ));
}
inline int Igather( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm , MPI_Request *request )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  if ( root == Impl::prank(comm) ) {
    Impl::RecvVolumeStats( recvcount, recvtype );
  }
  return Impl::mpi_check_err(MPI_Igather( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , root , comm , request ));
}
inline int Igatherv( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , const int recvcounts[] , const int displs[] , MPI_Datatype recvtype , int root , MPI_Comm comm , MPI_Request *request )
{
  Impl::GatherTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  if ( root == Impl::prank(comm) ) {
    Impl::RecvVolumeStats( recvcounts, recvtype, comm );
  }
  return Impl::mpi_check_err(MPI_Igatherv( sendbuf , sendcount , sendtype , recvbuf , recvcounts , displs , recvtype , root , comm , request ));
}
inline int Improbe( int source , int tag , MPI_Comm comm , int *flag , MPI_Message *message , MPI_Status *status )
{
  return Impl::mpi_check_err(MPI_Improbe( source , tag , comm , flag , message , status ));
}
inline int Imrecv( void *buf , int count , MPI_Datatype datatype , MPI_Message *message , MPI_Request *request )
{
  Impl::RecvTimer timer;
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Imrecv( buf , count , datatype , message , request ));
}
inline int Ineighbor_allgather( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::GatherTimer timer;
  Impl::OneVolumeStats( sendcount, sendtype );
  return Impl::mpi_check_err(MPI_Ineighbor_allgather( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm , request ));
}
inline int Ineighbor_allgatherv( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , const int recvcounts[] , const int displs[] , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::AlltoallTimer timer;
  Impl::OneVolumeStats( sendcount, sendtype );
  return Impl::mpi_check_err(MPI_Ineighbor_allgatherv( sendbuf , sendcount , sendtype , recvbuf , recvcounts , displs , recvtype , comm , request ));
}
inline int Ineighbor_alltoall( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::AlltoallTimer timer;
  Impl::OneVolumeStats( sendcount, sendtype );
  return Impl::mpi_check_err(MPI_Ineighbor_alltoall( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm , request ));
}
inline int Ineighbor_alltoallv( const void *sendbuf , const int sendcounts[] , const int sdispls[] , MPI_Datatype sendtype , void *recvbuf , const int recvcounts[] , const int rdispls[] , MPI_Datatype recvtype , MPI_Comm comm , MPI_Request *request )
{
  Impl::AlltoallTimer timer;
  const int nn = Impl::num_neighbors(comm);
  int64_t scounts = 0;
  for (int i=0; i<nn; ++i) {
    scounts += sendcounts[i];
  }
  Impl::OneVolumeStats( scounts, sendtype );
  return Impl::mpi_check_err(MPI_Ineighbor_alltoallv( sendbuf , sendcounts , sdispls , sendtype , recvbuf , recvcounts , rdispls , recvtype , comm , request ));
}
inline int Ineighbor_alltoallw( const void *sendbuf , const int sendcounts[] , const MPI_Aint sdispls[] , const MPI_Datatype sendtypes[] , void *recvbuf , const int recvcounts[] , const MPI_Aint rdispls[] , const MPI_Datatype recvtypes[] , MPI_Comm comm , MPI_Request *request )
{
  // TODO handle multiple send and recv types
  Impl::AlltoallTimer timer;
  const int psize = Impl::psize(comm);
  int64_t scounts = 0;
  for (int i=0; i<psize; ++i) {
    scounts += sendcounts[i];
  }
  Impl::OneVolumeStats( scounts, sendtypes[0] );
  return Impl::mpi_check_err(MPI_Ineighbor_alltoallw( sendbuf , sendcounts , sdispls , sendtypes , recvbuf , recvcounts , rdispls , recvtypes , comm , request ));
}
#endif
inline int Info_create( MPI_Info *info )
{
  return Impl::mpi_check_err(MPI_Info_create( info ));
}
inline int Info_delete( MPI_Info info , MPICONST char *key )
{
  return Impl::mpi_check_err(MPI_Info_delete( info , key ));
}
inline int Info_dup( MPI_Info info , MPI_Info *newinfo )
{
  return Impl::mpi_check_err(MPI_Info_dup( info , newinfo ));
}
inline int Info_free( MPI_Info *info )
{
  return Impl::mpi_check_err(MPI_Info_free( info ));
}
inline int Info_get( MPI_Info info , MPICONST char *key , int valuelen , char *value , int *flag )
{
  return Impl::mpi_check_err(MPI_Info_get( info , key , valuelen , value , flag ));
}
inline int Info_get_nkeys( MPI_Info info , int *nkeys )
{
  return Impl::mpi_check_err(MPI_Info_get_nkeys( info , nkeys ));
}
inline int Info_get_nthkey( MPI_Info info , int n , char *key )
{
  return Impl::mpi_check_err(MPI_Info_get_nthkey( info , n , key ));
}
inline int Info_get_valuelen( MPI_Info info , MPICONST char *key , int *valuelen , int *flag )
{
  return Impl::mpi_check_err(MPI_Info_get_valuelen( info , key , valuelen , flag ));
}
inline int Info_set( MPI_Info info , MPICONST char *key , MPICONST char *value )
{
  return Impl::mpi_check_err(MPI_Info_set( info , key , value ));
}
inline int Init( int *argc , char ***argv )
{
  return Impl::mpi_check_err(MPI_Init( argc , argv ));
}
inline int Init_thread( int *argc , char ***argv , int required , int *provided )
{
  return Impl::mpi_check_err(MPI_Init_thread( argc , argv , required , provided ));
}
inline int Initialized( int *flag )
{
  return Impl::mpi_check_err(MPI_Initialized( flag ));
}
inline int Intercomm_create( MPI_Comm local_comm , int local_leader , MPI_Comm peer_comm , int remote_leader , int tag , MPI_Comm *newintercomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Intercomm_create( local_comm , local_leader , peer_comm , remote_leader , tag , newintercomm ));
}
inline int Intercomm_merge( MPI_Comm intercomm , int high , MPI_Comm *newintracomm )
{
  Impl::CommTimer timer;
  return Impl::mpi_check_err(MPI_Intercomm_merge( intercomm , high , newintracomm ));
}
inline int Iprobe( int source , int tag , MPI_Comm comm , int *flag , MPI_Status *status )
{
  return Impl::mpi_check_err(MPI_Iprobe( source , tag , comm , flag , status ));
}
inline int Irecv( void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Request *request )
{
  Impl::RecvTimer timer;
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Irecv( buf , count , datatype , source , tag , comm , request ));
}
#if UINTAH_ENABLE_MPI3
inline int Ireduce( const void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , int root , MPI_Comm comm , MPI_Request *request )
{
  Impl::ReduceTimer timer;
  Impl::SendVolumeStats( count, datatype );
  if ( root == Impl::prank(comm) ) {
    Impl::RecvVolumeStats( count, datatype );
  }
  return Impl::mpi_check_err(MPI_Ireduce( sendbuf , recvbuf , count , datatype , op , root , comm , request ));
}
inline int Ireduce_scatter( MPICONST void *sendbuf , void *recvbuf , const int recvcounts[] , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm , MPI_Request *request )
{
  Impl::ReduceTimer timer;
  int psize = Impl::psize(comm);
  int64_t counts = 0;
  for (int i=0; i<psize; ++i) {
    counts += recvcounts[i];
  }
  Impl::SendVolumeStats( counts, datatype );
  Impl::RecvVolumeStats( counts, datatype );
  return Impl::mpi_check_err(MPI_Ireduce_scatter( sendbuf , recvbuf , recvcounts , datatype , op , comm , request ));
}
inline int Ireduce_scatter_block( const void *sendbuf , void *recvbuf , int recvcount , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm , MPI_Request *request )
{
  Impl::ReduceTimer timer;
  Impl::SendVolumeStats( recvcount, datatype );
  Impl::RecvVolumeStats( recvcount, datatype );
  return Impl::mpi_check_err(MPI_Ireduce_scatter_block( sendbuf , recvbuf , recvcount , datatype , op , comm , request ));
}
#endif
inline int Irsend( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Irsend( buf , count , datatype , dest , tag , comm , request ));
}
inline int Is_thread_main( int *flag )
{
  return Impl::mpi_check_err(MPI_Is_thread_main( flag ));
}
#if UINTAH_ENABLE_MPI3
inline int Iscan( const void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm , MPI_Request *request )
{
  Impl::ScanTimer timer;
  Impl::SendVolumeStats( count, datatype );
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Iscan( sendbuf , recvbuf , count , datatype , op , comm , request ));
}
inline int Iscatter( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm , MPI_Request *request )
{
  Impl::ScatterTimer timer;
  if ( root == Impl::prank(comm) ) {
    Impl::SendVolumeStats( sendcount, sendtype );
  }
  else {
    Impl::RecvVolumeStats( recvcount, recvtype );
  }
  return Impl::mpi_check_err(MPI_Iscatter( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , root , comm , request ));
}
inline int Iscatterv( MPICONST void *sendbuf , MPICONST int sendcounts[] , MPICONST int displs[] , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm , MPI_Request *request )
{
  Impl::ScatterTimer timer;
  if ( root == Impl::prank(comm) ) {
    Impl::SendVolumeStats( sendcounts, sendtype, comm );
  }
  else {
    Impl::RecvVolumeStats( recvcount, recvtype );
  }
  return Impl::mpi_check_err(MPI_Iscatterv( sendbuf , sendcounts , displs , sendtype , recvbuf , recvcount , recvtype , root , comm , request ));
}
#endif
inline int Isend( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Isend( buf , count , datatype , dest , tag , comm , request ));
}
inline int Issend( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Issend( buf , count , datatype , dest , tag , comm , request ));
}
inline int Lookup_name( MPICONST char *service_name , MPI_Info info , char *port_name )
{
  return Impl::mpi_check_err(MPI_Lookup_name( service_name , info , port_name ));
}
#if UINTAH_ENABLE_MPI3
inline int Mprobe( int source , int tag , MPI_Comm comm , MPI_Message *message , MPI_Status *status )
{
  return Impl::mpi_check_err(MPI_Mprobe( source , tag , comm , message , status ));
}
inline int Mrecv( void *buf , int count , MPI_Datatype datatype , MPI_Message *message , MPI_Status *status )
{
  Impl::RecvTimer timer;
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Mrecv( buf , count , datatype , message , status ));
}
inline int Neighbor_allgather( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::GatherTimer timer;
  Impl::OneVolumeStats( sendcount, sendtype );
  return Impl::mpi_check_err(MPI_Neighbor_allgather( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm ));
}
inline int Neighbor_allgatherv( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , MPICONST int recvcounts[] , MPICONST int displs[] , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::AlltoallTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  const int nn = Impl::num_neighbors(comm);
  int64_t rcounts = 0;
  for (int i=0; i<nn; ++i) {
    rcounts += recvcounts[i];
  }
  Impl::RecvVolumeStats( rcounts, recvtype );
  return Impl::mpi_check_err(MPI_Neighbor_allgatherv( sendbuf , sendcount , sendtype , recvbuf , recvcounts , displs , recvtype , comm ));
}
inline int Neighbor_alltoall( const void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::AlltoallTimer timer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcount, recvtype );
  return Impl::mpi_check_err(MPI_Neighbor_alltoall( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , comm ));
}
inline int Neighbor_alltoallv( const void *sendbuf , const int sendcounts[] , const int sdispls[] , MPI_Datatype sendtype , void *recvbuf , const int recvcounts[] , const int rdispls[] , MPI_Datatype recvtype , MPI_Comm comm )
{
  Impl::AlltoallTimer timer;
  const int nn = Impl::num_neighbors(comm);
  int64_t rcounts = 0;
  int64_t scounts = 0;
  for (int i=0; i<nn; ++i) {
    scounts += sendcounts[i];
    rcounts += recvcounts[i];
  }
  Impl::SendVolumeStats( scounts, sendtype );
  Impl::RecvVolumeStats( rcounts, recvtype );
  return Impl::mpi_check_err(MPI_Neighbor_alltoallv( sendbuf , sendcounts , sdispls , sendtype , recvbuf , recvcounts , rdispls , recvtype , comm ));
}
inline int Neighbor_alltoallw( const void *sendbuf , const int sendcounts[] , const MPI_Aint sdispls[] , const MPI_Datatype sendtypes[] , void *recvbuf , const int recvcounts[] , const MPI_Aint rdispls[] , const MPI_Datatype recvtypes[] , MPI_Comm comm )
{
  // TODO handle multiple send and recv types
  Impl::AlltoallTimer timer;
  const int psize = Impl::psize(comm);
  int64_t rcounts = 0;
  int64_t scounts = 0;
  for (int i=0; i<psize; ++i) {
    scounts += sendcounts[i];
    rcounts += recvcounts[i];
  }
  Impl::SendVolumeStats( scounts, sendtypes[0] );
  Impl::RecvVolumeStats( rcounts, recvtypes[0] );
  return Impl::mpi_check_err(MPI_Neighbor_alltoallw( sendbuf , sendcounts , sdispls , sendtypes , recvbuf , recvcounts , rdispls , recvtypes , comm ));
}
inline int Op_commutative( MPI_Op op , int *commute )
{
  return Impl::mpi_check_err(MPI_Op_commutative( op , commute ));
}
#endif
inline int Op_create( MPI_User_function *user_fn , int commute , MPI_Op *op )
{
  return Impl::mpi_check_err(MPI_Op_create( user_fn , commute , op ));
}
inline int Op_free( MPI_Op *op )
{
  return Impl::mpi_check_err(MPI_Op_free( op ));
}
inline int Open_port( MPI_Info info , char *port_name )
{
  return Impl::mpi_check_err(MPI_Open_port( info , port_name ));
}
inline int Pack( MPICONST void *inbuf , int incount , MPI_Datatype datatype , void *outbuf , int outsize , int *position , MPI_Comm comm )
{
  return Impl::mpi_check_err(MPI_Pack( inbuf , incount , datatype , outbuf , outsize , position , comm ));
}
inline int Pack_external( MPICONST char datarep[] , MPICONST void *inbuf , int incount , MPI_Datatype datatype , void *outbuf , MPI_Aint outsize , MPI_Aint *position )
{
  return Impl::mpi_check_err(MPI_Pack_external( datarep , inbuf , incount , datatype , outbuf , outsize , position ));
}
inline int Pack_external_size( MPICONST char datarep[] , int incount , MPI_Datatype datatype , MPI_Aint *size )
{
  return Impl::mpi_check_err(MPI_Pack_external_size( datarep , incount , datatype , size ));
}
inline int Pack_size( int incount , MPI_Datatype datatype , MPI_Comm comm , int *size )
{
  return Impl::mpi_check_err(MPI_Pack_size( incount , datatype , comm , size ));
}
inline int Probe( int source , int tag , MPI_Comm comm , MPI_Status *status )
{
  return Impl::mpi_check_err(MPI_Probe( source , tag , comm , status ));
}
inline int Publish_name( MPICONST char *service_name , MPI_Info info , MPICONST char *port_name )
{
  return Impl::mpi_check_err(MPI_Publish_name( service_name , info , port_name ));
}
inline int Put( MPICONST void *origin_addr , int origin_count , MPI_Datatype origin_datatype , int target_rank , MPI_Aint target_disp , int target_count , MPI_Datatype target_datatype , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err(MPI_Put( origin_addr , origin_count , origin_datatype , target_rank , target_disp , target_count , target_datatype , win ));
}
inline int Query_thread( int *provided )
{
  return Impl::mpi_check_err(MPI_Query_thread( provided ));
}
#if UINTAH_ENABLE_MPI3
inline int Raccumulate( const void *origin_addr , int origin_count , MPI_Datatype origin_datatype , int target_rank , MPI_Aint target_disp , int target_count , MPI_Datatype target_datatype , MPI_Op op , MPI_Win win , MPI_Request *request )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err(MPI_Raccumulate( origin_addr , origin_count , origin_datatype , target_rank , target_disp , target_count , target_datatype , op , win , request ));
}
#endif
inline int Recv( void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Status *status )
{
  Impl::RecvTimer timer;
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Recv( buf , count , datatype , source , tag , comm , status ));
}
inline int Recv_init( void *buf , int count , MPI_Datatype datatype , int source , int tag , MPI_Comm comm , MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Recv_init( buf , count , datatype , source , tag , comm , request ));
}
inline int Reduce( MPICONST void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , int root , MPI_Comm comm )
{
  Impl::ReduceTimer timer;
  Impl::SendVolumeStats( count, datatype );
  if (Impl::prank(comm) == root) {
    Impl::RecvVolumeStats( count, datatype );
  }
  return Impl::mpi_check_err(MPI_Reduce( sendbuf , recvbuf , count , datatype , op , root , comm ));
}
inline int Reduce_local( MPICONST void *inbuf , void *inoutbuf , int count , MPI_Datatype datatype , MPI_Op op )
{
  Impl::ReduceTimer timer;
  return Impl::mpi_check_err(MPI_Reduce_local( inbuf , inoutbuf , count , datatype , op ));
}
inline int Reduce_scatter( MPICONST void *sendbuf , void *recvbuf , MPICONST int recvcounts[] , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm )
{
  Impl::ReduceTimer timer;
  int psize = Impl::psize(comm);
  int64_t counts = 0;
  for (int i=0; i<psize; ++i) {
    counts += recvcounts[i];
  }
  Impl::SendVolumeStats( counts, datatype );
  Impl::RecvVolumeStats( recvcounts, datatype, comm );
  return Impl::mpi_check_err(MPI_Reduce_scatter( sendbuf , recvbuf , recvcounts , datatype , op , comm ));
}
#if UINTAH_ENABLE_MPI3
inline int Reduce_scatter_block( const void *sendbuf , void *recvbuf , int recvcount , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm )
{
  Impl::ReduceTimer timer;
  Impl::SendVolumeStats( recvcount, datatype );
  Impl::RecvVolumeStats( recvcount, datatype );
  return Impl::mpi_check_err(MPI_Reduce_scatter_block( sendbuf , recvbuf , recvcount , datatype , op , comm ));
}
#endif
inline int Request_free( MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Request_free( request ));
}
inline int Request_get_status( MPI_Request request , int *flag , MPI_Status *status )
{
  return Impl::mpi_check_err(MPI_Request_get_status( request , flag , status ));
}
#if UINTAH_ENABLE_MPI3
inline int Rget( void *origin_addr , int origin_count , MPI_Datatype origin_datatype , int target_rank , MPI_Aint target_disp , int target_count , MPI_Datatype target_datatype , MPI_Win win , MPI_Request *request )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err(MPI_Rget( origin_addr , origin_count , origin_datatype , target_rank , target_disp , target_count , target_datatype , win , request ));
}
inline int Rget_accumulate( const void *origin_addr , int origin_count , MPI_Datatype origin_datatype , void *result_addr , int result_count , MPI_Datatype result_datatype , int target_rank , MPI_Aint target_disp , int target_count , MPI_Datatype target_datatype , MPI_Op op , MPI_Win win , MPI_Request *request )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err(MPI_Rget_accumulate( origin_addr , origin_count , origin_datatype , result_addr , result_count , result_datatype , target_rank , target_disp , target_count , target_datatype , op , win , request ));
}
inline int Rput( const void *origin_addr , int origin_count , MPI_Datatype origin_datatype , int target_rank , MPI_Aint target_disp , int target_count , MPI_Datatype target_datatype , MPI_Win win , MPI_Request *request )
{
  Impl::OneSidedTimer timer;
  Impl::OneVolumeStats( origin_count, origin_datatype);
  return Impl::mpi_check_err(MPI_Rput( origin_addr , origin_count , origin_datatype , target_rank , target_disp , target_count , target_datatype , win , request ));
}
#endif
inline int Rsend( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Rsend( buf , count , datatype , dest , tag , comm ));
}
inline int Rsend_init( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Rsend_init( buf , count , datatype , dest , tag , comm , request ));
}
inline int Scan( MPICONST void *sendbuf , void *recvbuf , int count , MPI_Datatype datatype , MPI_Op op , MPI_Comm comm )
{
  Impl::ScanTimer timer;
  Impl::SendVolumeStats( count, datatype );
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Scan( sendbuf , recvbuf , count , datatype , op , comm ));
}
inline int Scatter( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm )
{
  Impl::ScatterTimer timer;
  if ( root == Impl::prank(comm) ) {
    Impl::SendVolumeStats( sendcount, sendtype );
  }
  else {
    Impl::RecvVolumeStats( recvcount, recvtype );
  }
  return Impl::mpi_check_err(MPI_Scatter( sendbuf , sendcount , sendtype , recvbuf , recvcount , recvtype , root , comm ));
}
inline int Scatterv( MPICONST void *sendbuf , MPICONST int *sendcounts , MPICONST int *displs , MPI_Datatype sendtype , void *recvbuf , int recvcount , MPI_Datatype recvtype , int root , MPI_Comm comm )
{
  Impl::ScatterTimer timer;
  if ( root == Impl::prank(comm) ) {
    Impl::SendVolumeStats( sendcounts, sendtype, comm );
  }
  else {
    Impl::RecvVolumeStats( recvcount, recvtype );
  }
  return Impl::mpi_check_err(MPI_Scatterv( sendbuf , sendcounts , displs , sendtype , recvbuf , recvcount , recvtype , root , comm ));
}
inline int Send( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Send( buf , count , datatype , dest , tag , comm ));
}
inline int Send_init( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Send_init( buf , count , datatype , dest , tag , comm , request ));
}
inline int Sendrecv( MPICONST void *sendbuf , int sendcount , MPI_Datatype sendtype , int dest , int sendtag , void *recvbuf , int recvcount , MPI_Datatype recvtype , int source , int recvtag , MPI_Comm comm , MPI_Status *status )
{
  Impl::SendTimer stimer;
  Impl::RecvTimer rtimer;
  Impl::SendVolumeStats( sendcount, sendtype );
  Impl::RecvVolumeStats( recvcount, recvtype );
  return Impl::mpi_check_err(MPI_Sendrecv( sendbuf , sendcount , sendtype , dest , sendtag , recvbuf , recvcount , recvtype , source , recvtag , comm , status ));
}
inline int Sendrecv_replace( void *buf , int count , MPI_Datatype datatype , int dest , int sendtag , int source , int recvtag , MPI_Comm comm , MPI_Status *status )
{
  Impl::SendTimer stimer;
  Impl::RecvTimer rtimer;
  Impl::SendVolumeStats( count, datatype );
  Impl::RecvVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Sendrecv_replace( buf , count , datatype , dest , sendtag , source , recvtag , comm , status ));
}
inline int Ssend( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm )
{
  Impl::SendTimer timer;
  Impl::SendVolumeStats( count, datatype );
  return Impl::mpi_check_err(MPI_Ssend( buf , count , datatype , dest , tag , comm ));
}
inline int Ssend_init( MPICONST void *buf , int count , MPI_Datatype datatype , int dest , int tag , MPI_Comm comm , MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Ssend_init( buf , count , datatype , dest , tag , comm , request ));
}
inline int Start( MPI_Request *request )
{
  return Impl::mpi_check_err(MPI_Start( request ));
}
inline int Startall( int count , MPI_Request array_of_requests[] )
{
  return Impl::mpi_check_err(MPI_Startall( count , array_of_requests ));
}
inline int Status_set_cancelled( MPI_Status *status , int flag )
{
  return Impl::mpi_check_err(MPI_Status_set_cancelled( status , flag ));
}
inline int Status_set_elements( MPI_Status *status , MPI_Datatype datatype , int count )
{
  return Impl::mpi_check_err(MPI_Status_set_elements( status , datatype , count ));
}
#if UINTAH_ENABLE_MPI3
inline int Status_set_elements_x( MPI_Status *status , MPI_Datatype datatype , MPI_Count count )
{
  return Impl::mpi_check_err(MPI_Status_set_elements_x( status , datatype , count ));
}
#endif
inline int Test( MPI_Request *request , int *flag , MPI_Status *status )
{
  Impl::TestTimer timer;
  return Impl::mpi_check_err(MPI_Test( request , flag , status ));
}
inline int Test_cancelled( MPICONST MPI_Status *status , int *flag )
{
  return Impl::mpi_check_err(MPI_Test_cancelled( status , flag ));
}
inline int Testall( int count , MPI_Request array_of_requests[] , int *flag , MPI_Status array_of_statuses[] )
{
  Impl::TestTimer timer;
  return Impl::mpi_check_err(MPI_Testall( count , array_of_requests , flag , array_of_statuses ));
}
inline int Testany( int count , MPI_Request array_of_requests[] , int *indx , int *flag , MPI_Status *status )
{
  Impl::TestTimer timer;
  return Impl::mpi_check_err(MPI_Testany( count , array_of_requests , indx , flag , status ));
}
inline int Testsome( int incount , MPI_Request array_of_requests[] , int *outcount , int array_of_indices[] , MPI_Status array_of_statuses[] )
{
  Impl::TestTimer timer;
  return Impl::mpi_check_err(MPI_Testsome( incount , array_of_requests , outcount , array_of_indices , array_of_statuses ));
}
inline int Topo_test( MPI_Comm comm , int *status )
{
  return Impl::mpi_check_err(MPI_Topo_test( comm , status ));
}
inline int Type_commit( MPI_Datatype *datatype )
{
  return Impl::mpi_check_err(MPI_Type_commit( datatype ));
}
inline int Type_contiguous( int count , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_contiguous( count , oldtype , newtype ));
}
inline int Type_create_darray( int size , int rank , int ndims , MPICONST int array_of_gsizes[] , MPICONST int array_of_distribs[] , MPICONST int array_of_dargs[] , MPICONST int array_of_psizes[] , int order , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_darray( size , rank , ndims , array_of_gsizes , array_of_distribs , array_of_dargs , array_of_psizes , order , oldtype , newtype ));
}
inline int Type_create_hindexed( int count , MPICONST int array_of_blocklengths[] , MPICONST MPI_Aint array_of_displacements[] , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_hindexed( count , array_of_blocklengths , array_of_displacements , oldtype , newtype ));
}
#if UINTAH_ENABLE_MPI3
inline int Type_create_hindexed_block( int count , int blocklength , const MPI_Aint array_of_displacements[] , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_hindexed_block( count , blocklength , array_of_displacements , oldtype , newtype ));
}
#endif
inline int Type_create_hvector( int count , int blocklength , MPI_Aint stride , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_hvector( count , blocklength , stride , oldtype , newtype ));
}
inline int Type_create_indexed_block( int count , int blocklength , MPICONST int array_of_displacements[] , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_indexed_block( count , blocklength , array_of_displacements , oldtype , newtype ));
}
inline int Type_create_keyval( MPI_Type_copy_attr_function *type_copy_attr_fn , MPI_Type_delete_attr_function *type_delete_attr_fn , int *type_keyval , void *extra_state )
{
  return Impl::mpi_check_err(MPI_Type_create_keyval( type_copy_attr_fn , type_delete_attr_fn , type_keyval , extra_state ));
}
inline int Type_create_resized( MPI_Datatype oldtype , MPI_Aint lb , MPI_Aint extent , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_resized( oldtype , lb , extent , newtype ));
}
inline int Type_create_struct( int count , MPICONST int array_of_blocklengths[] , MPICONST MPI_Aint array_of_displacements[] , MPICONST MPI_Datatype array_of_types[] , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_struct( count , array_of_blocklengths , array_of_displacements , array_of_types , newtype ));
}
inline int Type_create_subarray( int ndims , MPICONST int array_of_sizes[] , MPICONST int array_of_subsizes[] , MPICONST int array_of_starts[] , int order , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_create_subarray( ndims , array_of_sizes , array_of_subsizes , array_of_starts , order , oldtype , newtype ));
}
inline int Type_delete_attr( MPI_Datatype datatype , int type_keyval )
{
  return Impl::mpi_check_err(MPI_Type_delete_attr( datatype , type_keyval ));
}
inline int Type_dup( MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_dup( oldtype , newtype ));
}
inline int Type_free( MPI_Datatype *datatype )
{
  return Impl::mpi_check_err(MPI_Type_free( datatype ));
}
inline int Type_free_keyval( int *type_keyval )
{
  return Impl::mpi_check_err(MPI_Type_free_keyval( type_keyval ));
}
inline int Type_get_attr( MPI_Datatype datatype , int type_keyval , void *attribute_val , int *flag )
{
  return Impl::mpi_check_err(MPI_Type_get_attr( datatype , type_keyval , attribute_val , flag ));
}
inline int Type_get_contents( MPI_Datatype datatype , int max_integers , int max_addresses , int max_datatypes , int array_of_integers[] , MPI_Aint array_of_addresses[] , MPI_Datatype array_of_datatypes[] )
{
  return Impl::mpi_check_err(MPI_Type_get_contents( datatype , max_integers , max_addresses , max_datatypes , array_of_integers , array_of_addresses , array_of_datatypes ));
}
inline int Type_get_envelope( MPI_Datatype datatype , int *num_integers , int *num_addresses , int *num_datatypes , int *combiner )
{
  return Impl::mpi_check_err(MPI_Type_get_envelope( datatype , num_integers , num_addresses , num_datatypes , combiner ));
}
inline int Type_get_extent( MPI_Datatype datatype , MPI_Aint *lb , MPI_Aint *extent )
{
  return Impl::mpi_check_err(MPI_Type_get_extent( datatype , lb , extent ));
}
#if UINTAH_ENABLE_MPI3
inline int Type_get_extent_x( MPI_Datatype datatype , MPI_Count *lb , MPI_Count *extent )
{
  return Impl::mpi_check_err(MPI_Type_get_extent_x( datatype , lb , extent ));
}
#endif
inline int Type_get_name( MPI_Datatype datatype , char *type_name , int *resultlen )
{
  return Impl::mpi_check_err(MPI_Type_get_name( datatype , type_name , resultlen ));
}
inline int Type_get_true_extent( MPI_Datatype datatype , MPI_Aint *true_lb , MPI_Aint *true_extent )
{
  return Impl::mpi_check_err(MPI_Type_get_true_extent( datatype , true_lb , true_extent ));
}
#if UINTAH_ENABLE_MPI3
inline int Type_get_true_extent_x( MPI_Datatype datatype , MPI_Count *lb , MPI_Count *extent )
{
  return Impl::mpi_check_err(MPI_Type_get_true_extent_x( datatype , lb , extent ));
}
#endif
inline int Type_indexed( int count , MPICONST int *array_of_blocklengths , MPICONST int *array_of_displacements , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_indexed( count , array_of_blocklengths , array_of_displacements , oldtype , newtype ));
}
inline int Type_set_attr( MPI_Datatype datatype , int type_keyval , void *attribute_val )
{
  return Impl::mpi_check_err(MPI_Type_set_attr( datatype , type_keyval , attribute_val ));
}
inline int Type_set_name( MPI_Datatype datatype , MPICONST char *type_name )
{
  return Impl::mpi_check_err(MPI_Type_set_name( datatype , type_name ));
}
inline int Type_size( MPI_Datatype datatype , int *size )
{
  return Impl::mpi_check_err(MPI_Type_size( datatype , size ));
}
#if UINTAH_ENABLE_MPI3
inline int Type_size_x( MPI_Datatype datatype , MPI_Count *size )
{
  return Impl::mpi_check_err(MPI_Type_size_x( datatype , size ));
}
#endif
inline int Type_vector( int count , int blocklength , int stride , MPI_Datatype oldtype , MPI_Datatype *newtype )
{
  return Impl::mpi_check_err(MPI_Type_vector( count , blocklength , stride , oldtype , newtype ));
}
inline int Unpack( MPICONST void *inbuf , int insize , int *position , void *outbuf , int outcount , MPI_Datatype datatype , MPI_Comm comm )
{
  return Impl::mpi_check_err(MPI_Unpack( inbuf , insize , position , outbuf , outcount , datatype , comm ));
}
inline int Unpack_external( MPICONST char datarep[] , MPICONST void *inbuf , MPI_Aint insize , MPI_Aint *position , void *outbuf , int outcount , MPI_Datatype datatype )
{
  return Impl::mpi_check_err(MPI_Unpack_external( datarep , inbuf , insize , position , outbuf , outcount , datatype ));
}
inline int Unpublish_name( MPICONST char *service_name , MPI_Info info , MPICONST char *port_name )
{
  return Impl::mpi_check_err(MPI_Unpublish_name( service_name , info , port_name ));
}
inline int Wait( MPI_Request *request , MPI_Status *status )
{
  Impl::WaitTimer timer;
  return Impl::mpi_check_err(MPI_Wait( request , status ));
}
inline int Waitall( int count , MPI_Request array_of_requests[] , MPI_Status array_of_statuses[] )
{
  Impl::WaitTimer timer;
  return Impl::mpi_check_err(MPI_Waitall( count , array_of_requests , array_of_statuses ));
}
inline int Waitany( int count , MPI_Request array_of_requests[] , int *indx , MPI_Status *status )
{
  Impl::WaitTimer timer;
  return Impl::mpi_check_err(MPI_Waitany( count , array_of_requests , indx , status ));
}
inline int Waitsome( int incount , MPI_Request array_of_requests[] , int *outcount , int array_of_indices[] , MPI_Status array_of_statuses[] )
{
  Impl::WaitTimer timer;
  return Impl::mpi_check_err(MPI_Waitsome( incount , array_of_requests , outcount , array_of_indices , array_of_statuses ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_allocate( MPI_Aint size , int disp_unit , MPI_Info info , MPI_Comm comm , void *baseptr , MPI_Win *win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_allocate( size , disp_unit , info , comm , baseptr , win ));
}
inline int Win_allocate_shared( MPI_Aint size , int disp_unit , MPI_Info info , MPI_Comm comm , void *baseptr , MPI_Win *win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_allocate_shared( size , disp_unit , info , comm , baseptr , win ));
}
inline int Win_attach( MPI_Win win , void *base , MPI_Aint size )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_attach( win , base , size ));
}
#endif
inline int Win_call_errhandler( MPI_Win win , int errorcode )
{
  return Impl::mpi_check_err(MPI_Win_call_errhandler( win , errorcode ));
}
inline int Win_complete( MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_complete( win ));
}
inline int Win_create( void *base , MPI_Aint size , int disp_unit , MPI_Info info , MPI_Comm comm , MPI_Win *win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_create( base , size , disp_unit , info , comm , win ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_create_dynamic( MPI_Info info , MPI_Comm comm , MPI_Win *win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_create_dynamic( info , comm , win ));
}
#endif
inline int Win_create_errhandler( MPI_Win_errhandler_function *win_errhandler_fn , MPI_Errhandler *errhandler )
{
  return Impl::mpi_check_err(MPI_Win_create_errhandler( win_errhandler_fn , errhandler ));
}
inline int Win_create_keyval( MPI_Win_copy_attr_function *win_copy_attr_fn , MPI_Win_delete_attr_function *win_delete_attr_fn , int *win_keyval , void *extra_state )
{
  return Impl::mpi_check_err(MPI_Win_create_keyval( win_copy_attr_fn , win_delete_attr_fn , win_keyval , extra_state ));
}
inline int Win_delete_attr( MPI_Win win , int win_keyval )
{
  return Impl::mpi_check_err(MPI_Win_delete_attr( win , win_keyval ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_detach( MPI_Win win , const void *base )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_detach( win , base ));
}
#endif
inline int Win_fence( int assert , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::WaitTimer wtimer;
  return Impl::mpi_check_err(MPI_Win_fence( assert , win ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_flush( int rank , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_flush( rank , win ));
}
inline int Win_flush_all( MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_flush_all( win ));
}
inline int Win_flush_local( int rank , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_flush_local( rank , win ));
}
inline int Win_flush_local_all( MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_flush_local_all( win ));
}
#endif
inline int Win_free( MPI_Win *win )
{
  return Impl::mpi_check_err(MPI_Win_free( win ));
}
inline int Win_free_keyval( int *win_keyval )
{
  return Impl::mpi_check_err(MPI_Win_free_keyval( win_keyval ));
}
inline int Win_get_attr( MPI_Win win , int win_keyval , void *attribute_val , int *flag )
{
  return Impl::mpi_check_err(MPI_Win_get_attr( win , win_keyval , attribute_val , flag ));
}
inline int Win_get_errhandler( MPI_Win win , MPI_Errhandler *errhandler )
{
  return Impl::mpi_check_err(MPI_Win_get_errhandler( win , errhandler ));
}
inline int Win_get_group( MPI_Win win , MPI_Group *group )
{
  return Impl::mpi_check_err(MPI_Win_get_group( win , group ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_get_info( MPI_Win win , MPI_Info *info_used )
{
  return Impl::mpi_check_err(MPI_Win_get_info( win , info_used ));
}
#endif
inline int Win_get_name( MPI_Win win , char *win_name , int *resultlen )
{
  return Impl::mpi_check_err(MPI_Win_get_name( win , win_name , resultlen ));
}
inline int Win_lock( int lock_type , int rank , int assert , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_lock( lock_type , rank , assert , win ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_lock_all( int assert , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_lock_all( assert , win ));
}
#endif
inline int Win_post( MPI_Group group , int assert , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_post( group , assert , win ));
}
inline int Win_set_attr( MPI_Win win , int win_keyval , void *attribute_val )
{
  return Impl::mpi_check_err(MPI_Win_set_attr( win , win_keyval , attribute_val ));
}
inline int Win_set_errhandler( MPI_Win win , MPI_Errhandler errhandler )
{
  return Impl::mpi_check_err(MPI_Win_set_errhandler( win , errhandler ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_set_info( MPI_Win win , MPI_Info info )
{
  return Impl::mpi_check_err(MPI_Win_set_info( win , info ));
}
#endif
inline int Win_set_name( MPI_Win win , MPICONST char *win_name )
{
  return Impl::mpi_check_err(MPI_Win_set_name( win , win_name ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_shared_query( MPI_Win win , int rank , MPI_Aint *size , int *disp_unit , void *baseptr )
{
  return Impl::mpi_check_err(MPI_Win_shared_query( win , rank , size , disp_unit , baseptr ));
}
#endif
inline int Win_start( MPI_Group group , int assert , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_start( group , assert , win ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_sync( MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_sync( win ));
}
#endif
inline int Win_test( MPI_Win win , int *flag )
{
  Impl::OneSidedTimer timer;
  Impl::TestTimer ttimer;
  return Impl::mpi_check_err(MPI_Win_test( win , flag ));
}
inline int Win_unlock( int rank , MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_unlock( rank , win ));
}
#if UINTAH_ENABLE_MPI3
inline int Win_unlock_all( MPI_Win win )
{
  Impl::OneSidedTimer timer;
  return Impl::mpi_check_err(MPI_Win_unlock_all( win ));
}
#endif
inline int Win_wait( MPI_Win win )
{
  Impl::OneSidedTimer timer;
  Impl::WaitTimer wtimer;
  return Impl::mpi_check_err(MPI_Win_wait( win ));
}

} // end namespace Uintah::MPI
} // namespace Uintah

#endif // CORE_PARALLEL_UINTAHMPI_H
