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

#include <CCA/Components/Schedulers/RuntimeStats.hpp>
#include <Core/Util/DOUT.hpp>

#include <iomanip>

namespace Uintah {

namespace {

Dout g_exec_times{"ExecTimes", false};
Dout g_wait_times{"WaitTimes", false};

size_t g_num_global_task_ids{0};
std::unique_ptr< std::string[] >          g_task_names{nullptr};
std::unique_ptr< std::atomic<int64_t>[] > g_task_exec_times{nullptr};
std::unique_ptr< std::atomic<int64_t>[] > g_task_wait_times{nullptr};

// TODO implement a globally consisent task id
size_t impl_get_global_id( DetailedTask const* t)
{
  return ~static_cast<size_t>(0);
}

} // unnamed namespace

std::map< std::string, std::function<int64_t()> > RuntimeStats::s_allocators{};
std::map< std::string, std::function<int64_t()> > RuntimeStats::s_timers{};
std::vector< std::function<void()> >              RuntimeStats::s_reset_timers{};


std::atomic<int64_t> * RuntimeStats::get_atomic_exec_ptr( DetailedTask const* t)
{
  if (!g_exec_times) { return nullptr; }

  const size_t id = impl_get_global_id(t);

  return id < g_num_global_task_ids ?
           & g_task_exec_times[ id ] : nullptr ;
}

std::atomic<int64_t> * RuntimeStats::get_atomic_wait_ptr( DetailedTask const* t)
{
  if (!g_wait_times) { return nullptr; }

  const size_t id = impl_get_global_id(t);

  return id < g_num_global_task_ids ?
           & g_task_wait_times[ id ] : nullptr ;
}


void RuntimeStats::reset_timers()
{
  for (auto & f : s_reset_timers) { f(); }

  constexpr int64_t zero = 0;

  for (size_t i=0; i<g_num_global_task_ids; ++i) {
    g_task_exec_times[i].store( zero, std::memory_order_relaxed );
    g_task_wait_times[i].store( zero, std::memory_order_relaxed );
  }
}

void RuntimeStats::initialize_timestep( std::vector<TaskGraph *> const &  /*graph*/ )
{
  static bool first_init = true;
  if (first_init) {

    register_timer_tag( TaskExecTag{}, "Task Exec", Max );
    register_timer_tag( TaskWaitTag{}, "Task Wait", Min );

    register_timer_tag( CollectiveTag{}, "Total Coll", Total);
    register_timer_tag( RecvTag{}, "Total Recv", Total);
    register_timer_tag( SendTag{}, "Total Send", Total);
    register_timer_tag( TestTag{}, "Total Test", Total);
    register_timer_tag( WaitTag{}, "Total Wait", Total);

    register_timer_tag( CollectiveMPITag{}, "MPI Coll", Total);
    register_timer_tag( RecvMPITag{}, "MPI Recv", Total);
    register_timer_tag( SendMPITag{}, "MPI Send", Total);

    reset_timers();

    first_init = false;
  }
  // TODO
  // init
  // g_num_global_task_ids
  // g_task_names
  // g_task_exec_times
  // g_task_wait_times
}



namespace {


/// convert a bytes to human readable string
inline std::string bytes_to_string( int64_t bytes )
{
  constexpr int64_t one = 1;
  constexpr int64_t KB = one << 10;
  constexpr int64_t MB = one << 20;
  constexpr int64_t GB = one << 30;
  constexpr int64_t TB = one << 40;
  constexpr int64_t PB = one << 50;

  std::ostringstream out;

  if ( bytes < KB ) {
    out << bytes << " B";
  }
  else if ( bytes < MB ) {
    out << std::setprecision(3) << (static_cast<double>(bytes) / KB) << " KB";
  }
  else if ( bytes < GB ) {
    out << std::setprecision(3) << (static_cast<double>(bytes) / MB) << " MB";
  }
  else if ( bytes < TB ) {
    out << std::setprecision(3) << (static_cast<double>(bytes) / GB) << " GB";
  }
  else if ( bytes < PB ) {
    out << std::setprecision(3) << (static_cast<double>(bytes) / TB) << " TB";
  }
  else {
    out << std::setprecision(3) << (static_cast<double>(bytes) / PB) << " PB";
  }

  return out.str();
}

/// convert a nanoseconds to human readable string
inline std::string nanoseconds_to_string( int64_t ns )
{
  std::ostringstream out;

  if ( (double)ns < 1.0e5 ) {
    out << std::setprecision(3) << ns << " ns";
  }
  else if ( (double)ns < 1.0e8) {
    out << std::setprecision(3) << (ns * 1.0e-6) << " ms";
  }
  else {
    out << std::setprecision(3) << (ns * 1.0e-9) << " s ";
  }

  return out.str();
}

/// convert a nanoseconds to human readable string
inline std::string nanoseconds_to_string( double ns )
{
  std::ostringstream out;

  if ( ns < 1.0e5 ) {
    out << std::setprecision(3) << ns << " ns";
  }
  else if ( ns < 1.0e8) {
    out << std::setprecision(3) << (ns * 1.0e-6) << " ms";
  } else {
    out << std::setprecision(3) << (ns * 1.0e-9) << " s ";
  }

  return out.str();
}

enum { RANK=0, SUM, MIN, MAX };

template <typename T>
void rank_sum_min_max_impl( T const * in, T * inout, int len )
{
  const int size = len/4;
  for (int i=0; i<size; ++i) {
    const int off = 4*i;
    inout[off+RANK] = in[off+MAX] < inout[off+MAX] ? inout[off+RANK] : in[off+RANK] ; // max_rank
    inout[off+SUM] += in[off+SUM] ;                                           // sum
    inout[off+MIN]  = inout[off+MIN] < in[off+MIN] ? inout[off+MIN] : in[off+MIN] ; // min
    inout[off+MAX]  = in[off+MAX] < inout[off+MAX] ? inout[off+MAX] : in[off+MAX] ; // max
  }
}

extern "C" void rank_sum_min_max( void * in, void * inout, int * len, MPI_Datatype * type )
{
 rank_sum_min_max_impl( reinterpret_cast<int64_t*>(in), reinterpret_cast<int64_t*>(inout), *len );
}

MPI_Op rank_sum_min_max_op;

enum { Q1=0, Q2, Q3, Q4 };

} // unnamed namespace


void RuntimeStats::report( MPI_Comm comm, InfoStats & stats )
{
  {
    static bool init = false;
    if (!init) {
      MPI_Op_create( rank_sum_min_max, true, &rank_sum_min_max_op );

      init = true;
    }
  }

  int psize;
  int prank;

  MPI_Comm_size( comm, &psize );
  MPI_Comm_rank( comm, &prank );

  int num_timers = static_cast<int>(s_timers.size());

  if( g_exec_times ) {
    num_timers += (int)g_num_global_task_ids;
  }

  if( g_wait_times ) {
    num_timers += (int)g_num_global_task_ids;
  }

  const int num_allocators = static_cast<int>(s_allocators.size());

  const int num_values = num_timers + num_allocators;

  const int data_size = num_values * 4;

  std::vector<int64_t>     local_data;
  std::vector<int64_t>     data;
  std::vector<int64_t>     histograms;
  std::vector<int64_t>     global_data;
  std::vector<int64_t>     global_histograms;
  std::vector<std::string> names;

  local_data.reserve( num_values );
  data.reserve( data_size );

  if (prank==0) {
    names.reserve( num_values );
  }

  for( auto const& p : s_timers ) {
    if (prank==0) {
      names.push_back( p.first );
    }
    const int64_t ns = p.second();
    local_data.push_back(ns);
    data.push_back(prank); // rank
    data.push_back(ns); // total
    data.push_back(ns); // min
    data.push_back(ns); // max
  }

  if (g_exec_times) {
    for (size_t i=0; i<g_num_global_task_ids; ++i) {
      if (prank==0) {
        names.push_back( g_task_names[i] );
      }
      const int64_t ns = g_task_exec_times[i].load( std::memory_order_relaxed );
      local_data.push_back(ns);
      data.push_back(prank); // rank
      data.push_back(ns); // total
      data.push_back(ns); // min
      data.push_back(ns); // max
    }
  }

  if (g_wait_times) {
    for (size_t i=0; i<g_num_global_task_ids; ++i) {
      if (prank==0) {
        names.push_back( g_task_names[i] );
      }
      const int64_t ns = g_task_wait_times[i].load( std::memory_order_relaxed );
      local_data.push_back(ns);
      data.push_back(prank); // rank
      data.push_back(ns); // total
      data.push_back(ns); // min
      data.push_back(ns); // max
    }
  }

  for ( auto const& p : s_allocators ) {
    if (prank==0) {
      names.push_back( p.first );
    }
    const int64_t bytes = p.second();
    local_data.push_back(bytes);
    data.push_back(prank); // rank
    data.push_back(bytes); // total
    data.push_back(bytes); // min
    data.push_back(bytes); // max
  }


  global_data.resize(data_size);
  MPI_Allreduce( data.data(), global_data.data(), data_size, MPI_INT64_T, rank_sum_min_max_op, comm);

  histograms.resize( data_size, 0 );
  global_histograms.resize( data_size, 0 );

  for( int i=0; i<num_values; ++i ) {
    const int off = 4*i;
    const int64_t min = global_data[off+MIN];
    const int64_t max = global_data[off+MAX];
    const int64_t mag = max - min;

    const int64_t t = local_data[i] - min;

    int bin = 0 < mag ? 4*t / mag : 0;
    bin = bin < 4 ? bin : 3;

    histograms[off+bin] = 1;
  }

  MPI_Reduce(histograms.data(), global_histograms.data(), data_size, MPI_INT64_T, MPI_SUM, 0, comm);

  Dout mpi_report("MPIReport", false);

  if (mpi_report && prank == 0) {

    const int w_desc = 12;
    const int w_num  = 10;
    const int w_hist = 35;
    const int w_load = 18;

    printf( "%*s%*s%*s%*s%*s%*s%*s%*s\n"
        ,w_desc, "Description:"
        ,w_num,  "Total:"
        ,w_num,  "Avg:"
        ,w_num,  "Min:"
        ,w_num,  "Max:"
        ,w_num,  "Mrank:"
        ,w_hist, "Hist \% [  Q1 |  Q2 |  Q3 |  Q4 ]:"
        ,w_load, "\%Load Imbalance"
    );

    for (int i=0; i<num_values; ++i) {
      const int off = 4*i;
      const int max_rank  = global_data[off+RANK];
      const int64_t total = global_data[off+SUM];
      const int64_t min   = global_data[off+MIN];
      const int64_t max   = global_data[off+MAX];

      const double avg = static_cast<double>(total) / psize;

      const double q1 = static_cast<double>(100 * global_histograms[off+Q1]) / psize;
      const double q2 = static_cast<double>(100 * global_histograms[off+Q2]) / psize;
      const double q3 = static_cast<double>(100 * global_histograms[off+Q3]) / psize;
      const double q4 = static_cast<double>(100 * global_histograms[off+Q4]) / psize;

      const double load = max != 0 ? (100.0 * (1.0 - (avg / max))) : 0.0;

      printf("%*s:%*s%*s%*s%*s%*d          %6.1f%6.1f%6.1f%6.1f%8.1f\n"
          , w_desc-1, names[i].c_str()
          , w_num, ( i < num_timers ? nanoseconds_to_string( total ) : bytes_to_string( total )).c_str()
          , w_num, ( i < num_timers ? nanoseconds_to_string( avg ) : bytes_to_string( avg )).c_str()
          , w_num, ( i < num_timers ? nanoseconds_to_string( min ) : bytes_to_string( min )).c_str()
          , w_num, ( i < num_timers ? nanoseconds_to_string( max ) : bytes_to_string( max )).c_str()
          , w_num, max_rank
          , q1
          , q2
          , q3
          , q4
          , load
          );

    }
    printf("\n");
  }



  // update SimulationState
  stats[SimulationState::TaskExecTime]       += TripTimer< TaskExecTag >::max_seconds()
                                                - stats[SimulationState::OutputFileIOTime]  // don't count output time or bytes
                                                - stats[SimulationState::OutputFileIORate];
  stats[SimulationState::TaskLocalCommTime]  += RecvTimer::total_seconds() + SendTimer::total_seconds();
  stats[SimulationState::TaskWaitCommTime]   += TestTimer::total_seconds() + WaitTimer::total_seconds();
  stats[SimulationState::TaskGlobalCommTime] += CollectiveTimer::total_seconds();

  reset_timers();
}

} // namespace Uintah
