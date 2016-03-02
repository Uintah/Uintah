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
Dout g_print_histograms{"Histograms", false};

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
    out << std::setprecision(5) << (static_cast<double>(bytes) / KB) << " KB";
  }
  else if ( bytes < GB ) {
    out << std::setprecision(5) << (static_cast<double>(bytes) / MB) << " MB";
  }
  else if ( bytes < TB ) {
    out << std::setprecision(5) << (static_cast<double>(bytes) / GB) << " GB";
  }
  else if ( bytes < PB ) {
    out << std::setprecision(5) << (static_cast<double>(bytes) / TB) << " TB";
  }
  else {
    out << std::setprecision(5) << (static_cast<double>(bytes) / PB) << " PB";
  }

  return out.str();
}


/// convert a nanoseconds to human readable string
inline std::string nanoseconds_to_string( int64_t ns )
{
  constexpr int64_t one    = 1;
  constexpr int64_t MICRO  = one << 3;
  constexpr int64_t MILLA  = one << 6;
  constexpr int64_t SECOND = one << 9;
  constexpr int64_t MINUTE = SECOND * 60;
  constexpr int64_t HOUR   = MINUTE * 60;

  std::ostringstream out;

  if ( ns < MICRO ) {
    out << ns << " nanoseconds";
  }
  else if ( ns < MILLA ) {
    out << std::setprecision(5) << (static_cast<double>(ns) / MICRO) << " microseconds";
  }
  else if ( ns < SECOND ) {
    out << std::setprecision(5) << (static_cast<double>(ns) / MILLA) << " milliseconds";
  }
  else if ( ns < MINUTE ) {
    out << std::setprecision(5) << (static_cast<double>(ns) / SECOND) << " seconds";
  }
  else if ( ns < HOUR ) {
    out << std::setprecision(5) << (static_cast<double>(ns) / MINUTE) << " minutes";
  }
  else {
    out << std::setprecision(5) << (static_cast<double>(ns) / HOUR) << " hours";
  }

  return out.str();
}

template <typename T>
void sum_min_max_impl( T const * in, T * inout, int len )
{
  const int size = len/3;
  for (int i=0; i<size; ++i) {
    inout[3*i+0] += in[3*i+0];
    inout[3*i+1]  = inout[3*i+1] < in[3*i+1] ? inout[3*i+1] : in[3*i+1] ;
    inout[3*i+2]  = in[3*i+2] < inout[3*i+2] ? inout[3*i+2] : in[3*i+2] ;
  }
}

extern "C" void sum_min_max( void * in, void * inout, int * len, MPI_Datatype * type )
{
 sum_min_max_impl( reinterpret_cast<int64_t*>(in), reinterpret_cast<int64_t*>(inout), *len );
}

MPI_Op sum_min_max_op;

} // unnamed namespace


void RuntimeStats::report( MPI_Comm comm, InfoStats & stats )
{
  {
    static bool init = false;
    if (!init) {
      MPI_Op_create( sum_min_max, true, &sum_min_max_op );

      register_timer_tag( TaskExecTag{}, "Task Exec", Max );
      register_timer_tag( TaskWaitTag{}, "Task Wait", Min );

      register_timer_tag( CollectiveTag{}, "Total Collective", Total);
      register_timer_tag( RecvTag{}, "Total Recv", Total);
      register_timer_tag( SendTag{}, "Total Send", Total);
      register_timer_tag( TestTag{}, "Total Test", Total);
      register_timer_tag( WaitTag{}, "Total Wait", Total);

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

  const int data_size = (num_timers + num_allocators) * 3;
  const int histogram_size = (num_timers + num_allocators) * 4;

  std::vector<int64_t>     local_times;
  std::vector<int64_t>     data;
  std::vector<int64_t>     histograms;
  std::vector<std::string> names;

  local_times.reserve( num_timers );
  data.reserve( data_size );

  if (prank==0) {
    names.reserve( num_timers + num_allocators );
  }

  for( auto const& p : s_timers ) {
    if (prank==0) {
      names.push_back( p.first );
    }
    const int64_t ns = p.second();
    local_times.push_back(ns);
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
      local_times.push_back(ns);
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
      local_times.push_back(ns);
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
    local_times.push_back(bytes);
    data.push_back(bytes); // total
    data.push_back(bytes); // min
    data.push_back(bytes); // max
  }

  if (g_print_histograms) {

    MPI_Allreduce( MPI_IN_PLACE, data.data(), data_size, MPI_INT64_T, sum_min_max_op, comm);

    histograms.resize( histogram_size, 0 );

    for( int i=0; i<num_timers; ++i ) {
      const int64_t min = data[3*i+1]; // min
      const int64_t mag = data[3*i+2] - min;  // max - min
      const int64_t t = local_times[i] - min;

      int bin = 0 < mag ? 4*t / mag : 0;
      bin = bin < 4 ? bin : 3;

      histograms[i*4+bin] = 1;
    }

    MPI_Reduce( MPI_IN_PLACE, histograms.data(), histogram_size, MPI_INT64_T, MPI_SUM, 0, comm);

  }
  else {
    MPI_Reduce( MPI_IN_PLACE, data.data(), data_size, MPI_INT64_T, sum_min_max_op, 0, comm);
  }


  if (prank == 0) {
    const int end = num_timers + num_allocators;

    for (int i=0; i<end; ++i) {

      std::string hist{};
      if (g_print_histograms) {
        std::ostringstream out;
        out  << ", Histogram : [ "
          << std::setw(5) << std::setprecision(1) << (100.0 * histograms[4*i+0]) / psize << " | "
          << std::setw(5) << std::setprecision(1) << (100.0 * histograms[4*i+1]) / psize << " | "
          << std::setw(5) << std::setprecision(1) << (100.0 * histograms[4*i+2]) / psize << " | "
          << std::setw(5) << std::setprecision(1) << (100.0 * histograms[4*i+3]) / psize
          << " ] ";
        hist = out.str();
      }

      if ( i < num_timers ) {
        DOUT( true, names[i] << " : "
                             << "Total[ " << nanoseconds_to_string( data[3*i+0] ) << " ], "
                             << "Min[ " << nanoseconds_to_string( data[3*i+1] ) << " ], "
                             << "Max[ " << nanoseconds_to_string( data[3*i+2] ) << " ], "
                             << hist
            );
      }
      else {
        DOUT( true, names[i] << " : "
                             << "Total[ " << bytes_to_string( data[3*i+0] ) << " ], "
                             << "Min[ " << bytes_to_string( data[3*i+1] ) << " ], "
                             << "Max[ " << bytes_to_string( data[3*i+2] ) << " ], "
                             << hist
            );
      }
    }
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
