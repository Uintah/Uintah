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
#include <Core/Util/Timers/Timers.hpp>

#include <iomanip>

#include <streambuf>
#include <iostream>
#include <sstream>

#include <algorithm>

namespace Uintah {

namespace {

Dout g_exec_times{"ExecTimes", false};
Dout g_wait_times{"WaitTimes", false};

size_t g_num_tasks{0};

std::vector< std::string > g_task_names;
std::unique_ptr< std::atomic<int64_t>[] > g_task_exec_times{nullptr};
std::unique_ptr< std::atomic<int64_t>[] > g_task_wait_times{nullptr};

size_t impl_get_global_id( DetailedTask const* t)
{
  auto const itr = std::lower_bound( g_task_names.begin(), g_task_names.end(), t->getTask()->getName() );
  return itr - g_task_names.begin();
}

} // unnamed namespace

std::map< std::string, std::function<int64_t()> > RuntimeStats::s_allocators{};
std::map< std::string, std::function<int64_t()> > RuntimeStats::s_timers{};
std::vector< std::function<void()> >              RuntimeStats::s_reset_timers{};

std::mutex RuntimeStats::s_register_mutex{};


std::atomic<int64_t> * RuntimeStats::get_atomic_exec_ptr( DetailedTask const* t)
{
  if (!g_exec_times) { return nullptr; }

  const size_t id = impl_get_global_id(t);

  return id < g_num_tasks ?
           & g_task_exec_times[ id ] : nullptr ;
}

std::atomic<int64_t> * RuntimeStats::get_atomic_wait_ptr( DetailedTask const* t)
{
  if (!g_wait_times) { return nullptr; }

  const size_t id = impl_get_global_id(t);

  return id < g_num_tasks ?
           & g_task_wait_times[ id ] : nullptr ;
}


void RuntimeStats::reset_timers()
{
  for (auto & f : s_reset_timers) { f(); }

  if (g_exec_times) {
    g_task_exec_times.reset();
  }

  if (g_wait_times) {
    g_task_exec_times.reset();
  }
}


void RuntimeStats::initialize_timestep( std::vector<TaskGraph *> const &  graphs )
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

  if ( g_exec_times || g_wait_times ) {
    std::set<std::string> task_names;
    for (auto const tg : graphs) {
      const int tg_size = tg->getNumTasks();
      for (int i=0; i < tg_size; ++i) {
        Task * t = tg->getTask(i);
        task_names.insert( t->getName() );
      }
    }

    g_task_names.clear();
    g_task_names.resize( task_names.size() );

    for (auto const & n : task_names) {
      g_task_names.push_back( n );
    }
    g_num_tasks = g_task_names.size();

    if (g_exec_times) {
      g_task_exec_times.reset();
      g_task_exec_times = std::unique_ptr< std::atomic<int64_t>[] >( new std::atomic<int64_t>[g_num_tasks]{} );
    }

    if (g_wait_times) {
      g_task_wait_times.reset();
      g_task_wait_times = std::unique_ptr< std::atomic<int64_t>[] >( new std::atomic<int64_t>[g_num_tasks]{} );
    }
  }
}



namespace {


/// convert a bytes to human readable string
inline std::string bytes_to_string( double bytes )
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
    out << std::setprecision(3) << (bytes / KB) << " KB";
  }
  else if ( bytes < GB ) {
    out << std::setprecision(3) << (bytes / MB) << " MB";
  }
  else if ( bytes < TB ) {
    out << std::setprecision(3) << (bytes / GB) << " GB";
  }
  else if ( bytes < PB ) {
    out << std::setprecision(3) << (bytes / TB) << " TB";
  }
  else {
    out << std::setprecision(3) << (bytes / PB) << " PB";
  }

  return out.str();
}

/// convert a nanoseconds to human readable string
inline std::string nanoseconds_to_string( double ns )
{
  std::ostringstream out;

  constexpr double to_milli = 1.0e-6;
  constexpr double to_sec   = 1.0e-9;

  if ( ns < 1.0e4) {
    out << std::setprecision(3) << ns << " ns";
  } else if ( ns < 1.0e8) {
    out << std::setprecision(3) << (ns * to_milli) << " ms";
  } else {
    out << std::setprecision(3) << (ns * to_sec) << " s ";
  }

  return out.str();
}

enum { RANK, SUM, MIN, MAX };
enum { Q1, Q2, Q3, Q4,  };


void rank_sum_min_max_impl( int64_t const * in, int64_t * inout, int len )
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


} // unnamed namespace


void RuntimeStats::report( MPI_Comm comm
                         , Counts const& counts
                         , InfoStats & stats
                         )
{
  {
    static bool init = false;
    if (!init) {
      MPI_Op_create( rank_sum_min_max, true, &rank_sum_min_max_op );
      init = true;
    }
  }

  enum class DataType { Counts
                      , Time
                      , Bytes
                      , ExecTime
                      , WaitTime
                      };


  int psize;
  int prank;

  MPI_Comm_size( comm, &psize );
  MPI_Comm_rank( comm, &prank );

  int num_timers = static_cast<int>(s_timers.size());

  if( g_exec_times ) {
    num_timers += (int)g_num_tasks;
  }

  if( g_wait_times ) {
    num_timers += (int)g_num_tasks;
  }

  const int num_allocators = static_cast<int>(s_allocators.size());

  const int num_values = num_timers + num_allocators + counts.size();; // num_patches, num_cells, num_particles

  const int data_size = num_values * 4;


  std::vector<int64_t>     local_data;
  std::vector<int64_t>     data;
  std::vector<int64_t>     histograms;
  std::vector<int64_t>     global_data;
  std::vector<int64_t>     global_histograms;
  std::vector< std::pair<std::string, DataType> > names;

  local_data.reserve( num_values );
  data.reserve( data_size );

  auto push_data = [&](const std::string & name, DataType t, int64_t value) {
    if (prank==0) {
      names.emplace_back( name, t );
    }
    local_data.push_back(value);
    data.push_back(prank); // rank
    data.push_back(value); // total
    data.push_back(value); // min
    data.push_back(value); // max
  };

  if (prank==0) {
    names.reserve( num_values );
  }

  // Counts
  for (auto const & c : counts ) {
    push_data(c.first, DataType::Counts, c.second);
  }

  // Tag Timers
  for( auto const& p : s_timers ) {
    push_data( p.first, DataType::Time, p.second() );
  }

  // Tag Allocators
  for ( auto const& p : s_allocators ) {
    push_data( p.first, DataType::Bytes, p.second() );
  }

  // Exec Times
  if (g_exec_times) {
    for (size_t i=0; i<g_num_tasks; ++i) {
      const int64_t ns = g_task_exec_times[i].load( std::memory_order_relaxed );
      push_data( g_task_names[i], DataType::ExecTime, ns );
    }
  }

  // Wait Times
  if (g_wait_times) {
    for (size_t i=0; i<g_num_tasks; ++i) {
      const int64_t ns = g_task_exec_times[i].load( std::memory_order_relaxed );
      push_data( g_task_names[i], DataType::WaitTime, ns );
    }
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

    const int w_desc = 14;
    const int w_num  = 13;
    const int w_hist = 35;
    const int w_load = 18;

    auto print_header = [=](DataType t) {
      printf("\n--------------------------------------------------------------------------------\n");
      switch(t) {
      case DataType::Counts:
        printf("COUNTS\n");
        break;
      case DataType::Time:
        printf("TIMERS (wall)\n");
        break;
      case DataType::Bytes:
        printf("ALLOCATORS\n");
        break;
      case DataType::ExecTime:
        printf("TASK EXEC TIME (compute)\n");
        break;
      case DataType::WaitTime:
        printf("TASK WAIT TIME (compute)\n");
        break;
      }
      printf("--------------------------------------------------------------------------------\n");
      printf( "%*s%*s%*s%*s%*s%*s%*s%*s\n"
          ,w_desc, "Name:"
          ,w_num,  "Total:"
          ,w_num,  "Avg:"
          ,w_num,  "Min:"
          ,w_num,  "Max:"
          ,w_num,  "Max rank:"
          ,w_hist, "Hist \% [  Q1 |  Q2 |  Q3 |  Q4 ]:"
          ,w_load, "\%Load Imbalance"
      );
    };

    auto to_string = [](DataType t, double v)->std::string {
      std::string result;
      switch(t) {
      case DataType::Counts:
        {
          std::ostringstream out;
          out << std::setprecision(3) << v;
          result = out.str();
        }
        break;
      case DataType::Time:
        result = nanoseconds_to_string(v);
        break;
      case DataType::Bytes:
        result = bytes_to_string(v);
        break;
      case DataType::ExecTime:
        result = nanoseconds_to_string(v);
        break;
      case DataType::WaitTime:
        result = nanoseconds_to_string(v);
        break;
      }
      return result;
    };

    auto print_row = [&](int i) {
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

      const std::string & name = names[i].first;
      DataType t =  names[i].second;

      if ( total > 0) {
        if (name.size() < w_desc) {
          printf("%*s:%*s%*s%*s%*s%*d          %6.1f%6.1f%6.1f%6.1f%8.1f\n"
              , w_desc-1, name.c_str()
              , w_num, to_string(t, total).c_str()
              , w_num, to_string(t, avg).c_str()
              , w_num, to_string(t, min).c_str()
              , w_num, to_string(t, max).c_str()
              , w_num, max_rank
              , q1
              , q2
              , q3
              , q4
              , load
              );
        }
        else {
          printf("%s\n", name.c_str());
          printf("%*s:%*s%*s%*s%*s%*d          %6.1f%6.1f%6.1f%6.1f%8.1f\n"
              , w_desc-1, ""
              , w_num, to_string(t, total).c_str()
              , w_num, to_string(t, avg).c_str()
              , w_num, to_string(t, min).c_str()
              , w_num, to_string(t, max).c_str()
              , w_num, max_rank
              , q1
              , q2
              , q3
              , q4
              , load
              );

        }
      }
    };

    for (int i=0; i<num_values; ++i) {
      if (i==0 || names[i-1].second != names[i].second) {
        print_header( names[i].second );
      }
      print_row(i);
    }
    printf("\n");
  }



  // update SimulationState
  stats[SimulationState::TaskExecTime]       += TripTimer< TaskExecTag >::max().seconds()
                                                - stats[SimulationState::OutputFileIOTime]  // don't count output time or bytes
                                                - stats[SimulationState::OutputFileIORate];
  stats[SimulationState::TaskLocalCommTime]  += RecvTimer::total().seconds() + SendTimer::total().seconds();
  stats[SimulationState::TaskWaitCommTime]   += TestTimer::total().seconds() + WaitTimer::total().seconds();
  stats[SimulationState::TaskGlobalCommTime] += CollectiveTimer::total().seconds();

  reset_timers();
}

} // namespace Uintah
