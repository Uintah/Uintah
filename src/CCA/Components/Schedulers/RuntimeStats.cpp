/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/TaskGraph.h>

#include <Core/Parallel/MasterLock.h>
#include <Core/Util/DOUT.hpp>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <streambuf>


namespace Uintah {

namespace {

  Dout  mpi_stats( "MPIStats",  "Schedulers", "Comprehensive, fine-grained MPI summary each timestep",  false);
  Dout exec_times( "ExecTimes", "Schedulers", "Execution time for each task", false);
  Dout wait_times( "WaitTimes", "Schedulers", "Detailed summary of task wait times", false);
  Dout task_stats( "TaskStats", "Schedulers", "Runtime Task Stats", false);

  struct ReportValue
  {
    ReportValue()                     = default;
    ReportValue( const ReportValue &) = default;
    ReportValue( ReportValue &&)      = default;
    
    ReportValue & operator=( const ReportValue & ) = default;
    ReportValue & operator=( ReportValue && )      = default;
    
    ReportValue( RuntimeStats::ValueType  type
		 , std::function<int64_t()> get
		 , std::function<void()>    clear = [](){}
		 )
      : m_type{type}
      , m_get{get}
      , m_clear{clear}
    {}

    RuntimeStats::ValueType  m_type{};
    std::function<int64_t()> m_get{};
    std::function<void()>    m_clear{};
    int m_index{-1};
  };

  Uintah::MasterLock g_report_lock{};

  // [Dout][value_name] = ReportValue
  std::map< Dout, std::map< std::string, ReportValue> > g_report_values;

  size_t g_num_tasks;
  std::vector< std::string > g_task_names;
  std::unique_ptr< std::atomic<int64_t>[] > g_task_exec_times{nullptr};
  std::unique_ptr< std::atomic<int64_t>[] > g_task_wait_times{nullptr};

  size_t impl_get_global_id( DetailedTask const* t)
  {
    auto const itr = std::lower_bound( g_task_names.begin(), g_task_names.end(), t->getTask()->getName() );
    return itr - g_task_names.begin();
  }

} // namespace



void RuntimeStats::register_report( Dout const& dout
                                  , std::string const& name
                                  , ValueType type
                                  , std::function<int64_t()> get_value
                                  , std::function<void()>    clear_value
                                  )
{
  if (mpi_stats || exec_times || wait_times || task_stats) {
    std::unique_lock<Uintah::MasterLock> lock(g_report_lock);
    ReportValue value { type, get_value, clear_value };
    g_report_values[dout][name] = value;
  }
}


std::atomic<int64_t> * RuntimeStats::get_atomic_exec_ptr( DetailedTask const* t)
{
  if (exec_times) {
    const size_t id = impl_get_global_id(t);
    return id < g_num_tasks ? & g_task_exec_times[ id ] : nullptr ;
  }
  return nullptr;
}

std::atomic<int64_t> * RuntimeStats::get_atomic_wait_ptr( DetailedTask const* t)
{
  if (wait_times) {
    const size_t id = impl_get_global_id(t);
    return id < g_num_tasks ? & g_task_wait_times[ id ] : nullptr ;
  }
  return nullptr;
}

void RuntimeStats::initialize_timestep( std::vector<TaskGraph *> const &  graphs )
{
  if (exec_times || wait_times || task_stats) {

    std::unique_lock<Uintah::MasterLock> lock(g_report_lock);

    std::set<std::string> task_names;
    for (auto const tg : graphs) {
      const int tg_size = tg->getNumTasks();
      for (int i=0; i < tg_size; ++i) {
        Task * t = tg->getTask(i);
        task_names.insert( t->getName() );
      }
    }

    g_task_names.clear();
    g_task_names.insert( g_task_names.begin(), task_names.begin(), task_names.end() );

    g_num_tasks = g_task_names.size();

    if (exec_times) {
      g_task_exec_times.reset();
      g_task_exec_times = std::unique_ptr< std::atomic<int64_t>[] >( new std::atomic<int64_t>[g_num_tasks]{} );

      auto & exec_time_report = g_report_values[exec_times];
      exec_time_report.clear();

      for (size_t i=0; i<g_num_tasks; ++i) {
        exec_time_report[g_task_names[i]] = ReportValue{ RuntimeStats::Time
                                                       , [i]() { return g_task_exec_times[i].load( std::memory_order_relaxed ); }
                                                       };
      }
    }

    if (wait_times) {
      g_task_wait_times.reset();
      g_task_wait_times = std::unique_ptr< std::atomic<int64_t>[] >( new std::atomic<int64_t>[g_num_tasks]{} );

      auto & wait_time_report = g_report_values[wait_times];
      wait_time_report.clear();

      for (size_t i=0; i<g_num_tasks; ++i) {
        wait_time_report[g_task_names[i]] = ReportValue{ RuntimeStats::Time
                                                       , [i]() { return g_task_wait_times[i].load( std::memory_order_relaxed ); }
                                                       };
      }
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
    inout[off+SUM] += in[off+SUM] ;                                                   // sum
    inout[off+MIN]  = inout[off+MIN] < in[off+MIN] ? inout[off+MIN] : in[off+MIN] ;   // min
    inout[off+MAX]  = in[off+MAX] < inout[off+MAX] ? inout[off+MAX] : in[off+MAX] ;   // max
  }
}

extern "C" void rank_sum_min_max( void * in, void * inout, int * len, MPI_Datatype * type )
{
 rank_sum_min_max_impl( reinterpret_cast<int64_t*>(in), reinterpret_cast<int64_t*>(inout), *len );
}

MPI_Op rank_sum_min_max_op;


} // unnamed namespace


void RuntimeStats::report( MPI_Comm comm )
{
  if (!(mpi_stats || exec_times || wait_times || task_stats)) {
    return;
  }

  {
    static bool init = false;
    if (!init) {
      MPI::Op_create( rank_sum_min_max, true, &rank_sum_min_max_op );
      init = true;
    }
  }

  if (mpi_stats) {

    register_report( mpi_stats
                   , "Time: Comm"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::CommTimer::max(); }
                   , []() { MPI::Impl::CommTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Comm"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::CommTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Scan"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::ScanTimer::max(); }
                   , []() { MPI::Impl::ScanTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Scan"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::ScanTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Redu"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::ReduceTimer::max(); }
                   , []() { MPI::Impl::ReduceTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Redu"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::ReduceTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Scat"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::ScatterTimer::max(); }
                   , []() { MPI::Impl::ScatterTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Scat"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::ScatterTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Gath"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::GatherTimer::max(); }
                   , []() { MPI::Impl::GatherTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Gath"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::GatherTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Bcas"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::BcastTimer::max(); }
                   , []() { MPI::Impl::BcastTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Bcas"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::BcastTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: All2"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::AlltoallTimer::max(); }
                   , []() { MPI::Impl::AlltoallTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: All2"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::AlltoallTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Send"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::SendTimer::max(); }
                   , []() { MPI::Impl::SendTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Send"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::SendTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Recv"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::RecvTimer::max(); }
                   , []() { MPI::Impl::RecvTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Recv"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::RecvTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Wait"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::WaitTimer::min(); }
                   , []() { MPI::Impl::WaitTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Wait"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::WaitTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Time: Test"
                   , RuntimeStats::Time
                   , []() { return MPI::Impl::TestTimer::min(); }
                   , []() { MPI::Impl::TestTimer::reset_tag(); }
                   );
    register_report( mpi_stats
                   , "Count: Test"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::TestTimer::count(); }
                   );
    register_report( mpi_stats
                   , "Volume Send TOTAL   "
                   , RuntimeStats::Memory
                   , []() { return MPI::Impl::SendVolumeStats::get(MPI::Impl::COMM_SIZE); }
                   , []() { return MPI::Impl::SendVolumeStats::clear(); }
                   );
    register_report( mpi_stats
                   , "Volume Send0: <= 64B"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::SendVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_0); }
                   );
    register_report( mpi_stats
                   , "Volume Send1: <= 4KB"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::SendVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_1); }
                   );
    register_report( mpi_stats
                   , "Volume Send2: <= 2MB"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::SendVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_2); }
                   );
    register_report( mpi_stats
                   , "Volume Send3: >  2MB"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::SendVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_3); }
                   );
    register_report( mpi_stats
                   , "Volume Recv TOTAL   "
                   , RuntimeStats::Memory
                   , []() { return MPI::Impl::RecvVolumeStats::get(MPI::Impl::COMM_SIZE); }
                   , []() { return MPI::Impl::RecvVolumeStats::clear(); }
                   );
    register_report( mpi_stats
                   , "Volume Recv0: <= 64B"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::RecvVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_0); }
                   );
    register_report( mpi_stats
                   , "Volume Recv1: <= 4KB"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::RecvVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_1); }
                   );
    register_report( mpi_stats
                   , "Volume Recv2: <= 2MB"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::RecvVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_2); }
                   );
    register_report( mpi_stats
                   , "Volume Recv3: >  2MB"
                   , RuntimeStats::Count
                   , []() { return MPI::Impl::RecvVolumeStats::get(MPI::Impl::COMM_HISTOGRAM_3); }
                   );
  }

  if (task_stats) {

    register_report( task_stats
                   , "Coll"
                   , RuntimeStats::Time
                   , []() { return CollectiveTimer::max(); }
                   , []() { CollectiveTimer::reset_tag(); }
                   );
    register_report( task_stats
                   , "Send"
                   , RuntimeStats::Time
                   , []() { return SendTimer::max(); }
                   , []() { SendTimer::reset_tag(); }
                   );
    register_report( task_stats
                   , "Recv"
                   , RuntimeStats::Time
                   , []() { return RecvTimer::max(); }
                   , []() { RecvTimer::reset_tag(); }
                   );
    register_report( task_stats
                   , "Test"
                   , RuntimeStats::Time
                   , []() { return TestTimer::max(); }
                   , []() { TestTimer::reset_tag(); }
                   );
    register_report( task_stats
                   , "Wait"
                   , RuntimeStats::Time
                   , []() { return WaitTimer::max(); }
                   , []() { WaitTimer::reset_tag(); }
                   );
    register_report( task_stats
                   , "Exec"
                   , RuntimeStats::Time
                   , []() { return ExecTimer::max(); }
                   , []() { ExecTimer::reset_tag(); }
                   );
  }

  std::unique_lock<Uintah::MasterLock> lock(g_report_lock);

  int psize;
  int prank;

  MPI::Comm_size( comm, &psize );
  MPI::Comm_rank( comm, &prank );

  int  num_report_values = 0;
  // count
  {
    int i = 0;
    for (auto & group : g_report_values) {

      // group.first Dout, group.second map< string, ReportValue>
      num_report_values += static_cast<int>(group.second.size());

      for (auto & value : group.second ) {
        value.second.m_index = i++;
      }
    }
  }

  const int data_size = num_report_values*4;

  std::vector<int64_t>     data(data_size,0);
  std::vector<int64_t>     global_data(data_size,0);

  // fill
  {
    for (auto & group : g_report_values) {
      // group.first Dout, group.second map< string, ReportValue>
      for (auto & value : group.second ) {
        int64_t i = 4 * value.second.m_index;
        const int64_t v = value.second.m_get();
        data[i+RANK] = prank; // rank
        data[i+SUM] = v;      // total
        data[i+MIN] = v;      // min
        data[i+MAX] = v;      // max
      }
    }
  }

  global_data.resize(data_size);
  MPI::Allreduce( data.data(), global_data.data(), data_size, MPI_INT64_T, rank_sum_min_max_op, comm);

  std::vector<int64_t>     histograms(data_size,0);
  std::vector<int64_t>     global_histograms(data_size,0);

  // histogram
  {
    for (auto & group : g_report_values) {
      // group.first Dout, group.second map< string, ReportValue>
      for (auto & value : group.second ) {
        const int64_t i = 4 * value.second.m_index;
        const int64_t min = global_data[i+MIN];
        const int64_t max = global_data[i+MAX];
        const int64_t mag = max - min;
        const int64_t t =   data[i+SUM] - min;

        int bin = (0 < mag) ? 4*t / mag : 0;
        bin = bin < 4 ? bin : 3;
        histograms[i+bin] = 1;

        // clear the value to avoid another loop
        value.second.m_clear();
      }
    }
  }

  MPI::Reduce(histograms.data(), global_histograms.data(), data_size, MPI_INT64_T, MPI_SUM, 0, comm);

  if (prank == 0) {

    auto to_string = [](ValueType t, double v)->std::string {
      std::string result;
      switch(t) {
      case Count:
        {
          std::ostringstream out;
          out << std::setprecision(3) << v;
          result = out.str();
        }
        break;
      case Time:
        result = nanoseconds_to_string(v);
        break;
      case Memory:
        result = bytes_to_string(v);
        break;
      }
      return result;
    };

    const int w_num  = 13;
    const int w_hist = 35;
    const int w_load = 18;

    printf("\n________________________________________________________________________________");
    for (auto const& group : g_report_values) {

      int w_desc = 14;
      for (auto const& value : group.second ) {
        const int s = static_cast<int>(value.first.size()) + 1;
        w_desc = s < w_desc ? w_desc : s;
      }

      const std::string & group_name = group.first.name() +
                                       (group.first.active() ? ":+" : ":-");
      
      printf("\n--------------------------------------------------------------------------------\n");
      printf("%s\n", group_name.c_str() );
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

      for (auto const& value : group.second) {

        const int64_t i = 4 * value.second.m_index;

        const int max_rank  = global_data[i+RANK];
        const int64_t total = global_data[i+SUM];
        const int64_t min   = global_data[i+MIN];
        const int64_t max   = global_data[i+MAX];
        const double avg = static_cast<double>(total) / psize;

        const double q1 = static_cast<double>(100 * global_histograms[i+Q1]) / psize;
        const double q2 = static_cast<double>(100 * global_histograms[i+Q2]) / psize;
        const double q3 = static_cast<double>(100 * global_histograms[i+Q3]) / psize;
        const double q4 = static_cast<double>(100 * global_histograms[i+Q4]) / psize;

        const double load = max != 0 ? (100.0 * (1.0 - (avg / max))) : 0.0;

        const std::string & value_name = value.first;

        const RuntimeStats::ValueType t = value.second.m_type;

        if (total > 0) {
          printf("%*s:%*s%*s%*s%*s%*d          %6.1f%6.1f%6.1f%6.1f%8.1f\n"
              , w_desc-1, value_name.c_str()
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
    }

    printf("\n");
  }

  // clear the registered report values
  g_report_values.clear();
}

} // namespace Uintah
