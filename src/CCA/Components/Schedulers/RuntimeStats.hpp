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

#ifndef CCA_COMPONENTS_SCHEDULERS_RUNTIME_STATS_HPP
#define CCA_COMPONENTS_SCHEDULERS_RUNTIME_STATS_HPP

#include <Core/Util/Timers/Timers.hpp>
#include <Core/Util/InfoMapper.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Malloc/Allocators/TrackingAllocator.hpp>

#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <functional>


namespace Uintah {

class RuntimeStats
{
  static std::map< std::string, std::function<int64_t()> > s_allocators;
  static std::map< std::string, std::function<int64_t()> > s_timers;

  static std::vector< std::function<void()> >              s_reset_timers;

  static std::atomic<int64_t> * get_atomic_exec_ptr( DetailedTask const* t );
  static std::atomic<int64_t> * get_atomic_wait_ptr( DetailedTask const* t );

  static void reset_timers();
public:

  // used to declare timers
  template <typename Tag> using TripTimer = Timers::ThreadTrip< Tag >;

  using InfoStats = InfoMapper< SimulationState::RunTimeStat, double >;

  enum TimerType {Total, Min, Max};


  // NOT THREAD SAFE -- should only be called from the master thread
  template <typename Tag>
  static void register_timer_tag( Tag, std::string const& name, TimerType t )
  {
    auto const itr = s_timers.find(name);
    if (itr == s_timers.end()) {

      if (t == Total) {
        s_timers[ name ] = []()->int64_t { return TripTimer<Tag>::total_nanoseconds(); };
      } else if (t == Min) {
        s_timers[ name ] = []()->int64_t { return TripTimer<Tag>::min_nanoseconds(); };
      } else if (t == Max) {
        s_timers[ name ] = []()->int64_t { return TripTimer<Tag>::max_nanoseconds(); };
      }

      s_reset_timers.emplace_back( [](){ TripTimer<Tag>::reset_tag(); } );
    }
  }

  enum AllocatorType { Current, HighWater };

  // NOT THREAD SAFE -- should only be called from the master thread
  template <typename Tag>
  static void register_allocator_tag( Tag, std::string const& name, AllocatorType t )
  {
    auto const itr = s_allocators.find(name);
    if (itr == s_allocators.end()) {
      if (t == Current) {
        s_allocators[ name ] = []()->int64_t { return Allocators::TagStats< Tag >::alloc_size(); };
      } else if (t == HighWater) {
        s_allocators[ name ] = []()->int64_t { return Allocators::TagStats< Tag >::high_water(); };
      }
    }
  }

  // NOT THREAD SAFE -- should only be called from the master thread
  // by the parent scheduler
  static void initialize_timestep( std::vector<TaskGraph *> const & graphs );


  // NOT THREAD SAFE -- should only be called from the master thread
  // by the parent scheduler
  static void report( MPI_Comm comm, InfoStats & info_stats );

  struct TaskExecTag {};       // Total Task Exec
  struct TaskWaitTag {};       // Total Task Wait

  struct CollectiveTag {}; // Total Reduce
  struct RecvTag {};       // Total Recv
  struct SendTag {};       // Total Send
  struct TestTag {};       // Total Test
  struct WaitTag {};       // Total Wait
  struct CollectiveMPITag {}; // Total Reduce
  struct RecvMPITag {};       // Total Recv
  struct SendMPITag {};       // Total Send

  // RAII timer types

  using CollectiveTimer = TripTimer< CollectiveTag >;
  using RecvTimer       = TripTimer< RecvTag >;
  using SendTimer       = TripTimer< SendTag >;
  using TestTimer       = TripTimer< TestTag >;
  using WaitTimer       = TripTimer< WaitTag >;

  using CollectiveMPITimer = TripTimer< CollectiveMPITag >;
  using RecvMPITimer       = TripTimer< RecvMPITag >;
  using SendMPITimer       = TripTimer< SendMPITag >;

  using ExecTimer = TripTimer< TaskExecTag >;

  struct TaskExecTimer
    : public ExecTimer
  {
    template <typename... ExcludeTimers>
    TaskExecTimer( DetailedTask const* t, ExcludeTimers&... exclude_timers )
      : ExecTimer{ exclude_timers... }
    {
      m_task_time = get_atomic_exec_ptr(t);
    }

    ~TaskExecTimer()
    {
      if (m_task_time) {
        m_task_time->fetch_add( this->nanoseconds(), std::memory_order_relaxed );
      }
    }

  private:
    std::atomic<int64_t> * m_task_time{nullptr};
  };

  struct TaskWaitTimer
    : public TripTimer< TaskWaitTag >
  {
    template <typename... ExcludeTimers>
    TaskWaitTimer( DetailedTask const* t, ExcludeTimers&... exclude_timers )
      : TripTimer< TaskWaitTag >{ exclude_timers... }
    {
      m_task_time = get_atomic_wait_ptr(t);
    }

    ~TaskWaitTimer()
    {
      if (m_task_time) {
        m_task_time->fetch_add( this->nanoseconds(), std::memory_order_relaxed );
      }
    }

  private:
    std::atomic<int64_t> * m_task_time{nullptr};
  };

};

} // namespace Uintah

#endif //CCA_COMPONENTS_SCHEDULERS_RUNTIME_STATS_HPP
