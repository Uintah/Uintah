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

#ifndef CCA_COMPONENTS_SCHEDULERS_RUNTIME_STATS_HPP
#define CCA_COMPONENTS_SCHEDULERS_RUNTIME_STATS_HPP

#include <Core/Grid/SimulationState.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/Timers/Timers.hpp>

#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>


namespace Uintah {

class DetailedTask;
class TaskGraph;

class RuntimeStats
{
  static std::atomic<int64_t> * get_atomic_exec_ptr( DetailedTask const* t );
  static std::atomic<int64_t> * get_atomic_wait_ptr( DetailedTask const* t );

public:

  enum ValueType { Count, Time, Memory };

  static void register_report( Dout const& dout
                             , std::string const & name
                             , ValueType type
                             , std::function<int64_t()> get_value
                             , std::function<void()> clear_value = [](){}
                             );

  // used to declare timers
  template <typename Tag> using TripTimer = Timers::ThreadTrip< Tag >;

  using InfoStats = InfoMapper< SimulationState::RunTimeStat, double >;

  // NOT THREAD SAFE -- should only be called from the master thread
  // by the parent scheduler
  static void initialize_timestep( std::vector<TaskGraph *> const & graphs );

  // NOT THREAD SAFE -- should only be called from the master thread
  // by the parent scheduler
  static void report( MPI_Comm comm );

  struct TaskExecTag {};       // Total Task Exec
  struct TaskWaitTag {};       // Total Task Wait

  struct CollectiveTag {}; // Total Reduce
  struct RecvTag {};       // Total Recv
  struct SendTag {};       // Total Send
  struct TestTag {};       // Total Test
  struct WaitTag {};       // Total Wait

  // RAII timer types
  using CollectiveTimer = TripTimer< CollectiveTag >;
  using RecvTimer       = TripTimer< RecvTag >;
  using SendTimer       = TripTimer< SendTag >;
  using TestTimer       = TripTimer< TestTag >;
  using WaitTimer       = TripTimer< WaitTag >;
  using ExecTimer       = TripTimer< TaskExecTag >;



  struct TaskExecTimer
    : public Timers::Simple
  {
    template <typename... ExcludeTimers>
    TaskExecTimer( DetailedTask const* t, ExcludeTimers&... exclude_timers )
      : Timers::Simple{ exclude_timers... }
      , m_task{t}
    {}

    ~TaskExecTimer()
    {
      stop();
    }

    bool stop()
    {
      if(Timers::Simple::stop()) {
        std::atomic<int64_t> * task_time = get_atomic_exec_ptr(m_task);
        if (task_time) {
          task_time->fetch_add( (*this)(), std::memory_order_relaxed );
        }
        return true;
      }
      return false;
    }

  private:
    DetailedTask const * m_task;
  };

  struct TaskWaitTimer
    : public Timers::Simple
  {
    template <typename... ExcludeTimers>
    TaskWaitTimer( DetailedTask const* t, ExcludeTimers&... exclude_timers )
      : Timers::Simple{ exclude_timers... }
      , m_task{t}
    {}

    ~TaskWaitTimer()
    {
      stop();
    }

    bool stop()
    {
      if(Timers::Simple::stop()) {
        std::atomic<int64_t> * task_time = get_atomic_wait_ptr(m_task);
        if (task_time) {
          task_time->fetch_add( (*this)(), std::memory_order_relaxed );
        }
        return true;
      }
      return false;
    }

  private:
    DetailedTask const * m_task;
  };

};

} // namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_RUNTIME_STATS_HPP
