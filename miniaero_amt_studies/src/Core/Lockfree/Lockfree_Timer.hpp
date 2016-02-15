#ifndef LOCKFREE_TIMER_HPP
#define LOCKFREE_TIMER_HPP

#include <chrono>

namespace Lockfree {

/// class Timer
///
/// provide simple wall clock timer
struct Timer
{
  using clock_type = std::chrono::high_resolution_clock;
  using nanoseconds = std::chrono::nanoseconds;

  Timer()
    : m_start{ clock_type::now() }
  {}

  void reset()
  {
    m_start = clock_type::now();
  }

  // number of seconds since construction or reset
  double elapsed() const
  {
    return std::chrono::duration_cast<nanoseconds>( clock_type::now() - m_start ).count() * 1.0e-9;
  }

  clock_type::time_point m_start;
};

} // namespace Lockfree

#endif // LOCKFREE_TIMER_HPP
