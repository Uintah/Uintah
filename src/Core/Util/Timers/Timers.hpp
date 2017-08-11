// The MIT License (MIT)
//
// Copyright (c) 1997-2017 The University of Utah
// Copyright (c) 2016 Daniel Sunderland
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef UTILITIES_TIMERS_HPP
#define UTILITIES_TIMERS_HPP

#include <chrono>
#include <limits>
#include <atomic>
#include <cstdint>
#include <vector>
#include <memory>

#include <ostream>

namespace Timers {

//------------------------------------------------------------------------------

struct nanoseconds
{
  // convert from int64_t
  constexpr nanoseconds() {};
  constexpr nanoseconds( nanoseconds const& ns ) : m_value{ns.m_value} {}
  constexpr nanoseconds( nanoseconds && ns ) : m_value{ns.m_value} {}
  nanoseconds & operator=( nanoseconds const& ns )
  {
    m_value=ns.m_value;
    return *this;
  }
  nanoseconds & operator=( nanoseconds && ns )
  {
    m_value=ns.m_value;
    return *this;
  }

  constexpr nanoseconds( int64_t ns )  : m_value{ns} {}
  constexpr nanoseconds( uint64_t ns ) : m_value{static_cast<int64_t>(ns)} {}
  constexpr nanoseconds( double ns )   : m_value{static_cast<int64_t>(ns)} {}

  constexpr operator int64_t()  const { return m_value; }

  explicit constexpr operator bool()     const { return m_value != 0; }
  explicit constexpr operator uint64_t() const { return static_cast<uint64_t>(m_value); }
  explicit constexpr operator double()   const { return static_cast<double>(m_value); }

  constexpr double microseconds() const { return m_value * 1e-3; }
  constexpr double milliseconds() const { return m_value * 1e-6; }
  constexpr double seconds()      const { return m_value * 1e-9; }
  constexpr double minutes()      const { return m_value * (1e-9/60.0); }
  constexpr double hours()        const { return m_value * (1e-9/3600.0); }


  friend std::ostream & operator<<(std::ostream & out, const nanoseconds & ns)
  {
    return out << ns.m_value << "ns";
  }

  int64_t m_value{};
};


//------------------------------------------------------------------------------

// Simple timer
struct Simple
{
  using clock_type  = std::chrono::high_resolution_clock;
  using time_point  = clock_type::time_point;

  Simple() = default;

  template <typename... ExcludeTimers>
  Simple( ExcludeTimers&... exclude_timers )
    : m_excludes{ sizeof...(ExcludeTimers) > 0 ? new Simple*[sizeof...(ExcludeTimers)+1]{} : nullptr }
  {
    set_excludes(0, std::addressof(exclude_timers)...);
  }

  ~Simple()
  {
    stop();
  }

  // disable copy, assignment
  Simple( const Simple & ) = delete;
  Simple( Simple && ) = delete;
  Simple & operator=( const Simple &timer ) {
    m_offset = timer.m_offset;
    return *this; };
  Simple & operator=( Simple && ) = delete;
  Simple & operator=( const nanoseconds &offset ) {
    m_offset = offset;
    return *this; };

  // reset the timer
  void reset( bool running )
  {
    m_start = m_finish = clock_type::now();
    m_offset = 0;
    m_excluded = 0;
    m_running = running;
  }

  // stop the timer
  bool stop()
  {
    if (m_running) {
      m_finish = clock_type::now();
      const int64_t t = std::chrono::duration_cast<std::chrono::nanoseconds>(m_finish-m_start).count();
      m_offset += t;
      exclude( t );
      m_running = false;
      return true;
    }
    return false;
  }

  // start or restart the timer
  void start()
  {
    m_start = m_finish = clock_type::now();
    m_running = true;
  }

  // lap the timer
  nanoseconds lap()
  {
    const int64_t tmp = m_offset;
    if (stop()) {
      start();
    }
    return m_offset - tmp;
  }

  // number of nanoseconds since construction, start, or reset
  nanoseconds operator()() const
  {
    return (m_running ? static_cast<int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>( clock_type::now()-m_start).count()) : 0)
      + m_offset
      - m_excluded;
  }

private:
  template <typename... ExcludeTimers>
  void set_excludes(int i, Simple* e, ExcludeTimers... exclude_timers)
  {
    m_excludes[i] = e;
    set_excludes(i+1, exclude_timers...);
  }

  void set_excludes(int i, Simple* e)
  {
    m_excludes[i] = e;
    m_excludes[i+1] = nullptr;
  }

  void set_excludes(int) {}

  void exclude(int64_t t)
  {
    if (m_excludes) {
      for( int i=0; m_excludes[i] != nullptr; ++i) {
        m_excludes[i]->m_excluded += t;
      }
    }
  }

  // member
  time_point                 m_start    {clock_type::now()};
  time_point                 m_finish   {m_start};
  int64_t                    m_offset   {0};
  int64_t                    m_excluded {0};
  std::unique_ptr<Simple*[]> m_excludes {};
  std::atomic<bool>          m_running  {false};
};


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

namespace Impl {

inline int tid()
{
  static std::atomic<int> count{0};
  const static thread_local int id = count.fetch_add( 1, std::memory_order_relaxed );
  return id;
}

}

// ThreadTrip timer
//
// RAII timer
template <typename Tag, int MaxThreads=64>
struct ThreadTrip : public Simple
{
  using tag = Tag;

  static constexpr int size = MaxThreads;
  ThreadTrip() = default;

  template <typename... ExcludeTimers>
  ThreadTrip( ExcludeTimers&... exclude_timers )
    : Simple{ exclude_timers... }
  {}

  // disable copy, assignment, and move
  ThreadTrip( const ThreadTrip & )             = delete;
  ThreadTrip & operator=( const ThreadTrip & ) = delete;
  ThreadTrip( ThreadTrip && )                  = delete;
  ThreadTrip & operator=( ThreadTrip && )      = delete;

  ~ThreadTrip()
  {
    stop();
  }

  bool stop()
  {
    if( Simple::stop()) {
      const int t = Impl::tid();
      s_total[ t % size ] += (*this)();
      s_used[  t % size ] = true;
      ++s_count[ t % size ];
      return true;
    }
    return false;
  }

  // NOT thread safe
  static void reset_tag()
  {
    for (int i=0; i<size; ++i) {
      s_total[i] = 0;
      s_count[i] = 0;
      s_used[i]  = false;
    }
  }

  // total time among all threads
  static nanoseconds total()
  {
    int64_t result = 0;
    for (int i=0; i<size; ++i) {
      result = (s_used[i] && result < s_total[i]) ? s_total[i] : result ;
    }
    return result;
  }

  // min time among all threads
  static nanoseconds min()
  {
    int64_t result = std::numeric_limits<int64_t>::max();
    for (int i=0; i<size; ++i) {
      result = (s_used[i] && s_total[i] < result) ? s_total[i] : result ;
    }
    return result != std::numeric_limits<int64_t>::max() ? result : 0 ;
  }

  // max time among all threads
  static nanoseconds max()
  {
    int64_t result = std::numeric_limits<int64_t>::min();
    for (int i=0; i<size; ++i) {
      result = (s_used[i] && result < s_total[i]) ? s_total[i] : result ;
    }
    return result != std::numeric_limits<int64_t>::min() ? result : 0 ;
  }

  // time given thread
  static nanoseconds thread(int t)  { return s_total[ t ]; }

  static int num_threads()
  {
    int result = 0;
    for (int i=0; i<size; ++i) {
      result += s_used[i] ? 1 : 0;
    }
    return result;
  }

  static int64_t count()
  {
    int64_t result = 0;
    for (int i=0; i<size; ++i) {
      result = (s_used[i] && result < s_count[i]) ? s_count[i] : result ;
    }
    return result;
  }

private:
  static int64_t s_total[size];
  static int64_t s_count[size];
  static bool     s_used[size];

};

template <typename Tag, int MaxThreads> int64_t ThreadTrip<Tag,MaxThreads>::s_total[size] = {};
template <typename Tag, int MaxThreads> int64_t ThreadTrip<Tag,MaxThreads>::s_count[size] = {};
template <typename Tag, int MaxThreads> bool    ThreadTrip<Tag,MaxThreads>::s_used[size]  = {};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

} // namespace Timers

#endif //UTILITIES_TIMERS_HPP
