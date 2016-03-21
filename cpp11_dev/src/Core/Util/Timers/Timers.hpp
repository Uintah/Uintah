// The MIT License (MIT)
//
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
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>

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
  Simple & operator=( const Simple & ) = delete;
  Simple & operator=( Simple && ) = delete;


  /// reset the timer
  void reset()
  {
    m_start = m_finish = clock_type::now();
    m_offset = 0;
    m_excluded = 0;
  }

  // stop the timer
  bool stop()
  {
    if (m_finish <= m_start) {
      m_finish = clock_type::now();
      const int64_t t = std::chrono::duration_cast<std::chrono::nanoseconds>(m_finish-m_start).count();
      m_offset += t;
      exclude( t );
      return true;
    }
    return false;
  }

  // start or restart the timer
  void start()
  {
    m_start = m_finish = clock_type::now();
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
    return std::chrono::duration_cast<std::chrono::nanoseconds>( (m_start < m_finish) ?
             (m_finish - m_start) : (clock_type::now() - m_start) ).count()
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
};


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// ThreadTrip timer
//
// RAII timer
template <typename Tag>
struct ThreadTrip : public Simple
{
  using tag = Tag;

  ThreadTrip() = default;

  template <typename... ExcludeTimers>
  ThreadTrip( ExcludeTimers&... exclude_timers )
    : Simple{ exclude_timers... }
  {}

  // disable copy, assignment, and move
  ThreadTrip( const ThreadTrip & ) = delete;
  ThreadTrip & operator=( const ThreadTrip & ) = delete;
  ThreadTrip( ThreadTrip && ) = delete;
  ThreadTrip & operator=( ThreadTrip && ) = delete;

  ~ThreadTrip()
  {
    stop();
  }

  bool stop()
  {
    if( Simple::stop()) {
      t_node.m_value += Simple::operator()();
      return true;
    }
    return false;
  }

  // NOT thread safe
  static void reset_tag()
  {
    Node::apply( [](volatile int64_t & v) { v = 0; } );
  }

  // total time among all threads
  static nanoseconds total()
  {
    int64_t result = 0;
    Node::apply( [&result]( int64_t v ) { result += v; } );
    return result;
  }

  // min time among all threads
  static nanoseconds min()
  {
    int64_t result = std::numeric_limits<int64_t>::max();
    Node::apply( [&result]( int64_t v ) { result = result <  v ? result : v; } );
    return result != std::numeric_limits<int64_t>::max() ? result : 0 ;
  }

  // max time among all threads
  static nanoseconds max()
  {
    int64_t result = std::numeric_limits<int64_t>::min();
    Node::apply( [&result]( int64_t v ) { result = v <  result ? result : v; } );
    return result != std::numeric_limits<int64_t>::min() ? result : 0 ;
  }

  // time given thread
  static nanoseconds thread()  { return t_node.m_value; }

  static int num_threads()
  {
    int result = 0;
    Node::apply( [&result](int64_t) { ++result; });
    return result;
  }

private:

  struct Node
  {
    Node()
    {
      std::unique_lock<std::mutex> lock(s_lock);
      m_next = s_head;
      s_head = this;
    }


    template <typename Functor>
    static void apply( Functor && f )
    {
      Node * curr = s_head;

      while (curr) {
        f(curr->m_value);
        curr = curr->m_next;
      }
    }

    int64_t  m_value{0};
    Node   * m_next{nullptr};

    static Node *     s_head;
    static std::mutex s_lock;
  };

  static thread_local Node t_node;
};

template<typename Tag>
typename ThreadTrip<Tag>::Node * ThreadTrip<Tag>::Node::s_head{nullptr};

template<typename Tag>
std::mutex ThreadTrip<Tag>::Node::s_lock{};

template <typename Tag>
thread_local typename ThreadTrip<Tag>::Node ThreadTrip<Tag>::t_node{};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

} // namespace Timers

#endif //UTILITIES_TIMERS_HPP
