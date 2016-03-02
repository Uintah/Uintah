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
#include <atomic>
#include <cstdint>
#include <vector>
#include <memory>

namespace Timers {

//------------------------------------------------------------------------------


struct ConvertTo{
static constexpr double microseconds( int64_t nano ) { return nano * 1e-3; }
static constexpr double milliseconds( int64_t nano ) { return nano * 1e-6; }
static constexpr double seconds( int64_t nano )      { return nano * 1e-9; }
static constexpr double minutes( int64_t nano )      { return nano * (1e-9/60.0); }
static constexpr double hours( int64_t nano )        { return nano * (1e-9/3600.0); }
};

//------------------------------------------------------------------------------

// Simple timer
struct Simple
{
  using clock_type  = std::chrono::high_resolution_clock;

  Simple() = default;

  template <typename... ExcludeTimers>
  Simple( ExcludeTimers&... exclude_timers )
    : m_excludes{ std::addressof(exclude_timers)... }
  {}

  ~Simple()
  {
    exclude();
  }

  // disable copy, assignment
  Simple( const Simple & ) = delete;
  Simple( Simple && ) = delete;
  Simple & operator=( const Simple & ) = delete;
  Simple & operator=( Simple && ) = delete;

  /// reset the timer
  void reset()
  {
    m_start = clock_type::now();
    m_offset = 0;
  }

  /// reset the timer
  void exclude_and_reset()
  {
    exclude();
    m_start = clock_type::now();
    m_offset = 0;
  }

  /// number of nanosecond since construction or reset
  int64_t nanoseconds() const
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>( clock_type::now() - m_start ).count() - m_offset;
  }

  double microseconds() const { return ConvertTo::microseconds(nanoseconds()); }
  double milliseconds() const { return ConvertTo::milliseconds(nanoseconds()); }
  double seconds()      const { return ConvertTo::seconds(nanoseconds()); }
  double minutes()      const { return ConvertTo::minutes(nanoseconds()); }
  double hours()        const { return ConvertTo::hours(nanoseconds()); }

private:

  void exclude()
  {
    const int64_t t = nanoseconds();
    for (auto p : m_excludes) {
      p->m_offset += t;
    }
  }

  // member
  clock_type::time_point m_start{ clock_type::now() };
  int64_t                m_offset{0};
  std::vector<Simple *>  m_excludes{};
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Atomic Trip timer
//
// RAII timer
template <typename Tag>
struct Trip : public Simple
{
  using tag = Tag;

  static constexpr int64_t min_value = std::numeric_limits<int64_t>::min();
  static constexpr int64_t max_value = std::numeric_limits<int64_t>::max();
  static constexpr int64_t zero = 0;

  Trip() = default;

  template <typename... ExcludeTimers>
  Trip( ExcludeTimers&... exclude_timers )
    : Simple{ exclude_timers... }
  {}

  ~Trip()
  {
    const int64_t tmp = this->nanoseconds();
    s_total.fetch_add( tmp, std::memory_order_relaxed );

    int64_t old;

    old = s_min.load( std::memory_order_relaxed );
    while ( tmp < old && s_min.compare_exchange_weak( old, tmp, std::memory_order_relaxed, std::memory_order_relaxed) ) {}

    old = s_max.load( std::memory_order_relaxed );
    while ( old < tmp && !s_max.compare_exchange_weak( old, tmp, std::memory_order_relaxed, std::memory_order_relaxed) ) {}

    constexpr int64_t one = 1u;
    s_trips.fetch_add( one, std::memory_order_relaxed );
  }

  // disable copy, assignment, and move
  Trip( const Trip & ) = delete;
  Trip( Trip && ) = delete;
  Trip & operator=( const Trip & ) = delete;
  Trip & operator=( Trip && ) = delete;

  // thread safe
  static void reset_tag()
  {
    s_total.store( zero, std::memory_order_relaxed );
    s_min.store( max_value, std::memory_order_relaxed );
    s_max.store( min_value, std::memory_order_relaxed );
    s_trips.store( zero, std::memory_order_relaxed );
  }

  static int64_t trips() { return s_trips.load( std::memory_order_relaxed ); }

  static int64_t total_nanoseconds()  { return s_total.load( std::memory_order_relaxed ); }
  static double  total_microseconds() { return ConvertTo::microseconds(total_nanoseconds()); }
  static double  total_milliseconds() { return ConvertTo::milliseconds(total_nanoseconds()); }
  static double  total_seconds()      { return ConvertTo::seconds(total_nanoseconds()); }
  static double  total_minutes()      { return ConvertTo::minutes(total_nanoseconds()); }
  static double  total_hours()        { return ConvertTo::hours(total_nanoseconds()); }

  static int64_t min_nanoseconds()  { return s_min.load( std::memory_order_relaxed ); }
  static double  min_microseconds() { return ConvertTo::microseconds(min_nanoseconds()); }
  static double  min_milliseconds() { return ConvertTo::milliseconds(min_nanoseconds()); }
  static double  min_seconds()      { return ConvertTo::seconds(min_nanoseconds()); }
  static double  min_minutes()      { return ConvertTo::minutes(min_nanoseconds()); }
  static double  min_hours()        { return ConvertTo::hours(min_nanoseconds()); }

  static int64_t max_nanoseconds()  { return s_max.load( std::memory_order_relaxed ); }
  static double  max_microseconds() { return ConvertTo::microseconds(max_nanoseconds()); }
  static double  max_milliseconds() { return ConvertTo::milliseconds(max_nanoseconds()); }
  static double  max_seconds()      { return ConvertTo::seconds(max_nanoseconds()); }
  static double  max_minutes()      { return ConvertTo::minutes(max_nanoseconds()); }
  static double  max_hours()        { return ConvertTo::hours(max_nanoseconds()); }

private:
  static std::atomic<int64_t> s_trips;
  static std::atomic<int64_t> s_total;
  static std::atomic<int64_t> s_min;
  static std::atomic<int64_t> s_max;
};

template <typename Tag> std::atomic<int64_t> Trip<Tag>::s_trips{0u};
template <typename Tag> std::atomic<int64_t> Trip<Tag>::s_total{0};
template <typename Tag> std::atomic<int64_t> Trip<Tag>::s_min{std::numeric_limits<int64_t>::max()};
template <typename Tag> std::atomic<int64_t> Trip<Tag>::s_max{0};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

namespace Impl {

struct tid
{
  static int get()
  {
    static std::atomic<int> count{0};
    const static thread_local int id = count.fetch_add( 1, std::memory_order_relaxed );
    return id;
  }
};

}

// ThreadTrip timer
//
// RAII timer
template <typename Tag, int MaxThreads=512>
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
  ThreadTrip( const ThreadTrip & ) = delete;
  ThreadTrip & operator=( const ThreadTrip & ) = delete;
  ThreadTrip( ThreadTrip && ) = delete;
  ThreadTrip & operator=( ThreadTrip && ) = delete;

  ~ThreadTrip()
  {
    s_total[ Impl::tid::get() % size ] += this->nanoseconds();
    s_used[  Impl::tid::get() % size ] = true;
  }

  // NOT thread safe
  static void reset_tag()
  {
    for (int i=0; i<size; ++i) {
      s_total[i] = 0;
      s_used[i] = false;
    }
  }

  static int64_t total_nanoseconds()
  {
    int64_t result = 0;
    for (int i=0; i<size; ++i) {
      result = (s_used[i] && result < s_total[i]) ? s_total[i] : result ;
    }
    return result;
  }

  static double total_microseconds() { return ConvertTo::microseconds(total_nanoseconds()); }
  static double total_milliseconds() { return ConvertTo::milliseconds(total_nanoseconds()); }
  static double total_seconds()      { return ConvertTo::seconds(total_nanoseconds()); }
  static double total_minutes()      { return ConvertTo::minutes(total_nanoseconds()); }
  static double total_hours()        { return ConvertTo::hours(total_nanoseconds()); }

  static int64_t min_nanoseconds()
  {
    int64_t result = std::numeric_limits<int64_t>::max();
    for (int i=0; i<size; ++i) {
      result = (s_used[i] && s_total[i] < result) ? s_total[i] : result ;
    }

    return result != std::numeric_limits<int64_t>::max() ? result : 0 ;
  }
  static double min_microseconds() { return ConvertTo::microseconds(min_nanoseconds()); }
  static double min_milliseconds() { return ConvertTo::milliseconds(min_nanoseconds()); }
  static double min_seconds()      { return ConvertTo::seconds(min_nanoseconds()); }
  static double min_minutes()      { return ConvertTo::minutes(min_nanoseconds()); }
  static double min_hours()        { return ConvertTo::hours(min_nanoseconds()); }

  static int64_t max_nanoseconds()
  {
    int64_t result = std::numeric_limits<int64_t>::min();
    for (int i=0; i<size; ++i) {
      result = (s_used[i] && result < s_total[i]) ? s_total[i] : result ;
    }

    return result != std::numeric_limits<int64_t>::min() ? result : 0 ;
  }
  static double max_microseconds() { return ConvertTo::microseconds(max_nanoseconds()); }
  static double max_milliseconds() { return ConvertTo::milliseconds(max_nanoseconds()); }
  static double max_seconds()      { return ConvertTo::seconds(max_nanoseconds()); }
  static double max_minutes()      { return ConvertTo::minutes(max_nanoseconds()); }
  static double max_hours()        { return ConvertTo::hours(max_nanoseconds()); }

  static int threads()
  {
    int result = 0;
    for (int i=0; i<size; ++i) {
      result += s_used[i] ? 1 : 0;
    }
    return result;
  }

private:
  static int64_t s_total[size];
  static bool     s_used[size];
};

template <typename Tag, int MaxThreads> int64_t ThreadTrip<Tag,MaxThreads>::s_total[size] = {};
template <typename Tag, int MaxThreads> bool    ThreadTrip<Tag,MaxThreads>::s_used[size] = {};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

} // namespace Timers

#endif //UTILITIES_TIMERS_HPP
