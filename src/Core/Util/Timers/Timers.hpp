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

namespace Timers {

using clock_type  = std::chrono::high_resolution_clock;

struct ConvertTo{
static constexpr double microseconds( uint64_t nano )    { return nano * 1e-3; }
static constexpr double milliseconds( uint64_t nano )    { return nano * 1e-6; }
static constexpr double seconds( uint64_t nano )         { return nano * 1e-9; }
static constexpr double minutes( uint64_t nano )         { return nano * (1e-9/60.0); }
static constexpr double hours( uint64_t nano )           { return nano * (1e-9/3600.0); }
};

/// Simple
struct Simple
{

  Simple()
    : m_start{ clock_type::now() }
  {}

  // disable copy, assignment, and move
  Simple( const Simple & ) = delete;
  Simple & operator=( const Simple & ) = delete;
  Simple( Simple && ) = delete;
  Simple & operator=( Simple && ) = delete;

  /// reset the timer
  void reset() { m_start = clock_type::now(); }

  /// number of nanosecond since construction or reset
  uint64_t nanoseconds() const
  { return std::chrono::duration_cast<std::chrono::nanoseconds>( clock_type::now() - m_start ).count(); }

  double microseconds() const { return ConvertTo::microseconds(nanoseconds()); }
  double milliseconds() const { return ConvertTo::milliseconds(nanoseconds()); }
  double seconds()      const { return ConvertTo::seconds(nanoseconds()); }
  double minutes()      const { return ConvertTo::minutes(nanoseconds()); }
  double hours()        const { return ConvertTo::hours(nanoseconds()); }

  // member
  clock_type::time_point m_start;
};


/// Trip timer
///
/// RAII timer
template <typename Tag>
struct Trip
{
  using tag = Tag;

  Trip() = default;

  // disable copy, assignment, and move
  Trip( const Trip & ) = delete;
  Trip & operator=( const Trip & ) = delete;
  Trip( Trip && ) = delete;
  Trip & operator=( Trip && ) = delete;

  ~Trip()
  {
    const uint64_t tmp = m_simple.nanoseconds();
    s_total += tmp;
    s_min = s_min < tmp ? s_min : tmp;
    s_max = tmp < s_max ? s_max : tmp;
    s_trips += 1u;
  }

  static void reset()
  {

    s_total = 0u;
    s_min = std::numeric_limits<uint64_t>::max();
    s_max = 0u;
    s_trips = 0u;
  }

  /// number of completed trips
  static uint64_t trips() { return s_trips; }

  static uint64_t nanoseconds()  { return s_total; }
  static double   microseconds() { return ConvertTo::microseconds(nanoseconds()); }
  static double   milliseconds() { return ConvertTo::milliseconds(nanoseconds()); }
  static double   seconds()      { return ConvertTo::seconds(nanoseconds()); }
  static double   minutes()      { return ConvertTo::minutes(nanoseconds()); }
  static double   hours()        { return ConvertTo::hours(nanoseconds()); }

  static uint64_t min_nanoseconds()  { return s_min; }
  static double   min_microseconds() { return ConvertTo::microseconds(min_nanoseconds()); }
  static double   min_milliseconds() { return ConvertTo::milliseconds(min_nanoseconds()); }
  static double   min_seconds()      { return ConvertTo::seconds(min_nanoseconds()); }
  static double   min_minutes()      { return ConvertTo::minutes(min_nanoseconds()); }
  static double   min_hours()        { return ConvertTo::hours(min_nanoseconds()); }

  static uint64_t max_nanoseconds()  { return s_max; }
  static double   max_microseconds() { return ConvertTo::microseconds(max_nanoseconds()); }
  static double   max_milliseconds() { return ConvertTo::milliseconds(max_nanoseconds()); }
  static double   max_seconds()      { return ConvertTo::seconds(max_nanoseconds()); }
  static double   max_minutes()      { return ConvertTo::minutes(max_nanoseconds()); }
  static double   max_hours()        { return ConvertTo::hours(max_nanoseconds()); }

private:
  static uint64_t   s_trips;
  static uint64_t s_total;
  static uint64_t s_min;
  static uint64_t s_max;

  Simple m_simple{};
};

template <typename Tag> uint64_t Trip<Tag>::s_trips = 0;
template <typename Tag> uint64_t Trip<Tag>::s_total = 0;
template <typename Tag> uint64_t Trip<Tag>::s_min = std::numeric_limits<uint64_t>::max();
template <typename Tag> uint64_t Trip<Tag>::s_max = 0;


/// Atomic Trip timer
///
/// RAII timer
template <typename Tag>
struct AtomicTrip
{
  using tag = Tag;

  AtomicTrip() = default;

  // disable copy, assignment, and move
  AtomicTrip( const AtomicTrip & ) = delete;
  AtomicTrip & operator=( const AtomicTrip & ) = delete;
  AtomicTrip( AtomicTrip && ) = delete;
  AtomicTrip & operator=( AtomicTrip && ) = delete;

  ~AtomicTrip()
  {
    const uint64_t tmp = m_simple.nanoseconds();
    std::atomic_fetch_add_explicit( &s_total, tmp, std::memory_order_relaxed);

    uint64_t old;

    old = std::atomic_load_explicit( &s_min, std::memory_order_relaxed);
    while ( tmp < old && !std::atomic_compare_exchange_weak_explicit( &s_min, &old, tmp, std::memory_order_relaxed, std::memory_order_relaxed) ) {}

    old = std::atomic_load_explicit( &s_max, std::memory_order_relaxed);
    while ( s_max < tmp && !std::atomic_compare_exchange_weak_explicit( &s_max, &old, tmp, std::memory_order_relaxed, std::memory_order_relaxed) ) {}

    constexpr uint64_t one = 1u;
    std::atomic_fetch_add_explicit( &s_trips, one, std::memory_order_relaxed);
  }

  static void reset()
  {
    constexpr uint64_t zero = 0u;
    std::atomic_store_explicit( &s_total, zero, std::memory_order_relaxed);
    std::atomic_store_explicit( &s_min, std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);
    std::atomic_store_explicit( &s_max, zero, std::memory_order_relaxed);
    std::atomic_store_explicit( &s_trips, zero, std::memory_order_relaxed);
  }

  /// number of completed trips
  static uint64_t trips() { return std::atomic_load_explicit(&s_trips, std::memory_order_relaxed); }

  static uint64_t nanoseconds()  { return std::atomic_load_explicit(&s_total, std::memory_order_relaxed); }
  static double   microseconds() { return ConvertTo::microseconds(nanoseconds()); }
  static double   milliseconds() { return ConvertTo::milliseconds(nanoseconds()); }
  static double   seconds()      { return ConvertTo::seconds(nanoseconds()); }
  static double   minutes()      { return ConvertTo::minutes(nanoseconds()); }
  static double   hours()        { return ConvertTo::hours(nanoseconds()); }

  static uint64_t min_nanoseconds()  { return std::atomic_load_explicit(&s_min, std::memory_order_relaxed); }
  static double   min_microseconds() { return ConvertTo::microseconds(min_nanoseconds()); }
  static double   min_milliseconds() { return ConvertTo::milliseconds(min_nanoseconds()); }
  static double   min_seconds()      { return ConvertTo::seconds(min_nanoseconds()); }
  static double   min_minutes()      { return ConvertTo::minutes(min_nanoseconds()); }
  static double   min_hours()        { return ConvertTo::hours(min_nanoseconds()); }

  static uint64_t max_nanoseconds()  { return std::atomic_load_explicit(&s_max, std::memory_order_relaxed); }
  static double   max_microseconds() { return ConvertTo::microseconds(max_nanoseconds()); }
  static double   max_milliseconds() { return ConvertTo::milliseconds(max_nanoseconds()); }
  static double   max_seconds()      { return ConvertTo::seconds(max_nanoseconds()); }
  static double   max_minutes()      { return ConvertTo::minutes(max_nanoseconds()); }
  static double   max_hours()        { return ConvertTo::hours(max_nanoseconds()); }

private:
  static std::atomic<uint64_t> s_trips;
  static std::atomic<uint64_t> s_total;
  static std::atomic<uint64_t> s_min;
  static std::atomic<uint64_t> s_max;

  Simple m_simple{};
};

template <typename Tag> std::atomic<uint64_t> AtomicTrip<Tag>::s_trips{0u};
template <typename Tag> std::atomic<uint64_t> AtomicTrip<Tag>::s_total{0};
template <typename Tag> std::atomic<uint64_t> AtomicTrip<Tag>::s_min{std::numeric_limits<uint64_t>::max()};
template <typename Tag> std::atomic<uint64_t> AtomicTrip<Tag>::s_max{0};

} // namespace Timers

#endif //UTILITIES_TIMERS_HPP
