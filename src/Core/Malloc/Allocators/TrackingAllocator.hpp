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

#ifndef UTILITIES_TRACKING_ALLOCATOR_HPP
#define UTILITIES_TRACKING_ALLOCATOR_HPP

#include <memory>   // for std::allocator
#include <iostream>
#include <atomic>
#include <utility>
#include <cstddef>


namespace Allocators {


template < typename Tag >
class TagStats
{
  static constexpr size_t one  = 1;
  static constexpr size_t zero = 0;

public:

  static constexpr size_t H0 = 64u;
  static constexpr size_t H1 = 4096u;
  static constexpr size_t H2 = H1 * 8u;

  enum {
     ALLOC_SIZE
   , ALLOC_NUM
   , DEALLOC_NUM
   , HIGH_WATER
   , HISTOGRAM_0
   , HISTOGRAM_1
   , HISTOGRAM_2
   , HISTOGRAM_3
   , SIZE
  };


  static void allocate( size_t n ) noexcept
  {
    const int i = n <= H0 ? 0 :
                  n <= H1 ? 1 :
                  n <= H2 ? 2 : 3;

    std::atomic_fetch_add_explicit( s_data + HISTOGRAM_0 + i, one, std::memory_order_relaxed );
    std::atomic_fetch_add_explicit( s_data + ALLOC_NUM, one, std::memory_order_relaxed );

    size_t cm = std::atomic_fetch_add_explicit( s_data + ALLOC_SIZE, n, std::memory_order_relaxed ) + n;
    size_t hw = std::atomic_load_explicit( s_data + HIGH_WATER, std::memory_order_relaxed );

    while ( hw < cm && ! std::atomic_compare_exchange_weak_explicit( s_data + HIGH_WATER, &hw, cm, std::memory_order_relaxed, std::memory_order_relaxed ) ) {}
  }

  static void deallocate( void *,  size_t n ) noexcept
  {
    std::atomic_fetch_add_explicit( s_data + DEALLOC_NUM, one, std::memory_order_relaxed );
    std::atomic_fetch_sub_explicit( s_data + ALLOC_SIZE, n, std::memory_order_relaxed );
  }

  static size_t alloc_size()  noexcept { return std::atomic_load_explicit( s_data + ALLOC_SIZE, std::memory_order_relaxed );  }
  static size_t num_alloc()   noexcept { return std::atomic_load_explicit( s_data + ALLOC_NUM, std::memory_order_relaxed );   }
  static size_t num_dealloc() noexcept { return std::atomic_load_explicit( s_data + DEALLOC_NUM, std::memory_order_relaxed ); }
  static size_t high_water()  noexcept { return std::atomic_load_explicit( s_data + HIGH_WATER, std::memory_order_relaxed );  }

  static size_t histogram( unsigned i ) noexcept
  {
    i = i < 4u ? i : 3u;
    return std::atomic_load_explicit( s_data + HISTOGRAM_0 + i, std::memory_order_relaxed );
  }

private:
  static std::atomic<size_t> s_data[SIZE];
};

// Declare linkage
template < typename Tag > std::atomic<size_t> TagStats<Tag>::s_data[SIZE] = {};


template <typename Tag>
std::ostream & operator<<(std::ostream & out, TagStats<Tag> const& stats)
{
  out << " alloc_size[ "  << stats.alloc_size()  << " ] ,";
  out << " high_water[ "  << stats.high_water()  << " ] ,";
  out << " num_alloc[ "   << stats.num_alloc()   << " ] ,";
  out << " num_dealloc[ " << stats.num_dealloc() << " ] : ";
  out << " H0[ " << stats.histogram(0) << " ]";
  out << " H1[ " << stats.histogram(1) << " ]";
  out << " H2[ " << stats.histogram(2) << " ]";
  out << " H3[ " << stats.histogram(3) << " ]";
  return out;
}



template <   typename T
           , typename Tag = void
           , template < typename > class BaseAllocator = std::allocator
         >
class TrackingAllocator
{
public:

  using base_allocator_type = BaseAllocator<T>;

  using tag = Tag;

  using size_type       = typename base_allocator_type::size_type;
  using difference_type = typename base_allocator_type::difference_type;
  using value_type      = typename base_allocator_type::value_type;
  using pointer         = typename base_allocator_type::pointer;
  using reference       = typename base_allocator_type::reference;
  using const_pointer   = typename base_allocator_type::const_pointer;
  using const_reference = typename base_allocator_type::const_reference;

  template <class U>
  struct rebind
  {
    using other = TrackingAllocator<  U
                                    , Tag
                                    , BaseAllocator
                                   >;
  };

  // forward the arguments to the base allocator
  template <typename... Args>
  explicit
  TrackingAllocator( Args&&... args )
    : m_base_allocator{ std::forward<Args>(args)... }
  {}

  explicit
  TrackingAllocator( base_allocator_type const& base )
    : m_base_allocator{ base }
  {}

  explicit
  TrackingAllocator( base_allocator_type && base )
    : m_base_allocator{ std::move(base) }
  {}

  TrackingAllocator & operator=( base_allocator_type const& base )
  {
    m_base_allocator = base;
    return *this;
  }

  TrackingAllocator & operator=( base_allocator_type && base )
  {
    m_base_allocator = std::move(base);
    return *this;
  }

  TrackingAllocator( TrackingAllocator const & rhs )
    : m_base_allocator{ rhs.m_base_allocator }
  {}

  TrackingAllocator & operator=( TrackingAllocator const& rhs )
  {
    m_base_allocator = rhs.m_base_allocator;
    return *this;
  }

  TrackingAllocator( TrackingAllocator & rhs )
    : m_base_allocator{ rhs.m_base_allocator }
  {}

  TrackingAllocator & operator=( TrackingAllocator & rhs )
  {
    m_base_allocator = rhs.m_base_allocator;
    return *this;
  }

  TrackingAllocator( TrackingAllocator && rhs )
    : m_base_allocator{ std::move( rhs.m_base_allocator ) }
  {}

  TrackingAllocator & operator=( TrackingAllocator && rhs )
  {
    m_base_allocator = std::move( rhs.m_base_allocator );
    return *this;
  }

  ~TrackingAllocator() {}

        pointer address(       reference x ) const noexcept { return m_base_allocator.address(x); }
  const_pointer address( const_reference x ) const noexcept { return m_base_allocator.address(x); }

  size_type max_size() const { return m_base_allocator.max_size(); }

  template <class U, class... Args>
  void construct (U* p, Args&&... args)
  {
    m_base_allocator.construct(p, std::forward<Args>(args)...);
  }

  template <class U>
  void destroy (U* p)
  {
    m_base_allocator.destroy(p);
  }

  pointer allocate( size_type n, void * hint = nullptr)
  {
    TagStats<tag>::allocate(n*sizeof(value_type));
    return m_base_allocator.allocate(n, hint);
  }

  void deallocate( pointer ptr, size_type n )
  {
    TagStats<tag>::deallocate(ptr, n*sizeof(value_type));
    m_base_allocator.deallocate(ptr, n);
  }

private:

  base_allocator_type m_base_allocator;
};

} // namespace Allocators

#endif // UTILITIES_TRACKING_ALLOCATOR_HPP
