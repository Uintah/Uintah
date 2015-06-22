#ifndef LOCKFREE_TRACKING_ALLOCATOR_HPP
#define LOCKFREE_TRACKING_ALLOCATOR_HPP

#include "impl/Lockfree_Macros.hpp"
#include "impl/Lockfree_IOHelpers.hpp"

#include <memory>   // for std::allocator
#include <iostream>


namespace Lockfree {


template < typename Tag >
class TagStats
{
  static constexpr size_t one  = 1;
  static constexpr size_t zero = 0;

public:

  enum {
     H0 = 64ull
   , H1 = 4096ull
   , H2 = 8ull*H1
  };

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


  static void allocate( size_t n )
  {
    // compute histogram
    if ( n <= H0 ) {
      __sync_fetch_and_add( s_data + HISTOGRAM_0, one );
    }
    else if ( n <= H1 ) {
      __sync_fetch_and_add( s_data + HISTOGRAM_1, one );
    }
    else if ( n <= H2 ) {
      __sync_fetch_and_add( s_data + HISTOGRAM_2, one );
    }
    else {
      __sync_fetch_and_add( s_data + HISTOGRAM_3, one );
    }

    __sync_fetch_and_add(s_data + ALLOC_NUM, one);
    size_t current_mem = __sync_add_and_fetch(s_data + ALLOC_SIZE, n);

    // adjust highwater
    size_t old_high_water = zero;
    do {
      old_high_water = __sync_fetch_and_add(s_data + HIGH_WATER, zero);
      if (current_mem > old_high_water) {
        bool success =  __sync_bool_compare_and_swap(s_data + HIGH_WATER, old_high_water, current_mem);
        if (success) {
          old_high_water = current_mem;
        }
      }
    } while (current_mem > old_high_water);
  }

  static void deallocate( void *,  size_t n )
  {
    __sync_fetch_and_add(s_data + DEALLOC_NUM, one);
    __sync_fetch_and_sub(s_data + ALLOC_SIZE, n);
  }

  static uint64_t alloc_size()  { return s_data[ALLOC_SIZE];  }
  static uint64_t num_alloc()   { return s_data[ALLOC_NUM];   }
  static uint64_t num_dealloc() { return s_data[DEALLOC_NUM]; }
  static uint64_t high_water()  { return s_data[HIGH_WATER];  }

  static uint64_t histogram( unsigned i )
  {
    i = i < 4 ? i : 3;
    return *(s_data + HISTOGRAM_0 + i);
  }

  static volatile uint64_t const * const data() { return s_data; }

private:

  static volatile uint64_t s_data[SIZE];
};

// Declare linkage
template < typename Tag > volatile uint64_t TagStats<Tag>::s_data[SIZE] = {};


template <typename Tag>
std::ostream & operator<<(std::ostream & out, TagStats<Tag> const& stats)
{
  out << Tag::name() << " :";
  out << " alloc_size[ "  << Impl::bytes_to_string( stats.alloc_size()  ) << " ] ,";
  out << " high_water[ "  << Impl::bytes_to_string( stats.high_water()  ) << " ] ,";
  out << " num_alloc[ "   << stats.num_alloc()   << " ] ,";
  out << " num_dealloc[ " << stats.num_dealloc() << " ] : ";
  out << " H0[ " << stats.histogram(0) << " ]";
  out << " H1[ " << stats.histogram(1) << " ]";
  out << " H2[ " << stats.histogram(2) << " ]";
  out << " H3[ " << stats.histogram(3) << " ]";
  return out;
}



template <   typename T
           , typename Tag
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

  TrackingAllocator( base_allocator_type arg_base_allocator = base_allocator_type{} )
    : m_base_allocator{ arg_base_allocator }
  {}

  TrackingAllocator( const TrackingAllocator & rhs )
    : m_base_allocator{ rhs.m_base_allocator }
  {}

  TrackingAllocator & operator=( const TrackingAllocator & rhs )
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

  // Let other allocators determine const correctness
  base_allocator_type m_base_allocator;

};

} // namespace Lockfree

#endif // LOCKFREE_TRACKING_ALLOCATOR_HPP
