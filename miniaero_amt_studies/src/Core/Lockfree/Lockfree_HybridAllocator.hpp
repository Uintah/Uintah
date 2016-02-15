#ifndef LOCKFREE_HYBRID_ALLOCATOR_HPP
#define LOCKFREE_HYBRID_ALLOCATOR_HPP

#include "impl/Lockfree_Macros.hpp"
#include <cstdlib>

namespace Lockfree {

template <  typename T
          , size_t CompareBytes
          , template <typename> class LEAllocator  // use if num_bytes < CompareBytes
          , template <typename> class GTAllocator  // use if num_bytes >= CompareBytes
         >
class HybridAllocator
{
  using le_allocator_type = LEAllocator<T>;
  using gt_allocator_type = GTAllocator<T>;

  static_assert( std::is_same< typename le_allocator_type::value_type, typename gt_allocator_type::value_type >::value
                , "ERROR: different value_types." );
  static_assert( std::is_same< typename le_allocator_type::pointer, typename gt_allocator_type::pointer >::value
                , "ERROR: different pointers." );
  static_assert( std::is_same< typename le_allocator_type::const_pointer, typename gt_allocator_type::const_pointer >::value
                , "ERROR: different const_pointers." );
  static_assert( std::is_same< typename le_allocator_type::reference, typename gt_allocator_type::reference >::value
                , "ERROR: different references." );
  static_assert( std::is_same< typename le_allocator_type::const_reference, typename gt_allocator_type::const_reference >::value
                , "ERROR: different const_references." );
  static_assert( std::is_same< typename le_allocator_type::size_type, typename gt_allocator_type::size_type >::value
                , "ERROR: different size_types." );

public:

  using size_type       = typename le_allocator_type::size_type;
  using difference_type = typename le_allocator_type::difference_type;
  using value_type      = typename le_allocator_type::value_type;
  using pointer         = typename le_allocator_type::pointer;
  using reference       = typename le_allocator_type::reference;
  using const_pointer   = typename le_allocator_type::const_pointer;
  using const_reference = typename le_allocator_type::const_reference;

  template <class U>
  struct rebind
  {
    using other = HybridAllocator<  U
                                  , CompareBytes
                                  , LEAllocator
                                  , GTAllocator
                                 >;
  };

  HybridAllocator()
    : le_allocator{}
    , gt_allocator{}
  {}

  HybridAllocator(  le_allocator_type arg_le_allocator
                  , gt_allocator_type arg_gt_allocator
                 )
    : le_allocator{ arg_le_allocator }
    , gt_allocator{ arg_gt_allocator }
  {}

  HybridAllocator( const HybridAllocator & rhs )
    : le_allocator{ rhs.le_allocator }
    , gt_allocator{ rhs.gt_allocator }
  {}

  HybridAllocator & operator=( const HybridAllocator & rhs )
  {
    le_allocator = rhs.le_allocator;
    gt_allocator = rhs.gt_allocator;
    return *this;
  }

  HybridAllocator( HybridAllocator && rhs )
    : le_allocator{ std::move( rhs.le_allocator ) }
    , gt_allocator{ std::move( rhs.gt_allocator ) }
  {}

  HybridAllocator & operator=( HybridAllocator && rhs )
  {
    le_allocator = std::move( rhs.le_allocator );
    gt_allocator = std::move( rhs.gt_allocator );
    return *this;
  }

  ~HybridAllocator() {}

  pointer address( reference x ) LOCKFREE_NOEXCEPT
  {
    return  le_allocator.address( x );
  }

  const_pointer address( const_reference x ) LOCKFREE_NOEXCEPT
  {
    return  le_allocator.address( x );
  }

  size_type max_size() const LOCKFREE_NOEXCEPT
  {
    return gt_allocator.max_size();
  }

  template <class U, class... Args>
  void construct (U* ptr, Args&&... args)
  {
    le_allocator.construct( ptr, std::forward<Args>(args)... );
  }

  template <class U>
  void destroy (U* ptr)
  {
    le_allocator.destroy( ptr );
  }

  pointer allocate( size_type n, void * hint = nullptr)
  {
    const size_type num_bytes = n * sizeof( value_type );
    pointer ptr = nullptr;
    if ( num_bytes < CompareBytes ) {
      ptr = le_allocator.allocate( n, hint );
    }
    else {
      ptr = gt_allocator.allocate( n, hint );
    }
    return ptr;
  }

  void deallocate( pointer ptr, size_type n )
  {
    const size_type num_bytes = n * sizeof( value_type );
    if ( num_bytes < CompareBytes ) {
      le_allocator.deallocate( ptr, n );
    }
    else {
      gt_allocator.deallocate( ptr, n );
    }
  }

private:
  le_allocator_type le_allocator;
  gt_allocator_type gt_allocator;
};

} // namespace Lockfree

#endif //LOCKFREE_HYBRID_ALLOCATOR_HPP
