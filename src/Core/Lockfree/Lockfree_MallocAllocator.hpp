#ifndef LOCKFREE_MALLOC_ALLOCATOR_HPP
#define LOCKFREE_MALLOC_ALLOCATOR_HPP

#include "impl/Lockfree_Macros.hpp"
#include <cstdlib>

namespace Lockfree {

template <  typename T >
class MallocAllocator
{
public:

  using size_type       = size_t;
  using difference_type = ptrdiff_t;
  using value_type      = T;
  using pointer         = T*;
  using reference       = T&;
  using const_pointer   = const T*;
  using const_reference = const T&;

  template <class U>
  struct rebind
  {
    using other = MallocAllocator<U>;
  };

  MallocAllocator() {}
  MallocAllocator( const MallocAllocator & ) {}
  MallocAllocator & operator=( const MallocAllocator & ) { return *this; }

  static       pointer address(       reference x ) noexcept { return &x; }
  static const_pointer address( const_reference x ) noexcept { return &x; }

  static constexpr size_type max_size() { return std::numeric_limits<size_t>::max(); }

  template <class U, class... Args>
  static void construct (U* ptr, Args&&... args)
  {
    new ((void*)ptr) U( std::forward<Args>(args)... );
  }

  template <class U>
  static void destroy (U* ptr)
  {
    ptr->~U();
  }

  pointer allocate( size_type n, void * = nullptr)
  {
    return reinterpret_cast<pointer>(malloc( n * sizeof(value_type) ));
  }

  void deallocate( pointer ptr, size_type )
  {
    free( ptr );
  }

private:
};

} // namespace Lockfree

#endif //LOCKFREE_MALLOC_ALLOCATOR_HPP
