#ifndef LOCKFREE_MMAP_ALLOCATOR_HPP
#define LOCKFREE_MMAP_ALLOCATOR_HPP

#include "impl/Lockfree_Macros.hpp"

#include <utility>
#include <cstdlib>

namespace Lockfree { namespace Impl {

void * mmap_allocate( size_t num_bytes );
void mmap_deallocate( void * ptr, size_t num_bytes );

}} //namespace Lockfree::Impl

namespace Lockfree {

template <  typename T >
class MMapAllocator
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
    using other = MMapAllocator<U>;
  };

  MMapAllocator() {}
  MMapAllocator( const MMapAllocator & ) {}
  MMapAllocator & operator=( const MMapAllocator & ) { return *this; }

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
    return reinterpret_cast<pointer>(Impl::mmap_allocate( n * sizeof(value_type) ));
  }

  void deallocate( pointer ptr, size_type n )
  {
    Impl::mmap_deallocate( ptr, n * sizeof(value_type) );
  }

private:
};

} // namespace Lockfree

#endif //LOCKFREE_MMAP_ALLOCATOR_HPP
