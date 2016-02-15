#ifndef LOCKFREE_POOL_ALLOCATOR_HPP
#define LOCKFREE_POOL_ALLOCATOR_HPP

#include "impl/Lockfree_Macros.hpp"

#include "Lockfree_CircularPool.hpp"

#include <new> // for bad_alloc

namespace Lockfree {

template <  typename T
          , template <typename> class BaseAllocator = std::allocator
          , template <typename> class SizeTypeAllocator = BaseAllocator
        >
class PoolAllocator
{

  struct LOCKFREE_ALIGNAS( LOCKFREE_ALIGNOF(T) ) Node
  {
    static constexpr size_t alignment = LOCKFREE_ALIGNOF(T) ? LOCKFREE_ALIGNOF(T) : 16;
    static constexpr size_t buffer_size =  (alignment * ((sizeof(T) + sizeof(void *) - 1u + alignment) / alignment)) - sizeof(void*);

    char   m_buffer[buffer_size];
    void * m_circular_pool_node;
  };

public:
  using internal_circular_pool_type = CircularPool<  Node
                                                   , DISABLE_SIZE
                                                   , SHARED_INSTANCE
                                                   , BaseAllocator
                                                   , SizeTypeAllocator
                                                  >;

private:
  using circular_pool_node_type = typename internal_circular_pool_type::impl_node_type;

public:

  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;

  template <class U>
  struct rebind
  {
    using other = PoolAllocator<   U
                                 , BaseAllocator
                                 , SizeTypeAllocator
                               >;
  };

  PoolAllocator()
    : m_circular_pool{}
  {}

  PoolAllocator( const PoolAllocator & rhs )
    : m_circular_pool{ rhs.m_circular_pool }
  {}

  PoolAllocator & operator=( const PoolAllocator & rhs )
  {
    m_circular_pool = rhs.m_circular_pool;
    return *this;
  }

  PoolAllocator( PoolAllocator && rhs )
    : m_circular_pool{ std::move( rhs.m_circular_pool ) }
  {}

  PoolAllocator & operator=( PoolAllocator && rhs )
  {
    m_circular_pool = std::move( rhs.m_circular_pool );
    return *this;
  }

  ~PoolAllocator() {}

  static       pointer address(       reference x ) LOCKFREE_NOEXCEPT { return &x; }
  static const_pointer address( const_reference x ) LOCKFREE_NOEXCEPT { return &x; }

  static constexpr size_type max_size() { return 1u; }

  template <class U, class... Args>
  static void construct (U* p, Args&&... args)
  {
    new ((void*)p) U( std::forward<Args>(args)... );
  }

  template <class U>
  static void destroy (U* p)
  {
    p->~U();
  }

  pointer allocate( size_type n, void * = nullptr)
  {
    if (n > max_size() ) {
      throw std::bad_alloc();
    }

    typename internal_circular_pool_type::iterator itr = m_circular_pool.emplace();

    if (!itr) {
      throw std::bad_alloc();
    }

    // set the circular_pool node to allow O(1) deallocate
    itr->m_circular_pool_node = reinterpret_cast<void*>(circular_pool_node_type::get_node(itr));
    __sync_synchronize();

    return reinterpret_cast<pointer>(itr->m_buffer);
  }

  void deallocate( pointer p, size_type n )
  {
    if (n > max_size() ) {
      printf("Error: Pool allocator cannot deallocate arrays.");
      return;
    }

    Node * node = reinterpret_cast<Node *>(p);
    circular_pool_node_type * circular_pool_node = reinterpret_cast<circular_pool_node_type *>(node->m_circular_pool_node);
    typename internal_circular_pool_type::iterator itr = circular_pool_node->get_iterator( node );

    if (itr) {
      m_circular_pool.erase(itr);
    }
    else {
      printf("Error: double deallocate.");
    }
  }

private:
  internal_circular_pool_type m_circular_pool;
};

} // namespace Lockfree

#endif // LOCKFREE_POOL_ALLOCATOR_HPP
