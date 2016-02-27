#ifndef LOCKFREE_POOL_ALLOCATOR_HPP
#define LOCKFREE_POOL_ALLOCATOR_HPP

#include "impl/Lockfree_Macros.hpp"

#include "Lockfree_LevelPool.hpp"
#include "Lockfree_Mappers.hpp"

#include <new> // for bad_alloc

namespace Lockfree {

template <  typename T
          , typename BitsetBlockType = uint64_t
          , unsigned BitsetNumBlocks = 1u
          , template <typename> class Allocator = std::allocator
          , template <typename> class SizeAllocator = std::allocator
        >
class PoolAllocator
{

  struct LOCKFREE_ALIGNAS( LOCKFREE_ALIGNOF(T) ) Node
  {
    static constexpr size_t alignment = LOCKFREE_ALIGNOF(T) ? LOCKFREE_ALIGNOF(T) : 16;
    static constexpr size_t buffer_size =  (alignment * ((sizeof(T) + sizeof(void *) - 1u + alignment) / alignment)) - sizeof(void*);

    char   m_buffer[buffer_size];
    void * m_pool_node;
  };

public:
  using internal_pool_type = LevelPool<  Node
                                       , BitsetBlockType
                                       , BitsetNumBlocks
                                       , Allocator
                                       , SizeAllocator
                                      >;

private:
  using pool_node_type = typename internal_pool_type::impl_node_type;

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
                                 , BitsetBlockType
                                 , BitsetNumBlocks
                                 , Allocator
                                 , SizeAllocator
                               >;
  };

  PoolAllocator( size_t n = 31)
    : m_pool{n}
  {}

  PoolAllocator( const PoolAllocator & rhs )
    : m_pool{ rhs.m_pool }
  {}

  PoolAllocator & operator=( const PoolAllocator & rhs )
  {
    m_pool = rhs.m_pool;
    return *this;
  }

  PoolAllocator( PoolAllocator && rhs )
    : m_pool{ std::move( rhs.m_pool ) }
  {}

  PoolAllocator & operator=( PoolAllocator && rhs )
  {
    m_pool = std::move( rhs.m_pool );
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
    std::atomic_thread_fence( std::memory_order_seq_cst );
  }

  template <class U>
  static void destroy (U* p)
  {
    p->~U();
    std::atomic_thread_fence( std::memory_order_seq_cst );
  }

  pointer allocate( size_type n, void * hint = nullptr)
  {
    if (n > max_size() ) {
      throw std::bad_alloc();
    }

    size_t level = m_mapper( m_pool.num_levels() );

    typename internal_pool_type::iterator itr;
    if ( hint ) {
      Node * node = reinterpret_cast<Node *>(hint);
      pool_node_type * pool_node = reinterpret_cast<pool_node_type *>(node->m_pool_node);
      typename internal_pool_type::handle h = pool_node->get_handle( node );
      itr = m_pool.emplace( h, level );
    }
    else {
      itr = m_pool.emplace( level );
    }

    if (!itr) {
      throw std::bad_alloc();
    }

    // set the pool node to allow O(1) deallocate
    itr->m_pool_node = reinterpret_cast<void*>(pool_node_type::get_node(itr));
    std::atomic_thread_fence( std::memory_order_seq_cst );

    return reinterpret_cast<pointer>(itr->m_buffer);
  }

  void deallocate( pointer p, size_type n )
  {
    if (n > max_size() ) {
      printf("Error: Pool allocator cannot deallocate arrays.");
      return;
    }

    Node * node = reinterpret_cast<Node *>(p);
    pool_node_type * pool_node = reinterpret_cast<pool_node_type *>(node->m_pool_node);
    typename internal_pool_type::iterator itr = pool_node->get_iterator( node );

    if (itr) {
      m_pool.erase(itr);
    }
    else {
      printf("Error: double deallocate.");
    }
  }

private:
  internal_pool_type m_pool;
  ThreadIDMapper     m_mapper{};
};

} // namespace Lockfree

#endif // LOCKFREE_POOL_ALLOCATOR_HPP
