#ifndef LOCKFREE_POOL_ALLOCATOR_HPP
#define LOCKFREE_POOL_ALLOCATOR_HPP

#include "impl/Lockfree_Pool.hpp"
#include "Lockfree_Mappers.hpp"

#include <new> // for bad_alloc

namespace Lockfree {

template <  typename T
          , typename BitsetBlockType = uint64_t
          , unsigned BitsetNumBlocks = 2u
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
  using allocator = PoolAllocator< T
                                 , BitsetBlockType
                                 , BitsetNumBlocks
                                 , Allocator
                                 , SizeAllocator
                                 >;

  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;


private:
  using impl_pool_type = Impl::Pool< Node
                                   , BitsetBlockType
                                   , BitsetNumBlocks
                                   , Allocator
                                   >;
  using impl_node_type = typename impl_pool_type::node_type;

  using node_allocator_type = Allocator<impl_node_type>;
  using pool_allocator_type = SizeAllocator< impl_pool_type >;
  using size_allocator_type = SizeAllocator< std::atomic<size_type> >;

  static constexpr size_type one = 1;
  static constexpr size_type zero = 0;

public:

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

  PoolAllocator( size_type n = 31 )
    : m_num_levels{ n }
  {
    m_pools = m_pool_allocator.allocate( m_num_levels );
    for ( size_type i=0; i<m_num_levels; ++i) {
      m_pool_allocator.construct( m_pools + i, i, m_node_allocator );
    }

    m_refcount = m_size_allocator.allocate(1);
    m_size_allocator.construct( m_refcount, 1 );

    std::atomic_thread_fence( std::memory_order_seq_cst );
  }

  PoolAllocator( const PoolAllocator & rhs )
    : m_num_levels{ rhs.m_num_levels }
    , m_pools{ rhs.m_pools }
    , m_refcount{ rhs.m_refcount }
    , m_node_allocator{ rhs.m_node_allocator }
    , m_pool_allocator{ rhs.m_pool_allocator }
    , m_size_allocator{ rhs.m_size_allocator }
  {
    m_refcount->fetch_add(one, std::memory_order_relaxed );
  }

  PoolAllocator & operator=( const PoolAllocator & rhs )
  {
    // check for self assignment
    if ( this != & rhs ) {
      m_num_levels       = rhs.m_num_levels;
      m_pools            = rhs.m_pools;
      m_refcount         = rhs.m_refcount;
      m_node_allocator   = rhs.m_node_allocator;
      m_pool_allocator   = rhs.m_pool_allocator;
      m_size_allocator   = rhs.m_size_allocator;

      m_refcount->fetch_add(one, std::memory_order_relaxed);
    }

    return *this;
  }

  PoolAllocator( PoolAllocator && rhs )
    : m_num_levels{ std::move( rhs.m_num_levels ) }
    , m_pools{ rhs.m_pools }
    , m_refcount{ std::move( rhs.m_refcount ) }
    , m_node_allocator{ std::move( rhs.m_node_allocator ) }
    , m_pool_allocator{ std::move( rhs.m_pool_allocator ) }
    , m_size_allocator{ std::move( rhs.m_size_allocator ) }
  {
    // invalidate rhs
    rhs.m_num_levels = 0u;
    rhs.m_pools = nullptr;
    rhs.m_refcount = nullptr;
    rhs.m_node_allocator   = node_allocator_type{};
    rhs.m_pool_allocator   = pool_allocator_type{};
    rhs.m_size_allocator = size_allocator_type{};
  }

  PoolAllocator & operator=( PoolAllocator && rhs )
  {
    std::swap( m_num_levels, rhs.m_num_levels );
    std::swap( m_pools, rhs.m_pools );
    std::swap( m_refcount, rhs.m_refcount );
    std::swap( m_node_allocator, rhs.m_node_allocator );
    std::swap( m_pool_allocator, rhs.m_pool_allocator );
    std::swap( m_size_allocator, rhs.m_size_allocator );

    return *this;
  }

  ~PoolAllocator()
  {
    if ( m_refcount && m_refcount->fetch_sub(one, std::memory_order_relaxed ) == one ) {

      for ( size_type i=0; i<m_num_levels; ++i) {
        m_pool_allocator.destroy( m_pools + i );
      }
      m_pool_allocator.deallocate( m_pools, m_num_levels);

      m_size_allocator.deallocate( m_refcount, 1 );
    }
  }

  size_type num_levels() const { return m_num_levels; }

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


    typename impl_pool_type::iterator itr;
    typename impl_pool_type::handle h;
    if ( hint ) {
      Node * node = reinterpret_cast<Node *>(hint);
      impl_node_type * pool_node = reinterpret_cast<impl_node_type *>(node->m_pool_node);
      typename impl_pool_type::handle h = pool_node->get_handle( node );
    }

    const size_type level = static_cast<bool>(h) ?
                             impl_node_type::get_node(h)->impl_pool_id() :
                             m_mapper( num_levels() );

    itr = m_pools[level].emplace( h );

    if (!itr) { throw std::bad_alloc(); }

    // set the pool node to allow O(1) deallocate
    itr->m_pool_node = reinterpret_cast<void*>(impl_node_type::get_node(itr));
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
    impl_node_type * pool_node = reinterpret_cast<impl_node_type *>(node->m_pool_node);
    typename impl_pool_type::iterator itr = pool_node->get_iterator( node );

    if (itr) {
      m_pools[itr.level()].erase(itr);
    }
    else {
      printf("Error: double deallocate.");
    }
  }

private:
  size_type                m_num_levels;
  impl_pool_type         * m_pools{nullptr};
  std::atomic<size_type> * m_refcount{nullptr};
  node_allocator_type      m_node_allocator{};
  pool_allocator_type      m_pool_allocator{};
  size_allocator_type      m_size_allocator{};
  ThreadIDMapper           m_mapper{};
};

} // namespace Lockfree

#endif // LOCKFREE_POOL_ALLOCATOR_HPP
