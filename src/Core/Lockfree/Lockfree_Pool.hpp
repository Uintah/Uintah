#ifndef LOCKFREE_POOL_HPP
#define LOCKFREE_POOL_HPP

#include "Lockfree_Mappers.hpp"
#include "impl/Lockfree_Pool.hpp"


namespace Lockfree {


// Copies of pool are shallow,  i.e., they point to the same reference counted memory.
// So each thread should have its own copy of the pool
// It is not thread safe for multiple threads to interact with the same instance of a pool
template <  typename T
          , template <typename> class Allocator = std::allocator
          , template <typename> class SizeAllocator = std::allocator
          , typename Mapper = ThreadIDMapper
         >
class Pool
{
public:
  using size_type = size_t;
  using value_type = T;
  template <typename U> using allocator = Allocator<U>;
  template <typename U> using size_allocator = SizeAllocator<U>;
  using mapper = Mapper;

  using pool_type = Pool<  T
                         , Allocator
                         , SizeAllocator
                         , Mapper
                        >;

  using impl_pool_type = Impl::Pool<T, Allocator >;
  using impl_node_type = typename impl_pool_type::node_type;


private:
  using node_allocator_type = allocator<impl_node_type>;

  struct PoolNode {
    std::atomic<PoolNode *>  m_next{nullptr};
    impl_pool_type m_pool;

    PoolNode( size_t pid, node_allocator_type & node_allocator)
      : m_pool{ pid, node_allocator }
    {}

    PoolNode( const PoolNode & ) = delete;
    PoolNode & operator=( const PoolNode & ) = delete;
    PoolNode( PoolNode && ) = delete;
    PoolNode & operator=( PoolNode && ) = delete;
  };


  using pool_allocator_type = size_allocator< PoolNode >;
  using size_allocator_type = size_allocator< std::atomic<size_type> >;

  static constexpr size_type one = 1;
  static constexpr size_type zero = 0;

public:
  using iterator = typename impl_pool_type::iterator;

  /// insert( value )
  ///
  /// return an iterator to the newly inserted value
  iterator insert(value_type const & value)
  {
    return emplace(value);
  }

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename... Args>
  iterator emplace(Args&&... args)
  {
    m_size->fetch_add( one, std::memory_order_relaxed );

    const size_t id = m_mapper(m_num_levels, args...);

    iterator itr = m_pools[id].m_pool.emplace( std::forward<Args>(args)... );

    return itr;
  }



  /// find_any( predicate )
  ///
  /// return an iterator to a value for which predicate returns true
  /// Predicate is a function, functor, or lambda of the form
  /// bool ()( const value_type & )
  /// If there are no values for which predicate is true return an invalid iterator
  template <typename UnaryPredicate>
  iterator find_any( UnaryPredicate const & pred ) const
  {
    iterator itr{};

    size_t start = m_mapper( m_num_levels );

    for (size_t i=0u; !itr && i < m_num_levels; ++i) {
      itr = m_pools[(i+start)%m_num_levels].m_pool.find_any(pred);
    }

    return itr;
  }

  /// find_any()
  ///
  /// return any valid iterater
  //  may return an invalid iterator
  iterator find_any() const
  {
    auto pred = []( const value_type & ) { return true; };
    return find_any( pred );
  }

  /// erase( iterator )
  ///
  /// if the iterator is valid erase its value
  void erase( iterator & itr )
  {
    if (itr) {
      impl_pool_type * pool = reinterpret_cast<impl_pool_type *>( impl_node_type::get_node(itr)->impl_pool() );
      pool->erase( itr );
      m_size->fetch_sub( one, std::memory_order_relaxed );
    }
  }

  /// erase_and_advance( iterator, pred )
  ///
  /// if the iterator is valid erase its current value
  /// advance it to the next value for which predicate is true
  template <typename UnaryPredicate>
  void erase_and_advance( iterator & itr, UnaryPredicate const & pred )
  {
    if (itr) {
      impl_pool_type * pool = reinterpret_cast<impl_pool_type *>( impl_node_type::get_node(itr)->impl_pool() );
      pool->erase_and_advance( itr, pred );
      m_size->fetch_sub( one, std::memory_order_relaxed );
    }
  }

  /// erase_and_advance( iterator )
  ///
  /// if the iterator is valid erase its current value
  /// advance it to the next value
  void erase_and_advance( iterator & itr )
  {
    auto pred = []( value_type const& )->bool { return true; };
    erase_and_advance( itr, pred );
  }

  /// size()
  ///
  /// number of values currently in the pool
  LOCKFREE_FORCEINLINE
  size_type size() const
  {
    return m_size->load( std::memory_order_relaxed );
  }

  /// ref_count()
  ///
  /// number of references to the pool
  size_type ref_count() const
  {
    return m_refcount->load( std::memory_order_relaxed );
  }

  /// empty()
  ///
  /// is the pool empty
  LOCKFREE_FORCEINLINE
  bool empty() const
  {
    return size() == 0u;
  }


  /// Contruct a pool
  Pool( size_t num_levels = 31u )
    : m_num_levels{ num_levels }
  {
    m_pools = m_pool_allocator.allocate( num_levels );
    for ( size_t i=0; i<m_num_levels; ++i) {
      m_pool_allocator.construct( m_pools + i, i, m_node_allocator );
    }

    m_size = m_size_allocator.allocate(1);
    m_size_allocator.construct( m_size, 0 );

    m_refcount = m_size_allocator.allocate(1);
    m_size_allocator.construct( m_refcount, 1 );

    std::atomic_thread_fence( std::memory_order_seq_cst );
  }


  // shallow copy with a hint on how many task this thread will insert
  Pool( Pool const & rhs )
    : m_num_levels{ rhs.m_num_levels }
    , m_pools{ rhs.m_pools }
    , m_size{ rhs.m_size }
    , m_refcount{ rhs.m_refcount }
    , m_node_allocator{ rhs.m_node_allocator }
    , m_pool_allocator{ rhs.m_pool_allocator }
    , m_size_allocator{ rhs.m_size_allocator }
  {
    m_refcount->fetch_add(one, std::memory_order_relaxed );
  }


  // shallow copy
  Pool & operator=( Pool const & rhs )
  {
    // check for self assignment
    if ( this != & rhs ) {
      m_num_levels       = rhs.m_num_levels;
      m_pools            = rhs.m_pools;
      m_size             = rhs.m_size;
      m_refcount         = rhs.m_refcount;
      m_node_allocator   = rhs.m_node_allocator;
      m_pool_allocator   = rhs.m_pool_allocator;
      m_size_allocator   = rhs.m_size_allocator;

      size_type rcount = m_refcount->fetch_add(one, std::memory_order_relaxed);
    }

    return *this;
  }

  // move constructor
  Pool( Pool && rhs )
    : m_num_levels{ std::move( rhs.m_num_levels ) }
    , m_pools{ rhs.m_pools }
    , m_size{ std::move( rhs.m_size ) }
    , m_refcount{ std::move( rhs.m_refcount ) }
    , m_node_allocator{ std::move( rhs.m_node_allocator ) }
    , m_pool_allocator{ std::move( rhs.m_pool_allocator ) }
    , m_size_allocator{ std::move( rhs.m_size_allocator ) }
  {
    // invalidate rhs
    rhs.m_num_levels = 0u;
    rhs.m_pools = nullptr;
    rhs.m_size = nullptr;
    rhs.m_refcount = nullptr;
    rhs.m_node_allocator   = node_allocator_type{};
    rhs.m_pool_allocator   = pool_allocator_type{};
    rhs.m_size_allocator = size_allocator_type{};
  }

  // move assignement
  //
  // NOT thread safe
  Pool & operator=( Pool && rhs )
  {
    std::swap( m_num_levels, rhs.m_num_levels );
    std::swap( m_pools, rhs.m_pools );
    std::swap( m_size, rhs.m_size );
    std::swap( m_refcount, rhs.m_refcount );
    std::swap( m_node_allocator, rhs.m_node_allocator );
    std::swap( m_pool_allocator, rhs.m_pool_allocator );
    std::swap( m_size_allocator, rhs.m_size_allocator );

    return *this;
  }

  ~Pool()
  {

    if ( m_size &&  m_refcount->fetch_sub(one, std::memory_order_relaxed ) == one ) {

      for ( size_t i=0; i<m_num_levels; ++i) {
        m_pool_allocator.destroy( m_pools + i );
      }
      m_pool_allocator.deallocate( m_pools, m_num_levels);

      m_size_allocator.deallocate( m_size, 1 );
      m_size_allocator.deallocate( m_refcount, 1 );
    }
  }

  size_t num_levels() const { return m_num_levels; }

private: // data members

  size_t                   m_num_levels;
  PoolNode               * m_pools{nullptr};
  std::atomic<size_type> * m_size{nullptr};
  std::atomic<size_type> * m_refcount{nullptr};
  node_allocator_type      m_node_allocator{};
  pool_allocator_type      m_pool_allocator{};
  size_allocator_type      m_size_allocator{};
  mapper                   m_mapper{};
};


} // namespace Lockfree


#endif //LOCKFREE_POOL_HPP

