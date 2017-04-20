/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef LOCKFREE_LEVEL_POOL_HPP
#define LOCKFREE_LEVEL_POOL_HPP

#include "impl/Lockfree_Pool.hpp"


namespace Lockfree {

// Copies of pool are shallow,  i.e., they point to the same reference counted memory.
// So each thread should have its own copy of the pool
// It is not thread safe for multiple threads to interact with the same instance of a pool
template <  typename T
          , typename BitsetBlockType = uint64_t
          , unsigned BitsetNumBlocks = 1u
          , template <typename> class Allocator = std::allocator
          , template <typename> class SizeAllocator = std::allocator
         >
class LevelPool
{
public:
  using size_type = size_t;
  using value_type = T;
  template <typename U> using allocator = Allocator<U>;
  template <typename U> using size_allocator = SizeAllocator<U>;

  using pool_type = LevelPool<  T
                         , BitsetBlockType
                         , BitsetNumBlocks
                         , Allocator
                         , SizeAllocator
                        >;

private:
  using impl_pool_type = Impl::Pool< T
                                   , BitsetBlockType
                                   , BitsetNumBlocks
                                   , Allocator
                                   >;
  using impl_node_type = typename impl_pool_type::node_type;

  static constexpr size_type one = 1;
  static constexpr size_type zero = 0;

public:
  using node_allocator_type = allocator<impl_node_type>;
  using pool_allocator_type = size_allocator< impl_pool_type >;
  using size_allocator_type = size_allocator< std::atomic<size_type> >;

  using iterator = typename impl_pool_type::iterator;
  using handle   = typename impl_pool_type::handle;

  /// insert( value )
  ///
  /// return an iterator to the newly inserted value
  iterator insert(size_t level, value_type const & value) { return emplace( handle{}, level, value); }

  /// insert( value )
  ///
  /// return an iterator to the newly inserted value
  iterator insert( handle const& h, size_t level,  value_type const & value) { return emplace(h, level, value); }

  /// return an iterator to the newly created value
  iterator emplace(size_t level) { return emplace( handle{}, level ); }

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename U>
  typename std::enable_if< !Impl::equivalent< U, handle >::value, iterator >::type
  emplace(size_t level, U & u) { return emplace( handle{}, level, u ); }

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename U, typename... Args>
  typename std::enable_if< !Impl::equivalent< U, handle >::value, iterator >::type
  emplace(size_t level, U && u) { return emplace( handle{}, level, std::forward<U>(u) ); }

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename U, typename... Args>
  typename std::enable_if< !Impl::equivalent< U, handle >::value, iterator >::type
  emplace(size_t level, U & u, Args&&... args) { return emplace( handle{}, level, u, std::forward<Args>(args)...); }

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename U, typename... Args>
  typename std::enable_if< !Impl::equivalent< U, handle >::value, iterator >::type
  emplace(size_t level, U && u, Args&&... args) { return emplace( handle{}, level, std::forward<U>(u), std::forward<Args>(args)...); }

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename... Args>
  iterator emplace(handle h, size_t level, Args&&... args)
  {
    m_size->fetch_add( one, std::memory_order_relaxed );

    h = ( h && impl_node_type::get_node(h)->impl_pool_id() == level) ? h : handle{};

    iterator itr = m_pools[level].emplace( h, std::forward<Args>(args)... );
    return itr;
  }

  /// find_any()
  ///
  /// return any valid iterater
  //  may return an invalid iterator
  iterator find_any() const
  {
    return find_any( handle{}, []( value_type const& ) { return true; } );
  }

  /// find_any()
  ///
  /// return any valid iterater
  //  may return an invalid iterator
  iterator find_any(handle const& h) const
  {
    return find_any( h, []( value_type const& ) { return true; } );
  }

  /// find_any( predicate )
  ///
  /// return an iterator to a value for which predicate returns true
  /// Predicate is a function, functor, or lambda of the form
  /// bool ()( const value_type & )
  /// If there are no values for which predicate is true return an invalid iterator
  template <typename UnaryPredicate>
  typename  std::enable_if< !Impl::equivalent< UnaryPredicate, handle >::value, iterator >::type
  find_any( UnaryPredicate const & pred ) const
  {
    return find_any( handle{}, pred );
  }

  /// find_any( predicate )
  ///
  /// return an iterator to a value for which predicate returns true
  /// Predicate is a function, functor, or lambda of the form
  /// bool ()( const value_type & )
  /// If there are no values for which predicate is true return an invalid iterator
  template <typename UnaryPredicate>
  iterator find_any( handle const& h, UnaryPredicate const & pred ) const
  {
    iterator itr{};

    if (m_size->load( std::memory_order_relaxed ) == 0u ) return itr;

    for (size_t i=0; !itr && i<m_num_levels; ++i) {
      handle hh = (static_cast<bool>(h) && impl_node_type::get_node(h)->impl_pool_id() == i) ? h : handle{};
      itr = m_pools[i].find_any( hh, pred );
    }
    return itr;
  }

  /// find_any()
  ///
  /// return any valid iterater
  //  may return an invalid iterator
  iterator find_any(size_t level) const
  {
    return find_any( handle{}, level, []( value_type const& ) { return true; } );
  }

  /// find_any()
  ///
  /// return any valid iterater
  //  may return an invalid iterator
  iterator find_any(handle const& h, size_t level) const
  {
    return find_any( h, level, []( value_type const& ) { return true; } );
  }

  /// find_any( predicate )
  ///
  /// return an iterator to a value for which predicate returns true
  /// Predicate is a function, functor, or lambda of the form
  /// bool ()( const value_type & )
  /// If there are no values for which predicate is true return an invalid iterator
  template <typename UnaryPredicate>
  typename  std::enable_if< !Impl::equivalent< UnaryPredicate, handle >::value, iterator >::type
  find_any( size_t level, UnaryPredicate const & pred ) const
  {
    return find_any( handle{}, level, pred );
  }

  /// find_any( predicate )
  ///
  /// return an iterator to a value for which predicate returns true
  /// Predicate is a function, functor, or lambda of the form
  /// bool ()( const value_type & )
  /// If there are no values for which predicate is true return an invalid iterator
  template <typename UnaryPredicate>
  iterator find_any( handle h, size_t level, UnaryPredicate const & pred ) const
  {
    iterator itr{};

    if (m_size->load( std::memory_order_relaxed ) == 0u ) return itr;

    h = (static_cast<bool>(h) && impl_node_type::get_node(h)->impl_pool_id() == level) ? h : handle{};
    itr = m_pools[level].find_any(h, pred);

    return itr;
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

  /// size()
  ///
  /// number of values currently in the pool
  LOCKFREE_FORCEINLINE
  size_type size() const
  {
    return m_size->load( std::memory_order_relaxed );
  }

  /// size( size_t level )
  ///
  /// number of values currently in the given level
  LOCKFREE_FORCEINLINE
  size_type size(size_t level) const
  {
    return m_pools[level].size();
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

  /// empty( size_t level )
  ///
  /// is the pool empty
  LOCKFREE_FORCEINLINE
  bool empty( size_t level ) const
  {
    return size(level) == 0u;
  }


  /// Contruct a pool
  LevelPool( size_t num_levels = 31u
           , node_allocator_type const& node_alloc = node_allocator_type{}
           , pool_allocator_type const& pool_alloc = pool_allocator_type{}
           , size_allocator_type const& size_alloc = size_allocator_type{}
           )
    : m_num_levels{ num_levels }
    , m_node_allocator{ node_alloc }
    , m_pool_allocator{ pool_alloc }
    , m_size_allocator{ size_alloc }
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
  LevelPool( LevelPool const & rhs )
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
  LevelPool & operator=( LevelPool const & rhs )
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

      m_refcount->fetch_add(one, std::memory_order_relaxed);
    }

    return *this;
  }

  // move constructor
  LevelPool( LevelPool && rhs )
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
  LevelPool & operator=( LevelPool && rhs )
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

  ~LevelPool()
  {

    if ( m_refcount && m_refcount->fetch_sub(one, std::memory_order_relaxed ) == one ) {

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
  impl_pool_type         * m_pools{nullptr};
  std::atomic<size_type> * m_size{nullptr};
  std::atomic<size_type> * m_refcount{nullptr};
  node_allocator_type      m_node_allocator;
  pool_allocator_type      m_pool_allocator;
  size_allocator_type      m_size_allocator;
};


} // namespace Lockfree


#endif //LOCKFREE_LEVEL_POOL_HPP

