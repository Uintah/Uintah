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

#ifndef LOCKFREE_POOL_NODE_HPP
#define LOCKFREE_POOL_NODE_HPP

#include "Lockfree_Macros.hpp"
#include "Lockfree_Bits.hpp"

#include <utility>      // for std::swap
#include <type_traits>

#include <climits>
#include <cstdio>
#include <atomic>

namespace Lockfree { namespace Impl {

template <typename T>
struct remove_cvr
{
  using type = typename std::remove_reference< typename std::remove_cv< T >::type >::type;
};

template <typename T, typename U>
struct equivalent
{
  static constexpr bool value = std::is_same< typename remove_cvr<T>::type, typename remove_cvr<U>::type >::value;
};


template <typename T>
constexpr int align_to_cacheline()
{
  return LOCKFREE_ALIGNOF(T) > 64 ? LOCKFREE_ALIGNOF(T) : 64;
}

//-----------------------------------------------------------------------------
/// class Node
///
/// Fundamental building block for more complex lockfree data structures
//-----------------------------------------------------------------------------
template <typename T, typename BitsetBlockType, unsigned BitsetNumBlocks>
class LOCKFREE_ALIGNAS( align_to_cacheline<T>() ) Node
{
public:
  using node_type = Node<T, BitsetBlockType, BitsetNumBlocks>;
  using value_type = T;
  using bitset_type = Bitset<BitsetBlockType, BitsetNumBlocks>;
  using block_type = typename bitset_type::block_type;

  static constexpr int capacity = bitset_type::capacity;
  static constexpr block_type one  = 1u;
  static constexpr block_type zero = 0u;


  /// class iterator
  ///
  /// Node iterators are moveable but not copyable.  When a valid iterator
  /// is constructed its corresponding valid bit on the node is atomically claimed.
  /// Before a valid iterator is advanced or destroyed the valid bit is release.
  /// So a valid iterator is guaranteed to have exclusive read/write access to the
  /// referenced value.
  class iterator
  {
  public:
    using value_type = T;

    // get a reference of the value
    LOCKFREE_FORCEINLINE
    value_type & operator*() const { return m_node->get(m_idx); }

    // get a pointer of the value
    LOCKFREE_FORCEINLINE
    value_type * operator->() const { return m_node->get_ptr(m_idx); }

    // is this iterator valid ?
    LOCKFREE_FORCEINLINE
    explicit operator bool() const { return m_node != nullptr; }


    // release the iterator
    LOCKFREE_FORCEINLINE
    size_t level() const { return m_node ? m_node->impl_pool_id() : ~static_cast<size_t>(0); }

    // release the iterator
    LOCKFREE_FORCEINLINE
    void clear() { *this = iterator{}; }

    // construct an invalid iterator
    iterator()
      : m_node{ nullptr }
      , m_idx{ -1 }
    {}

    // move constructor
    iterator( iterator && itr )
      : m_node{ itr.m_node }
      , m_idx{ itr.m_idx }
    {
      itr.m_node = nullptr;
      itr.m_idx = -1;
      std::atomic_thread_fence( std::memory_order_seq_cst );
    }

    // move assignment
    iterator & operator=( iterator && itr )
    {
      // itr is going to be destroy
      // swap values to let it clean up "this" resource
      std::swap( m_node , itr.m_node );
      std::swap( m_idx , itr.m_idx );
      std::atomic_thread_fence( std::memory_order_seq_cst );
      return *this;
    }

    // destructor
    ~iterator()
    {
      // allow another iterator to point to m_idx
      if (m_node) {
        m_node->atomic_release(m_idx);
      }
    }

    // disallow copy and assignment
    iterator( const iterator & ) = delete;
    iterator & operator=( const iterator & ) = delete;

  private:
    friend class Node;
    friend class handle;

    // only Node is allowed to construct a valid iterator
    iterator( node_type * node, int idx )
      : m_node{ node }
      , m_idx{ idx }
    {
      std::atomic_thread_fence( std::memory_order_seq_cst );
    }

    node_type * m_node;
    int         m_idx;
  };

  class handle
  {
  public:

    // is this handle valid ?
    LOCKFREE_FORCEINLINE
    explicit operator bool() const { return m_node != nullptr; }

    LOCKFREE_FORCEINLINE
    explicit operator iterator() const
    {
      if ( m_node ) {
        return m_node->get_iterator(m_idx);
      }
      return iterator{};
    }

    bool exists() const
    {
      return (m_node != nullptr) && m_node->test_used( m_idx );
    }

    // release the iterator
    LOCKFREE_FORCEINLINE
    size_t level() const { return m_node ? m_node->impl_pool_id() : ~static_cast<size_t>(0); }

    // construct an invalid handle
    handle() = default;
    handle( const handle & ) = default;
    handle & operator=( const handle & ) = default;
    handle( handle && ) = default;
    handle & operator=( handle && ) = default;

    handle( const iterator & itr )
      : m_node{ itr.m_node }
      , m_idx{ itr.m_idx }
    {}

    handle & operator=( const iterator & itr )
    {
      m_node = itr.m_node;
      m_idx = itr.m_idx;
      return *this;
    }

  private:
    friend class Node;

    handle( node_type * n, int i )
      : m_node{n}
      , m_idx{i}
    {}

    node_type * m_node{nullptr};
    int         m_idx{-1};
  };

  /// get_node( iterator )
  ///
  /// get the node off the iterator
  LOCKFREE_FORCEINLINE
  static node_type * get_node( iterator const & itr ) { return itr.m_node; }

  /// get_node( handle )
  ///
  /// get the node off the handle
  LOCKFREE_FORCEINLINE
  static node_type * get_node( handle const & h ) { return h.m_node; }

  /// get_block( handle )
  ///
  /// get the block off the handle
  LOCKFREE_FORCEINLINE
  static int get_block( handle const & h ) { return h.m_idx / bitset_type::block_size ; }

  /// erase the iterator
  ///
  /// erase the value of iterator
  /// the iterator is invalidated
  LOCKFREE_FORCEINLINE
  static void erase( iterator & itr )
  {
    if (itr) {
      itr.m_node->atomic_erase( itr.m_idx );
      itr.m_node = nullptr;
      itr.m_idx = -1;
    }
  }


  /// find_any( node_ptr, predicate)
  ///
  /// return an iterator to a value for which predicate returns true
  /// predicate is a function, functor, or lambda of the form
  /// bool ()( const value_type & )
  /// If there are no values for which predicate is true return an invalid iterator
  template <typename UnaryPredicate>
  static iterator find_any( node_type * start, int start_block, node_type * end, UnaryPredicate const& pred)
  {
    iterator itr;

    if ( !start ) return itr;

    itr = find_any_block( start, start_block, pred );

    for (node_type * curr = start->next(); !itr && curr != end; curr = curr->next()) {
      itr = find_any_block( curr, 0, pred );
    }

    return itr;
  }

  static handle get_handle( node_type * start, node_type * end, void * p )
  {
    value_type * ptr = reinterpret_cast<value_type *>(p);
    handle h = start->get_handle(ptr);

    for ( node_type * curr = start->next(); !h && curr != end; curr = curr->next() ) {
      h = curr->get_handle(ptr);
    }
    return h;
  }

  // get_handle( node, value_ptr )
  //
  // get an handle to the value
  handle get_handle( const value_type * ptr )
  {
    if (  ptr < m_values ||  ptr >= (m_values + bitset_type::capacity) ) { return handle{}; };
    return handle{ this, static_cast<int>(ptr-m_values) };
  }

  // get_iterator( int )
  //
  // get an iterator to the value
  iterator get_iterator( int idx )
  {
    if ( idx >= capacity || idx < 0 ) {
      return iterator{};
    }

    // while the index is used
    while ( m_used_bitset.test( idx ) ) {
      if ( try_atomic_claim(idx) ) {
        return iterator{ this, idx };
      }
    }

    return iterator{};
  }

  // get_iterator( node, value_ptr )
  //
  // get an iterator to the value
  iterator get_iterator( const value_type * ptr )
  {
    return iterator{ get_handle(ptr) };
  }


  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename NodeAllocator, typename... Args>
  static iterator emplace_helper( size_t pid
                                , void * pool
                                , node_type * const start
                                , int start_block
                                , node_type * const end
                                , const int num_search_nodes
                                , NodeAllocator & allocator
                                , int & added_node
                                ,  Args&&... args
                                )
  {
    iterator itr = start->try_atomic_emplace( start_block, std::forward<Args>(args)...);

    node_type * curr = start->next();

    // try to insert the value into an existing node
    for (int n=0; !itr && (curr != end) && n< num_search_nodes; ++n) {
      itr = curr->try_atomic_emplace( 0, std::forward<Args>(args)... );
    }

    // wrapped around the pool
    // Allocate node and insert the value
    if ( !itr ) {
      // allocate and construct
      node_type * new_node = allocator.allocate(1, curr);
      allocator.construct( new_node, pid, pool );

      // will always succeed since the node is not in the pool
      itr = new_node->try_atomic_emplace( 0, std::forward<Args>(args)... );

      // insert the node at the end of pool (curr->next)
      node_type * next = curr->next();
      new_node->m_next.store( next, std::memory_order_relaxed );
      while ( !(curr->m_next.compare_exchange_strong( next, new_node, std::memory_order_relaxed, std::memory_order_relaxed) )) {
        new_node->m_next.store( next, std::memory_order_relaxed );
      }

      curr = new_node;
      added_node = 1;
    }

    return itr;
  }

  /// try_atomic_emplace( args... )
  ///
  /// try to create a value on the node
  /// if successful return the index where the value was created
  /// otherwise return capacity
  template <typename... Args>
  iterator try_atomic_emplace(int start, Args&&... args)
  {
    int idx;
    bool inserted = false;
    for (int b=start; !inserted && b < bitset_type::num_blocks; ++b) {
      block_type block = complement(m_used_bitset.load_block(b));
      while (!inserted && block) {
        int i = count_trailing_zeros(block);
        idx = (b << bitset_type::block_shift) + i;
        inserted = try_atomic_emplace_helper( idx, std::forward<Args>(args)... );

        block = block & complement(one << i);
      }
    }

    return inserted ? iterator{ this, idx } : iterator{};
  }


  /// get( idx )
  ///
  /// get a reference to the value at the given index
  LOCKFREE_FORCEINLINE
  value_type & get(int i) const
  {
    return *( m_values + i);
  }

  /// get_ptr( idx )
  ///
  /// get a ptr to the value at the given index
  LOCKFREE_FORCEINLINE
  value_type * get_ptr(int i) const
  {
    return m_values + i;
  }

  /// next()
  ///
  /// get the next node
  LOCKFREE_FORCEINLINE
  node_type * next() const
  {
    return m_next.load( std::memory_order_relaxed );
  }

  /// Construct a node pointing to itself
  Node( size_t pid, void * pool )
    : m_pool_id{ pid }
    , m_pool{ pool }
  {}

  /// Destroy the values in the node
  ~Node()
  {
    for ( int b = 0; b < bitset_type::num_blocks; ++b) {
      const block_type curr_valid = m_valid_bitset.and_block(b, zero, std::memory_order_seq_cst );
      for (int i=0; i<capacity; ++i) {
        if ( curr_valid & ( one << i) ) {
          (m_values + i)->~value_type();
        }
      }
    }
    std::atomic_thread_fence( std::memory_order_seq_cst );
  }

  size_t  impl_pool_id() const { return m_pool_id; }
  void *  impl_pool() const { return m_pool; }

private: // private functions

  template <typename UnaryPredicate>
  static iterator find_any_block( node_type * node, int start_block, UnaryPredicate const& pred)
  {
    bool found = false;
    int idx;

    for (int b=start_block; !found && b < bitset_type::num_blocks; ++b) {
      block_type old_valid = node->m_valid_bitset.and_block(b, zero);
      if (old_valid) {
        block_type test_valid = old_valid;

        while ( test_valid && !found ) {
          idx = count_trailing_zeros( test_valid );
          found = pred( node->get(idx) );
          // mask out idx
          test_valid = test_valid & complement(one << idx);
        }
      }

      if (found) {
        // release the other indexes
        // mask out idx
        const block_type new_valid = old_valid & complement(one << idx);
        node->m_valid_bitset.or_block( b, new_valid );

      }
      else {
        // release the indexes
        node->m_valid_bitset.or_block( b, old_valid );
      }
    }

    return ( found ) ? iterator{ node, idx } : iterator{};
  }

  /// atomic_erase( idx )
  ///
  /// destroy the value and release the used index
  void atomic_erase( int i )
  {
    // call the destructor
    (m_values + i)->~value_type();
    // allow index to be used again
    m_used_bitset.clear(i, std::memory_order_seq_cst );
  }

  // return true if this thread atomically set the ith bit
  template <typename... Args>
  bool try_atomic_emplace_helper( int i, Args&&... args)
  {
    const bool inserted =  m_used_bitset.set(i);
    if (inserted ) {
      // placement new with constructor
      new ((void*)(m_values + i)) value_type{ std::forward<Args>(args)... } ;
      // memory fence
      std::atomic_thread_fence( std::memory_order_seq_cst );
    }
    return inserted;
  }

  // claim the index
  LOCKFREE_FORCEINLINE
  bool try_atomic_claim( int i )
  {
    return m_valid_bitset.clear(i);
  }


  // release the index
  LOCKFREE_FORCEINLINE
  void atomic_release( int i )
  {
    m_valid_bitset.set( i, std::memory_order_seq_cst );
  }

  LOCKFREE_FORCEINLINE
  bool test_used( int i ) const
  {
    return m_used_bitset.test(i);
  }

private: // data members

  // align to cacheline (64 bytes)
  static constexpr size_t offset = sizeof(value_type * )            // m_values
                                 + sizeof(bitset_type)              // m_used_bitset
                                 + sizeof(bitset_type)              // m_valid_bitset
                                 + sizeof(std::atomic<node_type *>) // m_next
                                 + sizeof(size_t)                   // m_pool_id
                                 + sizeof(void *)                   // m_pool
                                 ;
  static constexpr size_t alignment = align_to_cacheline<T>();
  static constexpr size_t min_buffer_size = capacity * sizeof(value_type);
  static constexpr size_t buffer_size = ( alignment * ( min_buffer_size + offset + alignment - one ) / alignment ) - offset;

  char                      m_buffer[ buffer_size ]; // must be first to get correct alignment
  value_type *              m_values{ reinterpret_cast<value_type *>(m_buffer) };
  bitset_type               m_used_bitset{};
  bitset_type               m_valid_bitset{};
  std::atomic<node_type *>  m_next{ this };
  size_t                    m_pool_id;
  void *                    m_pool;
};

}} // namespace Lockfree::Impl

#endif //LOCKFREE_POOL_NODE_HPP
