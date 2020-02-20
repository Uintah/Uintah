/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef IMPL_LOCKFREE_POOL_HPP
#define IMPL_LOCKFREE_POOL_HPP

#include "Lockfree_Macros.hpp"
#include "Lockfree_Node.hpp"

#include <array>
#include <atomic>

#include <cstdio>


namespace Lockfree { namespace Impl {

// Copies of pool are shallow,  i.e., they point to the same reference counted memory.
// So each thread should have its own copy of the pool
// It is not thread safe for multiple threads to interact with the same instance of a pool
template <  typename T
          , typename BitsetBlockType
          , unsigned BitsetNumBlocks
          , template <typename> class Allocator
         >
class Pool
{
public:
  using value_type = T;
  template <typename U> using allocator = Allocator<U>;

  using node_type = Node<T, BitsetBlockType, BitsetNumBlocks >;
  using node_allocator_type = allocator<node_type>;

  using iterator = typename node_type::iterator;
  using handle   = typename node_type::handle;

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename... Args>
  iterator emplace( handle const& h, Args&&... args)
  {
    constexpr size_t one = 1u;

    m_size.fetch_add( one, std::memory_order_relaxed );

    int added_node = 0;
    node_type * const start = static_cast<bool>(h) ? node_type::get_node(h) : m_head.load( std::memory_order_relaxed );
    int start_block = static_cast<bool>(h) ? node_type::get_block(h) : 0;

    const int num_search_nodes = 100u * size() < 95u * capacity() ? m_nodes.load( std::memory_order_relaxed ) : 1;

    iterator itr = node_type::emplace_helper( m_pool_id
                                            , this
                                            , start
                                            , start_block
                                            , start
                                            , num_search_nodes
                                            , m_node_allocator
                                            , added_node
                                            , std::forward<Args>(args)...
                                            );


    if (added_node) {
      m_nodes.fetch_add( one, std::memory_order_relaxed );
    }

    if (!h) {
      m_head.store( node_type::get_node(itr), std::memory_order_relaxed );
    }

    return itr;
  }

  /// erase( iterator )
  ///
  /// if the iterator is valid erase its value
  void erase( iterator & itr )
  {
    if (itr) {
      node_type::erase( itr );
      constexpr size_t one = 1;
      m_size.fetch_sub( one, std::memory_order_relaxed );
    }
  }

  template <typename UnaryPredicate>
  iterator find_any( handle const& h, UnaryPredicate const & pred ) const
  {
    iterator itr{};

    if (m_size.load( std::memory_order_relaxed ) == 0u) return itr;

    const bool use_handle = static_cast<bool>(h) && node_type::get_node(h)->impl_pool_id() == m_pool_id;
    node_type * start =  use_handle ?
                        node_type::get_node(h) :
                        m_find_head.load( std::memory_order_relaxed )
                        ;
    int start_block = use_handle ? node_type::get_block(h) : 0;

    itr = node_type::find_any( start, start_block, start, pred );

    if ( !h && itr ) {
      m_find_head.store( node_type::get_node(itr), std::memory_order_relaxed );
    }

    return itr;
  }

  handle get_handle( void * p ) const {
    node_type * start = m_find_head.load( std::memory_order_relaxed );
    return node_type::get_handle( start, start, p );
  }

  /// Contruct a pool
  Pool( size_t pid, node_allocator_type const& node_allocator )
    : m_pool_id{pid}
    , m_head{}
    , m_find_head{}
    , m_node_allocator{ node_allocator }
  {
    m_head = m_node_allocator.allocate(1);
    m_node_allocator.construct( m_head.load( std::memory_order_relaxed ), m_pool_id, this );

    m_find_head.store( m_head.load( std::memory_order_relaxed ), std::memory_order_relaxed );
  }

  // shallow copy with a hint on how many task this thread will insert
  Pool( Pool const & rhs ) = delete;
  Pool & operator=( Pool const & rhs ) = delete;
  Pool( Pool && rhs ) = delete;
  Pool & operator=( Pool && rhs ) = delete;

  ~Pool()
  {
    node_type * start = m_head.load( std::memory_order_relaxed );
    node_type * curr = start;
    node_type * next;

    // iterate circular pool deleting nodes
    do {
      next = curr->next();
      m_node_allocator.destroy(curr);
      m_node_allocator.deallocate( curr, 1 );
      curr = next;
    } while ( curr != start );
  }

  size_t pool_id() const { return m_pool_id; }

  size_t size() const { return m_size.load( std::memory_order_relaxed ); }
  size_t capacity() const { return node_type::capacity * m_nodes.load( std::memory_order_relaxed ); }

private: // data members

  size_t                           m_pool_id;
  std::atomic<size_t>              m_size{0u};
  std::atomic<size_t>              m_nodes{1u};
  std::atomic<node_type *>         m_head;
  mutable std::atomic<node_type *> m_find_head;
  node_allocator_type              m_node_allocator;
};

}} // namespace Lockfree::Impl


#endif //LOCKFREE_POOL_HPP

