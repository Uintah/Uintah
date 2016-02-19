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
          , template <typename> class Allocator
         >
class Pool
{
public:
  using value_type = T;
  template <typename U> using allocator = Allocator<U>;

  using node_type = Node<T>;
  using node_allocator_type = allocator<node_type>;

  using iterator = typename node_type::iterator;

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename... Args>
  iterator emplace( Args&&... args)
  {
    constexpr size_t one = 1u;
    m_size.fetch_add( one, std::memory_order_relaxed );

    int added_node = 0;
    node_type * start = m_head.load( std::memory_order_relaxed );

    const int num_search_nodes = 100u * size() < 95u * capacity() ?
                                   m_nodes.load( std::memory_order_relaxed ) : 1;

    iterator itr = start->emplace_helper( num_search_nodes
                                         ,m_node_allocator
                                         ,added_node
                                         ,std::forward<Args>(args)...
                                        );


    m_head.store( node_type::get_node(itr), std::memory_order_relaxed );

    if (added_node) {
      m_nodes.fetch_add( one, std::memory_order_relaxed );
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

  /// erase_and_advance( iterator, pred )
  ///
  /// if the iterator is valid erase its current value
  /// advance it to the next value for which predicate is true
  template <typename UnaryPredicate>
  void erase_and_advance( iterator & itr, UnaryPredicate const & pred )
  {
    if (itr) {
      if (m_size.load( std::memory_order_relaxed) > 1u ) {
        node_type::erase_and_advance(itr, pred);
        constexpr size_t one = 1;
        m_size.fetch_sub( one, std::memory_order_relaxed );
      }
      else {
        erase( itr );
      }
    }
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

    if (m_size.load( std::memory_order_relaxed) > 0u ) {
      node_type * start = m_find_head.load( std::memory_order_relaxed );
      itr = node_type::find_any( start, pred );

      // try to set front to itr
      if ( itr ) {
        m_find_head.store( node_type::get_node(itr), std::memory_order_relaxed );
      }
    }

    return itr;
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

