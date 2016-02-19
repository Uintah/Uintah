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
constexpr int align_to_cacheline()
{
  return LOCKFREE_ALIGNOF(T) > 64 ? LOCKFREE_ALIGNOF(T) : 64;
}

//-----------------------------------------------------------------------------
/// class Node
///
/// Fundamental building block for more complex lockfree data structures
//-----------------------------------------------------------------------------
template <typename T>
class LOCKFREE_ALIGNAS( align_to_cacheline<T>() ) Node
{

public:
  using node_type = Node<T>;
  using value_type = T;
  using bitset_type = uint64_t;

  static constexpr int capacity = sizeof(bitset_type) * CHAR_BIT;
  static constexpr bitset_type one = 1u;
  static constexpr bitset_type zero = 0u;

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

    // advance the iterator to the next item
    LOCKFREE_FORCEINLINE
    void operator++() {
      node_type::advance(*this);
      std::atomic_thread_fence( std::memory_order_seq_cst );
    }

    // is this iterator valid ?
    LOCKFREE_FORCEINLINE
    explicit operator bool() const { return m_node != nullptr; }


    // release the iterator
    LOCKFREE_FORCEINLINE
    size_t level() const
    {
      return m_node->impl_pool_id();
    }

    // release the iterator
    LOCKFREE_FORCEINLINE
    void clear()
    {
      *this = iterator{};
    }

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

  /// get_node( iterator )
  ///
  /// get the node off the iterator
  LOCKFREE_FORCEINLINE
  static node_type * get_node( iterator const & itr ) { return itr.m_node; }

  /// erase the iterator
  ///
  /// erase the value of iterator
  /// the iterator is invalidated
  LOCKFREE_FORCEINLINE
  static void  erase( iterator & itr )
  {
    if (itr) {
      itr.m_node->atomic_erase( itr.m_idx );
      itr.m_node = nullptr;
      itr.m_idx = -1;
    }
  }


  /// advance( iterator & itr )
  ///
  /// advance the iterator to the next valid value in the circular_pool
  /// if there are no valid values invalidated the iterator
  static void advance( iterator & itr )
  {
    if ( !itr.m_node ) { return; }

    const node_type * start = itr.m_node;

    node_type * curr = itr.m_node;
    node_type * prev = curr;

    int idx = itr.m_idx;

    // release the current value so that another iterator can reference it
    itr = iterator{};

    // try to find another valid index within the current node
    idx = curr->try_atomic_find_next( itr.m_idx );

    // search next node
    if ( idx == capacity ) {
      do {
        idx = curr->try_atomic_find();
        prev = curr;
        curr = curr->next();
      } while ( idx == capacity && curr != start );
    }

    // successfully claimed an index
    if ( idx < capacity ) {
      itr = iterator{ prev, idx };
    }
  }

  /// erase_and_advance( iterator & itr )
  ///
  /// erase the value of iterator
  /// advance the iterator to the next valid value in the circular_pool
  /// if there are no valid values invalidated the iterator
  template <typename UnaryPredicate>
  static void erase_and_advance( iterator & itr, UnaryPredicate const & pred )
  {
    if ( itr.m_node ) {
      node_type * start = itr.m_node;
      erase( itr );
      itr = find_any( start, pred );
    }
  }

  /// front( node_type * node )
  ///
  /// get an iterator to the first valid value starting at node.
  /// if there are no valid values return an invalid iterator
  static iterator front( node_type * node )
  {
    if ( !node ) return iterator{};

    node_type * start = node;
    node_type * curr = start;
    node_type * prev = curr;

    int idx = capacity;

    do {
      idx = curr->try_atomic_find();
      prev = curr;
      curr = curr->next();
    } while ( idx == capacity && curr != start );

    iterator iter =  (idx < capacity ) ? iterator{ prev, idx } : iterator{};
    std::atomic_thread_fence( std::memory_order_seq_cst );

    return iter;
  }

  /// find_any( node_ptr, predicate)
  ///
  /// return an iterator to a value for which predicate returns true
  /// predicate is a function, functor, or lambda of the form
  /// bool ()( const value_type & )
  /// If there are no values for which predicate is true return an invalid iterator
  template <typename UnaryPredicate>
  static iterator find_any( node_type * node, UnaryPredicate const & pred)
  {
    if ( !node ) return iterator{};

    node_type * start = node;
    node_type * curr = start;
    node_type * prev = curr;

    int idx;
    bool found = false;

    do {
      // invalidate entire node
      bitset_type old_valid = curr->m_valid_bitset.fetch_and( zero, std::memory_order_seq_cst );

      if (old_valid) {

        bitset_type test_valid = old_valid;

        while ( test_valid && !found ) {
          idx = count_trailing_zeros( test_valid );
          found = pred( curr->get(idx) );
          // mask out idx
          test_valid = test_valid & complement(one << idx);
        }
      }

      if (found) {
        // release the other indexes
        // mask out idx
        const bitset_type new_valid = old_valid & complement(one << idx );
        curr->m_valid_bitset.fetch_or( new_valid, std::memory_order_relaxed );
      }
      else {
        // release the indexes
        curr->m_valid_bitset.fetch_or( old_valid, std::memory_order_relaxed );
      }

      prev = curr;
      curr = curr->next();

    } while ( !found && curr != start );

    return ( found ) ? iterator{ prev, idx } : iterator{};
  }

  // get_iterator( node, value_ptr )
  //
  // get an iterator to the value
  iterator get_iterator( const value_type * ptr )
  {
    int idx=0;
    for (; idx<capacity && (m_values + idx != ptr); ++idx) {}

    if ( idx == capacity ) {
      return iterator{};
    }

    const bitset_type bit = one << idx ;

    // while the index is used
    while ( m_used_bitset.load( std::memory_order_relaxed ) & bit ) {
      if ( try_atomic_claim( idx ) ) {
        std::atomic_thread_fence( std::memory_order_seq_cst );
        return iterator{ this, idx };
      }
    }

    return iterator{};
  }

  /// emplace( args... )
  ///
  /// return an iterator to the newly created value
  template <typename NodeAllocator, typename... Args>
  iterator emplace_helper( const int num_search_nodes, NodeAllocator & allocator, int & added_node,  Args&&... args)
  {
    iterator itr;
    node_type * curr;

    int n = 0;

    node_type * const start = this;
    node_type * next = start;

    // try to insert the value into an existing node
    do {
      curr = next;
      next = curr->next();
      itr = curr->try_atomic_emplace( std::forward<Args>(args)... );
    } while ( !itr && (next != start) && ( ++n < num_search_nodes ) );

    // wrapped around the pool
    // Allocate node and insert the value
    if ( !itr ) {
      // allocate and construct
      node_type * new_node = allocator.allocate(1);
      allocator.construct( new_node, m_pool_id, m_pool );

      // will always succeed since the node is not in the pool
      itr = new_node->try_atomic_emplace( std::forward<Args>(args)... );

      // insert the node at the end of pool (curr->next)
      node_type * next;
      do {
        next = curr->next();
        new_node->set_next(next);
      } while ( ! curr->try_update_next( next, new_node ) );

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
  iterator try_atomic_emplace(Args&&... args)
  {
    int idx;
    bool inserted = false;
    do {
      idx = find_unset_idx();
      inserted = (idx < capacity) &&  try_atomic_emplace_helper( idx, std::forward<Args>(args)... );
    } while ( idx < capacity && !inserted );

    return ( idx < capacity ) ? iterator{ this, idx } : iterator{};
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

  /// try_update_next( old_ptr, new_ptr )
  ///
  /// Try to atomically update next with a compare and swap.
  ///
  /// i.e. atomically do the following code
  ///
  /// curr_ptr = m_next;
  /// if ( curr_ptr == old_ptr ) { m_next = new_ptr; }
  /// return curr_ptr == old_ptr
  LOCKFREE_FORCEINLINE
  bool try_update_next( node_type * old_ptr, node_type * new_ptr )
  {
    return m_next.compare_exchange_strong( old_ptr, new_ptr, std::memory_order_relaxed, std::memory_order_relaxed );
  }

  /// set_next( new_ptr )
  ///
  /// NON-ATOMICALLY set next to given pointer
  LOCKFREE_FORCEINLINE
  void set_next( node_type * new_ptr )
  {
    m_next.store( new_ptr, std::memory_order_relaxed );
  }

  /// Construct a node pointing to itself
  Node( size_t pid, void * pool )
    : m_pool_id{ pid }
    , m_pool{ pool }
  {}

  /// Destroy the values in the node
  ~Node()
  {
    const bitset_type curr_valid = m_valid_bitset.fetch_and( zero, std::memory_order_seq_cst );
    for (int i=0; i<capacity; ++i) {
      if ( curr_valid & ( one << i) ) {
        (m_values + i)->~value_type();
      }
    }
    std::atomic_thread_fence( std::memory_order_seq_cst );
  }

  size_t  impl_pool_id() const { return m_pool_id; }
  void *  impl_pool() const { return m_pool; }

private: // private functions


  /// try_atomic_find_next( idx )
  int try_atomic_find_next( int idx )
  {
    bool found = false;
    do {
      idx = find_next_set_idx( idx );
      found = (idx < capacity) &&  try_atomic_claim( idx );
    } while ( idx < capacity && !found );

    return idx;
  }

  /// try_atomic_find()
  int try_atomic_find()
  {
    int idx;
    bool found = false;
    do {
      idx = find_set_idx();
      found = (idx < capacity) &&  try_atomic_claim( idx );
    } while ( idx < capacity && !found );

    return idx;
  }

  /// find_unset_idx()
  ///
  /// return an unused index
  /// if none found return capacity
  LOCKFREE_FORCEINLINE
  int find_unset_idx() const
  {
    // need to static cast to a bitset_type to handle uint16_t and uint8_t
    return count_trailing_zeros( complement(m_used_bitset.load( std::memory_order_relaxed )) );
  }

  /// find_set_idx()
  ///
  /// return the first used index in the node
  /// if none exist return capacity
  LOCKFREE_FORCEINLINE
  int find_set_idx() const
  {
    return count_trailing_zeros( m_valid_bitset.load( std::memory_order_relaxed ) );
  }

  /// find_next_set_idx( idx )
  ///
  /// return the first used index greater than the given index
  /// if none exist return capacity
  LOCKFREE_FORCEINLINE
  int find_next_set_idx( int idx ) const
  {
    // mask out bits less than idx
    return idx+1 < capacity ? count_trailing_zeros( m_valid_bitset.load( std::memory_order_relaxed ) & complement((one << (idx+1)) - one)) : 64;
  }

  /// atomic_erase( idx )
  ///
  /// destroy the value and release the used index
  void atomic_erase( int i )
  {
    const bitset_type bit = one << i ;

    // call the destructor
    (m_values + i)->~value_type();
    // allow index to be used again
    m_used_bitset.fetch_and( complement(bit), std::memory_order_relaxed );
  }

  // return true if this thread atomically set the ith bit
  template <typename... Args>
  bool try_atomic_emplace_helper( int i, Args&&... args)
  {
    const bitset_type bit = one << i ;
    const bool inserted =  !( m_used_bitset.fetch_or(bit, std::memory_order_relaxed) & bit);
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
    const bitset_type bit = one << i ;
    const bool result =  m_valid_bitset.fetch_and( complement(bit), std::memory_order_relaxed ) & bit;
    return result;
  }


  // release the index
  LOCKFREE_FORCEINLINE
  void atomic_release( int i )
  {
    const bitset_type bit = one << i ;
    m_valid_bitset.fetch_or( bit, std::memory_order_relaxed );
  }

private: // data members

  // align to cacheline (64 bytes)
  static constexpr bitset_type offset = sizeof(value_type * )            // m_values
                                      + sizeof(std::atomic<bitset_type>) // m_used_bitset
                                      + sizeof(std::atomic<bitset_type>) // m_valid_bitset
                                      + sizeof(std::atomic<node_type *>) // m_next
                                      + sizeof(size_t)                   // m_pool_id
                                      + sizeof(void *)                   // m_pool
                                      ;
  static constexpr bitset_type alignment = align_to_cacheline<T>();
  static constexpr bitset_type min_buffer_size = capacity * sizeof(value_type);
  static constexpr bitset_type buffer_size = ( alignment * ( min_buffer_size + offset + alignment - one ) / alignment ) - offset;

  char                      m_buffer[ buffer_size ]; // must be first to get correct alignment
  value_type *              m_values{ reinterpret_cast<value_type *>(m_buffer) };
  std::atomic<bitset_type>  m_used_bitset{ 0u };
  std::atomic<bitset_type>  m_valid_bitset{ 0u };
  std::atomic<node_type *>  m_next{ this };
  size_t                    m_pool_id;
  void *                    m_pool;
};

}} // namespace Lockfree::Impl

#endif //LOCKFREE_POOL_NODE_HPP
