#ifndef LOCKFREE_UNSTRUCTURED_NODE_HPP
#define LOCKFREE_UNSTRUCTURED_NODE_HPP

#include "Lockfree_Macros.hpp"
#include "Lockfree_Bits.hpp"

#include <utility>      // for std::swap
#include <type_traits>

#include <climits>
#include <cstdio>

namespace Lockfree { namespace Impl {

template< typename T, int Alignment>
constexpr int get_alignment()
{
  static_assert( (Alignment > 0) && !(Alignment & (Alignment - 1)), "ERROR: Alignment must be a power of 2.");
  return LOCKFREE_ALIGNOF(T) > Alignment ? LOCKFREE_ALIGNOF(T) : Alignment;
}

//-----------------------------------------------------------------------------
/// class UnstructuredNode
///
/// Fundamental building block for more complex lockfree data structures
//-----------------------------------------------------------------------------
template <typename T, typename BitsetType = uint64_t, int Alignment = 128>
class LOCKFREE_ALIGNAS((get_alignment<T,Alignment>())) UnstructuredNode
{
  static_assert( std::is_unsigned<BitsetType>::value, "ERROR: BitsetType must be an unsigned integer type." );

public:
  using node_type = UnstructuredNode<T, BitsetType, Alignment>;
  using value_type = T;
  using bitset_type = BitsetType;

  static constexpr int capacity = sizeof(bitset_type) * CHAR_BIT;
  static constexpr bitset_type one = 1u;
  static constexpr bitset_type zero = 0u;
  static constexpr int alignment = get_alignment<T,Alignment>();

  /// class iterator
  ///
  /// UnstructuredNode iterators are moveable but not copyable.  When a valid iterator
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
    void operator++() { node_type::advance(*this); }

    // is this iterator valid ?
    LOCKFREE_FORCEINLINE
    explicit operator bool() const { return m_node != nullptr; }


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
      __sync_synchronize();
    }

    // move assignment
    iterator & operator=( iterator && itr )
    {
      // itr is going to be destroy
      // swap values to let it clean up "this" resource
      std::swap( m_node , itr.m_node );
      std::swap( m_idx , itr.m_idx );
      __sync_synchronize();
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
    friend class UnstructuredNode;

    // only UnstructuredNode is allowed to construct a valid iterator
    iterator( node_type * node, int idx )
      : m_node{ node }
      , m_idx{ idx }
    {
      __sync_synchronize();
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


  /// advance( iterator & itr, [start] )
  ///
  /// advance the iterator to the next valid value in the list
  /// if there are no valid values invalidated the iterator
  static void advance( iterator & itr, node_type * start = nullptr )
  {
    if ( !itr.m_node ) { return; }

    // if start not specified uses the iterator as the starting node
    if (start == nullptr) {
      start = itr.m_node;
    }

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
  /// advance the iterator to the next valid value in the list
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

    return (idx < capacity ) ? iterator{ prev, idx } : iterator{};
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
      bitset_type old_valid = __sync_fetch_and_and( &curr->m_valid_bitset, zero );

      if (old_valid) {
        __sync_synchronize();

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
        __sync_fetch_and_or( &curr->m_valid_bitset, new_valid);
      }
      else {
        // release the indexes
        __sync_fetch_and_or( &curr->m_valid_bitset, old_valid);
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
    while ( __sync_fetch_and_or( &m_used_bitset, zero ) & bit ) {
      if ( try_atomic_claim( idx ) ) {
        return iterator{ this, idx };
      }
    }

    return iterator{};
  }

  /// try_atomic_emplace( args... )
  ///
  /// try to create a value on the node
  /// if successful return the index where the value was created
  /// otherwise return capacity
  template <typename... Args>
  iterator try_atomic_emplace(Args... args)
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
    return m_next;
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
  bool try_update_next( node_type * const old_ptr, node_type * const new_ptr )
  {
    return __sync_bool_compare_and_swap( &m_next, old_ptr, new_ptr );
  }

  /// set_next( new_ptr )
  ///
  /// NON-ATOMICALLY set next to given pointer
  LOCKFREE_FORCEINLINE
  void set_next( node_type * new_ptr )
  {
    m_next = new_ptr;
    __sync_synchronize();
  }

  /// Construct a node pointing to itself
  UnstructuredNode()
    : m_buffer{}
    , m_next{ this }
    , m_values{ reinterpret_cast< value_type * >( m_buffer ) }
    , m_used_bitset{ 0u }
    , m_valid_bitset{ 0u }
  {}

  /// Destroy the values in the node
  ~UnstructuredNode()
  {
    const bitset_type curr_valid = __sync_fetch_and_and( &m_valid_bitset, zero );
    for (int i=0; i<capacity; ++i) {
      if ( curr_valid & ( one << i) ) {
        (m_values + i)->~value_type();
      }
    }
    __sync_synchronize();
  }

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
    return count_trailing_zeros(complement(m_used_bitset));
  }

  /// find_set_idx()
  ///
  /// return the first used index in the node
  /// if none exist return capacity
  LOCKFREE_FORCEINLINE
  int find_set_idx() const
  {
    return count_trailing_zeros( m_valid_bitset );
  }

  /// find_next_set_idx( idx )
  ///
  /// return the first used index greater than the given index
  /// if none exist return capacity
  LOCKFREE_FORCEINLINE
  int find_next_set_idx( int idx ) const
  {
    // mask out bits less than idx
    return idx+1 < capacity ? count_trailing_zeros( m_valid_bitset & complement((one << (idx+1)) - one)) : 64;
  }

  /// atomic_erase( idx )
  ///
  /// destroy the value and release the used index
  void atomic_erase( int i )
  {
    const bitset_type bit = one << i ;

    // call the destructor
    (m_values + i)->~value_type();
    // memory fence
    __sync_synchronize();

    // allow index to be used again
    __sync_fetch_and_and( &m_used_bitset, complement(bit) );
  }

  // return true if this thread atomically set the ith bit
  template <typename... Args>
  bool try_atomic_emplace_helper( int i, Args... args)
  {
    const bitset_type bit = one << i ;
    const bool inserted =  !( __sync_fetch_and_or( &m_used_bitset, bit ) & bit);
    if (inserted ) {
      // placement new with constructor
      new ((void*)(m_values + i)) value_type{ std::forward<Args>(args)... } ;
      // memory fence
      __sync_synchronize();
    }
    return inserted;
  }

  // claim the index
  LOCKFREE_FORCEINLINE
  bool try_atomic_claim( int i )
  {
    const bitset_type bit = one << i ;
    const bool result =  __sync_fetch_and_and( &m_valid_bitset, complement(bit) ) & bit;
    return result;
  }


  // release the index
  LOCKFREE_FORCEINLINE
  void atomic_release( int i )
  {
    __sync_synchronize();
    const bitset_type bit = one << i ;
    __sync_fetch_and_or( &m_valid_bitset, bit );
  }

private: // data members

  char          m_buffer[ capacity * sizeof(value_type) ]; // must be first to get correct alignment
  node_type   * m_next;
  value_type  * m_values;
  bitset_type   m_used_bitset;
  bitset_type   m_valid_bitset;
};

}} // namespace Lockfree::Impl

#endif //LOCKFREE_UNSTRUCTURED_NODE_HPP
