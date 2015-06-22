#ifndef LOCKFREE_UNSTRUCTURED_LIST
#define LOCKFREE_UNSTRUCTURED_LIST

#include "impl/Lockfree_Macros.hpp"
#include "impl/Lockfree_UnstructuredNode.hpp"

#include "Lockfree_UsageModel.hpp"


#include <array>
#include <memory> // for std::allocator


namespace Lockfree {

// Copies of list are shallow,  i.e., they point to the same reference counted memory.
// So each thread should have its own copy of the list
// It is not thread safe for multiple threads to interact with the same instance of a list
template <  typename T
          , UsageModel Model = SHARED_INSTANCE
          , template <typename> class Allocator = std::allocator
          , template <typename> class SizeTypeAllocator = std::allocator
          , typename BitsetType = uint64_t
          , int Alignment = 16
         >
class UnstructuredList
{
public:
  using size_type = size_t;
  using value_type = T;
  using bitset_type = BitsetType;
  static constexpr int alignment = Alignment;
  static constexpr UsageModel usage_model = Model;
  template <typename U> using allocator = Allocator<U>;

  using list_type = UnstructuredList<  T
                                     , Model
                                     , Allocator
                                     , SizeTypeAllocator
                                     , BitsetType
                                     , Alignment
                                    >;

  using impl_node_type = Impl::UnstructuredNode<T, BitsetType, Alignment>;

  using node_allocator_type   = allocator<impl_node_type>;
  using size_t_allocator_type = SizeTypeAllocator<size_type>;

private:


  static constexpr size_type one = 1;
  static constexpr impl_node_type * null_node = nullptr;

  enum {
      SHARED_SIZE
    , SHARED_NUM_NODES
    , SHARED_REF_COUNT
    , SHARED_LENGTH
  };



public:
  using iterator = typename impl_node_type::iterator;

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
  iterator emplace(Args... args)
  {
    iterator itr;

    __sync_fetch_and_add( (m_shared + SHARED_SIZE), one );

    impl_node_type * const start = get_insert_head();
    impl_node_type * curr = start;
    impl_node_type * prev = curr;

    // try to insert the value into an existing node
    do {
      itr = curr->try_atomic_emplace( std::forward<Args>(args)... );
      prev = curr;
      curr = curr->next();
    } while ( !itr && (curr != start) && (100ull*size() < 95ull*capacity()) );

    // wrapped around the list
    // Allocate node and insert the value
    if ( !itr ) {
      // allocate and construct
      impl_node_type * new_node = m_node_allocator.allocate(1);
      m_node_allocator.construct( new_node );

      __sync_fetch_and_add( (m_shared + SHARED_NUM_NODES), 1 );

      // will always succeed since the node is not in the list
      itr = new_node->try_atomic_emplace( std::forward<Args>(args)... );

      // insert the node at the end of list (prev->next)
      impl_node_type * next;
      do {
        next = prev->next();
        new_node->set_next(next);
      } while ( ! prev->try_update_next( next, new_node ) );

      prev = new_node;
    }

    try_set_insert_head( prev );

    return itr;
  }

  /// front()
  ///
  /// return an iterator to the front of the list,
  /// may return an invalid iterator
  iterator front() const
  {
    iterator itr = impl_node_type::front( get_find_head() );

    // try to set front to itr
    if ( itr ) {
      try_set_find_head( impl_node_type::get_node(itr) );
    }

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
    iterator itr = impl_node_type::find_any( get_find_head(), pred );

    // try to set front to itr
    if ( itr ) {
      try_set_find_head( impl_node_type::get_node(itr) );
    }

    return itr;
  }

  /// erase( iterator )
  ///
  /// if the iterator is valid erase its value
  void erase( iterator & itr )
  {
    if ( itr ) {
      impl_node_type::erase( itr );
      __sync_sub_and_fetch( (m_shared + SHARED_SIZE), one );
    }
  }

  /// erase_and_advance( iterator, pred )
  ///
  /// if the iterator is valid erase its current value
  /// advance it to the next value for which predicate is true
  template <typename UnaryPredicate>
  void erase_and_advance( iterator & itr, UnaryPredicate const & pred )
  {
    if ( itr ) {
      impl_node_type::erase_and_advance( itr, pred );
      __sync_sub_and_fetch( (m_shared + SHARED_SIZE), one );
    }

    // set the front of the list to the new iterator
    if ( itr ) {
      try_set_find_head( impl_node_type::get_node(itr) );
    }
  }

  /// erase_and_advance( iterator )
  ///
  /// if the iterator is valid erase its current value
  /// advance it to the next value
  void erase_and_advance( iterator & itr )
  {
    auto get_next = []( value_type const& )->bool { return true; };
    erase_and_advance( itr, get_next );
  }

  /// size()
  ///
  /// number of values currently in the list
  LOCKFREE_FORCEINLINE
  size_type size() const
  {
    return m_shared[SHARED_SIZE];
  }

  /// capacity()
  ///
  /// number of values the list can currently hold
  LOCKFREE_FORCEINLINE
  size_type capacity() const
  {
    return impl_node_type::capacity * num_nodes();
  }

  /// num_nodes()
  ///
  /// number of nodes in the list
  LOCKFREE_FORCEINLINE
  size_type num_nodes() const
  {
    return m_shared[SHARED_NUM_NODES];
  }

  /// num_bytes()
  ///
  /// number of bytes the list is currently using
  LOCKFREE_FORCEINLINE
  size_type num_bytes() const
  {
    return sizeof(impl_node_type) * num_nodes() + sizeof(list_type) * ref_count();
  }

  /// ref_count()
  ///
  /// number of references to the list
  size_type ref_count() const
  {
    return m_shared[SHARED_REF_COUNT];
  }

  /// empty()
  ///
  /// is the list empty
  LOCKFREE_FORCEINLINE
  bool empty() const
  {
    return size() == 0u;
  }


  /// advance_head
  ///
  /// advance the head of the list by n nodes
  void advance_head( const size_type n )
  {
    impl_node_type * start = get_insert_head();
    for (size_type i=0; i<n; ++i) {
      start = start->next();
    }
    try_set_insert_head( start );
    try_set_find_head( start );
  }

  /// Contruct a list
  UnstructuredList(  node_allocator_type   arg_node_allocator   = node_allocator_type{}
                   , size_t_allocator_type arg_size_t_allocator = size_t_allocator_type{}
                  )
    : m_insert_head{}
    , m_find_head{}
    , m_shared{}
    , m_node_allocator{ arg_node_allocator }
    , m_shared_allocator{ arg_size_t_allocator }
  {
    {
      m_insert_head = m_node_allocator.allocate(1);
      m_node_allocator.construct( m_insert_head );
      m_find_head = m_insert_head;
    }

    {
      m_shared = m_shared_allocator.allocate(SHARED_LENGTH);
      m_shared[SHARED_SIZE] = 0;
      m_shared[SHARED_NUM_NODES] = 1;
      m_shared[SHARED_REF_COUNT] = 1;
    }
    __sync_synchronize();
  }

  // shallow copy with a hint on how many task this thread will insert
  UnstructuredList( UnstructuredList const & rhs, const size_type num_insert_hint = 0  )
    : m_insert_head{ rhs.m_insert_head }
    , m_find_head{ rhs.m_find_head }
    , m_shared{ rhs.m_shared }
    , m_node_allocator{ rhs.m_node_allocator }
    , m_shared_allocator{ rhs.m_shared_allocator }
  {
    __sync_fetch_and_add( (m_shared + SHARED_REF_COUNT), one );

    const size_type num_insert_nodes = (num_insert_hint + impl_node_type::capacity - one) / impl_node_type::capacity;

    if ( num_insert_nodes > 0u ) {
      impl_node_type * start = m_node_allocator.allocate(1);
      m_node_allocator.construct( start );

      impl_node_type * curr = start;

      // create new nodes
      for (size_type i=1; i<num_insert_nodes; ++i) {
        impl_node_type * new_node = m_node_allocator.allocate(1);
        m_node_allocator.construct( new_node );

        curr->set_next( new_node );
        curr = new_node;
      }

      // set the head of the list to a newly created node(s)
      m_insert_head = start;
      m_find_head = start;

      // memory fence
      __sync_synchronize();

      // add all the new nodes to the list
      impl_node_type * head = rhs.get_insert_head();
      impl_node_type * next;
      do {
        next = head->next();
        curr->set_next( next );
      } while ( ! head->try_update_next( next, start ) );

      __sync_fetch_and_add( (m_shared + SHARED_NUM_NODES), num_insert_nodes );
    }
    else {
      advance_head( ref_count() );
    }
  }


  // shallow copy
  UnstructuredList & operator=( UnstructuredList const & rhs )
  {
    // check for self assignment
    if ( this != & rhs ) {
      m_insert_head      = rhs.m_insert_head;
      m_find_head        = rhs.m_find_head;
      m_shared           = rhs.m_shared;
      m_node_allocator   = rhs.m_node_allocator;
      m_shared_allocator = rhs.m_shared_allocator;

      size_t rcount = __sync_add_and_fetch( (m_shared + SHARED_REF_COUNT), one );
      advance_head( rcount );
    }

    return *this;
  }

  // move constructor
  UnstructuredList( UnstructuredList && rhs )
    : m_insert_head{ std::move( rhs.m_insert_head ) }
    , m_find_head{ std::move( rhs.m_find_head ) }
    , m_shared{ std::move( rhs.m_shared ) }
    , m_node_allocator{ std::move( rhs.m_node_allocator ) }
    , m_shared_allocator{ std::move( rhs.m_shared_allocator ) }
  {
    // invalidate rhs
    rhs.m_insert_head      = nullptr;
    rhs.m_find_head        = nullptr;
    rhs.m_shared           = nullptr;
    rhs.m_node_allocator   = node_allocator_type{};
    rhs.m_shared_allocator = size_t_allocator_type{};
  }

  // move assignement
  //
  // NOT thread safe if UsageModel is SHARED_INSTANCE
  UnstructuredList & operator=( UnstructuredList && rhs )
  {
    std::swap( m_insert_head, rhs.m_insert_head );
    std::swap( m_find_head, rhs.m_find_head );
    std::swap( m_shared, rhs.m_shared );
    std::swap( m_node_allocator, rhs.m_node_allocator );
    std::swap( m_shared_allocator, rhs.m_shared_allocator );

    return *this;
  }

  ~UnstructuredList()
  {
    if ( m_shared &&  __sync_sub_and_fetch( (m_shared+SHARED_REF_COUNT), one ) == 0u ) {

      impl_node_type * start = get_insert_head();
      impl_node_type * curr = start;
      impl_node_type * next;

      // iterate circular list deleting nodes
      do {
        next = curr->next();
        m_node_allocator.destroy(curr);
        m_node_allocator.deallocate( curr, 1 );
        curr = next;
      } while ( curr != start );

      m_shared_allocator.deallocate( m_shared, SHARED_LENGTH );
    }
  }

private: // member functions

  LOCKFREE_FORCEINLINE
  impl_node_type * get_insert_head() const
  {
    if (usage_model == EXCLUSIVE_INSTANCE) {
      return m_insert_head;
    }
    return __sync_val_compare_and_swap( &m_insert_head, null_node, null_node );
  }

  LOCKFREE_FORCEINLINE
  impl_node_type * get_find_head() const
  {
    if (usage_model == EXCLUSIVE_INSTANCE) {
      return m_find_head;
    }
    return __sync_val_compare_and_swap( &m_find_head, null_node, null_node );
  }

  LOCKFREE_FORCEINLINE
  void try_set_insert_head( impl_node_type * new_head ) const
  {
    if (usage_model == EXCLUSIVE_INSTANCE) {
      m_insert_head = new_head;
    }
    else {
      impl_node_type * prev_head = get_insert_head();
      __sync_bool_compare_and_swap( &m_insert_head, prev_head, new_head );
    }
  }

  LOCKFREE_FORCEINLINE
  void try_set_find_head( impl_node_type * new_head ) const
  {
    if (usage_model == EXCLUSIVE_INSTANCE) {
      m_find_head = new_head;
    }
    else {
      impl_node_type * prev_head = get_find_head();
      __sync_bool_compare_and_swap( &m_find_head, prev_head, new_head );
    }
  }

private: // data members

  mutable impl_node_type * m_insert_head;
  mutable impl_node_type * m_find_head;
  size_type              * m_shared;
  node_allocator_type      m_node_allocator;
  size_t_allocator_type    m_shared_allocator;
};


} // namespace Lockfree


#endif //LOCKFREE_UNSTRUCTURED_LIST
