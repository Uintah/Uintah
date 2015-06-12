#ifndef LOCKFREE_POOL_ALLOCATOR_HPP
#define LOCKFREE_POOL_ALLOCATOR_HPP

#include "Lockfree_UnorderedList.hpp"

#include "impl/Lockfree_Macros.hpp"

#include <stdexcept> // for runtime_error
#include <new> // for bad_alloc

namespace Lockfree {

template <  typename T
          , int Alignment = 16
          , typename BitsetType = uint64_t
          , template <typename> class BaseAllocator = std::allocator
          , template <typename> class SizeTypeAllocator = std::allocator
        >
class PoolAllocator
{
  static_assert( std::is_unsigned<BitsetType>::value, "ERROR: BitsetType must be an unsigned integer type." );

  struct LOCKFREE_ALIGNAS(Alignment) Node
  {
    static constexpr size_t alignment = Alignment ;
    static constexpr size_t buffer_size =  (alignment * ((sizeof(T) + sizeof(void *) - 1u + alignment) / alignment)) - sizeof(void*);

    char   m_buffer[buffer_size];
    void * m_list_node;
  };

  using list_type = UnorderedList< Node, SHARED_INSTANCE, BitsetType, Alignment, size_t, BaseAllocator, SizeTypeAllocator >;
  using list_node_type = typename list_type::impl_node_type;

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
    using other = PoolAllocator<U, Alignment, BitsetType, BaseAllocator, SizeTypeAllocator>;
  };

  PoolAllocator()
    : m_list{s_list}
  {
    printf("Node Size: %d\n", (int)(sizeof(Node)) );
  }

  PoolAllocator( const PoolAllocator & )
    : m_list{s_list}
  {}

  PoolAllocator & operator=( const PoolAllocator & ) {}
  ~PoolAllocator() {}

  static       pointer address(       reference x ) noexcept { return &x; }
  static const_pointer address( const_reference x ) noexcept { return &x; }

  static constexpr size_type max_size() { return 1u; }

  template <class U, class... Args>
  static void construct (U* p, Args&&... args)
  {
    new ((void*)p) U( std::forward<Args>(args)... );
  }

  template <class U>
  static void destroy (U* p)
  {
    p->~U();
  }

  pointer allocate( size_type n, void * = nullptr)
  {
    if (n > max_size() ) {
      throw std::runtime_error("Error: Pool allocator cannot allocate arrays.");
    }

    typename list_type::iterator itr = m_list.emplace();

    if (!itr) {
      throw std::bad_alloc();
    }

    // set the list node to allow O(1) deallocate
    itr->m_list_node = reinterpret_cast<void*>(list_node_type::get_node(itr));
    __sync_synchronize();

    return reinterpret_cast<pointer>(itr->m_buffer);
  }

  void deallocate( pointer p, size_type n )
  {
    if (n > max_size() ) {
      printf("Error: Pool allocator cannot deallocate arrays.");
      return;
    }

    Node * node = reinterpret_cast<Node *>(p);
    list_node_type * list_node = reinterpret_cast<list_node_type *>(node->m_list_node);
    typename list_type::iterator itr = list_node->get_iterator( node );

    if (itr) {
      m_list.erase(itr);
    }
    else {
      printf("Error: double deallocate.");
    }
  }

private:
  static list_type s_list;
  list_type        m_list;
};

template <  typename T
          , int Alignment
          , typename BitsetType
          , template <typename> class BaseAllocator
          , template <typename> class SizeTypeAllocator
        >
typename PoolAllocator<T, Alignment, BitsetType, BaseAllocator, SizeTypeAllocator>::list_type
PoolAllocator<T, Alignment, BitsetType, BaseAllocator, SizeTypeAllocator>::s_list{};

} // namespace Lockfree

#endif // LOCKFREE_POOL_ALLOCATOR_HPP
