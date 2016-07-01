/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_CORE_UTIL_NEW_HPP
#define UINTAH_CORE_UTIL_NEW_HPP

#include <atomic>      // atomic
#include <cstddef>     // size_t
#include <cstdlib>     // malloc, posix_memalign, free
#include <limits>      // numeric_limits
#include <memory>      // addressof
#include <type_traits> // is_same

#define CREATE_ALLOCATION_TAG( tag, string_literal )        \
struct tag {                                                \
  static constexpr const char*  name() { return string_literal; } \
}

namespace Uintah { namespace Impl {

//-----------------------------------------------------------------------------
// Used to track memory for the given process
//-----------------------------------------------------------------------------
CREATE_ALLOCATION_TAG( ProcessTag, "Process" );

//-----------------------------------------------------------------------------
// class TagStats
// Tracks allocated memory and highwater mark for the given Tag
//-----------------------------------------------------------------------------
template <typename Tag>
struct TagStats
{
  // place each on its own cachline to avoid false sharing between atomic ops
  enum { ALLOC = 0
       , REGISTERED = 1
       , HIGH_WATER = 8
       , SIZE = 16
       };

  static void allocate( const size_t n ) noexcept
  {
    // prevent a race condition on registering the tag
    constexpr size_t one = 1;
    if ( s_data[REGISTERED].exchange( one, std::memory_order_relaxed ) != one ) {
      // TODO: register the tag with the MPI reporting
      // MPI_Report::register_tag( ... );
    }

    const size_t curr_memory = n + s_data[ALLOC].fetch_add( n, std::memory_order_relaxed );
    size_t high_water = s_data[HIGH_WATER].load( std::memory_order_relaxed );

    while (     high_water < curr_memory
            && !s_data[HIGH_WATER].compare_exchange_weak( high_water, curr_memory, std::memory_order_relaxed )
          )
    {}
  }

  static void deallocate( const size_t n ) noexcept
  {
    s_data[ALLOC].fetch_sub(n, std::memory_order_relaxed );
  }

  static size_t alloc_bytes()      { return s_data[ALLOC].load( std::memory_order_relaxed ); }
  static size_t high_water_bytes() { return s_data[HIGH_WATER].load( std::memory_order_relaxed ); }

  static std::atomic<size_t> s_data[SIZE];
};
template <typename Tag> std::atomic<size_t> TagStats<Tag>::s_data[SIZE] = {};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

// used to manipulate a parameter pack
template <typename... Types> struct TypeList {};

//-----------------------------------------------------------------------------
// tag_alloc
//-----------------------------------------------------------------------------
template <typename T, typename... TT>
inline void tag_alloc( const size_t n, TypeList<T, TT...> )
{
  if (!std::is_same<ProcessTag,T>::value) {
    TagStats<T>::allocate(n);
  }
  tag_alloc(n, TypeList<TT...>{});
}

inline void tag_alloc( const size_t n, TypeList<> )
{
  TagStats<ProcessTag>::allocate(n);
}

//-----------------------------------------------------------------------------
// tag_dealloc
//-----------------------------------------------------------------------------
template <typename T, typename... TT>
inline void tag_dealloc( const size_t n, TypeList<T, TT...> )
{
  if (!std::is_same<ProcessTag,T>::value) {
    TagStats<T>::deallocate(n);
  }
  tag_dealloc(n, TypeList<TT...>{});
}

inline void tag_dealloc( const size_t n, TypeList<> )
{
  TagStats<ProcessTag>::deallocate(n);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// class TagsRef
//   allows user oto Tag parameter pack in Delete
//-----------------------------------------------------------------------------
struct TagsRefBase
{
  TagsRefBase( size_t bytes, size_t size )
    : m_bytes( bytes )
    , m_size( size )
  {}

  virtual ~TagsRefBase() {}

  size_t m_bytes;
  size_t m_size;
};

template <typename... Tags>
struct TagsRef
  : public TagsRefBase
{
  using tags = TypeList<Tags...>;

  TagsRef( size_t bytes, size_t size )
    : TagsRefBase(bytes, size)
  {
    tag_alloc( bytes, tags{} );
  }

  ~TagsRef()
  {
    tag_dealloc( TagsRefBase::m_bytes, tags{} );
  }
};
//-----------------------------------------------------------------------------

}} // namespace Uintah::Impl


namespace Uintah {

//-----------------------------------------------------------------------------
// New
//   T* ptr = New<T,Tags...>(args...)
//-----------------------------------------------------------------------------
template <typename T, typename... Tags, typename... Args>
T * New(Args &&... args)
{
  using tags_ref = Impl::TagsRef<Tags...>;

  static constexpr size_t alignment = 64u;
  const size_t bytes = alignment + sizeof(T);

  char * buffer;
  posix_memalign( (void**)(&buffer), alignment, bytes );

  // track the tags used
  new (reinterpret_cast<tags_ref*>(buffer)) tags_ref{ bytes, 1u };

  // construct the object
  T* ptr = reinterpret_cast<T*>(buffer+alignment);
  new ((void*)ptr) T( std::forward<Args>(args)... );

  return ptr;
}

//-----------------------------------------------------------------------------
// NewArray
//   T* ptr = NewArray<T,Tags...>(n, args...)
//-----------------------------------------------------------------------------
template <typename T, typename... Tags, typename... Args>
T * NewArray(const size_t n, Args &&... args)
{
  using tags_ref = Impl::TagsRef<Tags...>;

  static constexpr size_t alignment = 64u;
  const size_t bytes = alignment + n*sizeof(T);

  char * buffer;
  posix_memalign( (void**)(&buffer), alignment, bytes );

  // track the tags used
  new (reinterpret_cast<tags_ref*>(buffer)) tags_ref{ bytes, n };

  // construct the objects
  T* ptr = reinterpret_cast<T*>(buffer+alignment);
  for (size_t i=0; i<n; ++i) {
    new ((void*)(ptr+i)) T( args... );
  }

  return ptr;
}

//-----------------------------------------------------------------------------
// NewArrayInit
//   T* ptr = NewArrayInit<T,Tags...>(n, init_lambda)
//   T* ptr = NewArrayInit<T,Tags...>(n, [&i](T * ptr) { *ptr = i++; } )
//-----------------------------------------------------------------------------
template <typename T, typename... Tags, typename Init>
T * NewArrayInit(const size_t n, const Init & init)
{
  using tags_ref = Impl::TagsRef<Tags...>;

  static constexpr size_t alignment = 64u;
  const size_t bytes = alignment + n*sizeof(T);

  char * buffer;
  posix_memalign( (void**)(&buffer), alignment, bytes );

  // track the tags used
  new (reinterpret_cast<tags_ref*>(buffer)) tags_ref{ bytes, n };

  // construct the objects
  T* ptr = reinterpret_cast<T*>(buffer+alignment);
  for (size_t i=0; i<n; ++i) {
    init(ptr+i);
  }

  return ptr;
}

//-----------------------------------------------------------------------------
// Delete
//-----------------------------------------------------------------------------
template <typename T>
void Delete(T* ptr)
{
  static constexpr size_t alignment = 64u;
  char * buffer = reinterpret_cast<char*>(ptr) - alignment;

  using base_type = Impl::TagsRefBase;
  //TagsRefBase
  base_type * ref_base = reinterpret_cast<base_type*>(buffer);

  // invoke destructor
  const size_t n = ref_base->m_size;
  for (size_t i=0; i<n; ++i) {
    (ptr+i)->~T();
  }

  //invoke TagsRefBase destructor
  ref_base->~base_type();

  free(buffer);
}

//-----------------------------------------------------------------------------
// Allocator
//-----------------------------------------------------------------------------
template <   typename T
           , typename... Tags
         >
struct Allocator
{
  using tags = Impl::TypeList<Tags...>;

  using size_type       = size_t;
  using difference_type = ptrdiff_t;
  using value_type      = T;
  using pointer         = typename std::add_pointer<T>::type;
  using reference       = typename std::add_lvalue_reference<T>::type;
  using const_pointer   = typename std::add_pointer<const T>;
  using const_reference = typename std::add_lvalue_reference<const T>;

  template <class U>
  struct rebind
  {
    using other = Allocator< U, Tags... >;
  };

  static       pointer address(       reference x ) noexcept { return std::addressof(x); }
  static const_pointer address( const_reference x ) noexcept { return std::addressof(x); }

  static constexpr size_type max_size() { return std::numeric_limits<T>::max(); }

  template <class U, class... Args>
  static void construct (U* p, Args&&... args)
  {
    ::new ((void*)p) U( std::forward<Args>(args)... );
  }

  template <class U>
  static void destroy (U* p)
  {
    p->~U();
  }

  static pointer allocate( size_type n, void * hint = nullptr)
  {
    const size_t bytes = n * sizeof(value_type);
    Impl::tag_alloc( bytes, tags{} );
    return (pointer)malloc(bytes);
  }

  static void deallocate( pointer ptr, size_type n )
  {
    const size_t bytes = n * sizeof(value_type);
    Impl::tag_dealloc( bytes, tags{} );
    free( ptr );
  }
};


//-----------------------------------------------------------------------------
// AlignedAllocator
//-----------------------------------------------------------------------------
template <   typename T
           , size_t Alignment // must be a power of 2
           , typename... Tags
         >
struct AlignedAllocator
{
  static constexpr size_t alignment = Alignment;

  using tags = Impl::TypeList<Tags...>;

  using size_type       = size_t;
  using difference_type = ptrdiff_t;
  using value_type      = T;
  using pointer         = typename std::add_pointer<T>::type;
  using reference       = typename std::add_lvalue_reference<T>::type;
  using const_pointer   = typename std::add_pointer<const T>;
  using const_reference = typename std::add_lvalue_reference<const T>;

  template <class U>
  struct rebind
  {
    using other = AlignedAllocator< U, Alignment, Tags... >;
  };

  static       pointer address(       reference x ) noexcept { return std::addressof(x); }
  static const_pointer address( const_reference x ) noexcept { return std::addressof(x); }

  static constexpr size_type max_size() { return std::numeric_limits<T>::max(); }

  template <class U, class... Args>
  static void construct (U* p, Args&&... args)
  {
    ::new ((void*)p) U( std::forward<Args>(args)... );
  }

  template <class U>
  static void destroy (U* p)
  {
    p->~U();
  }

  static pointer allocate( size_type n, void * hint = nullptr)
  {
    const size_t bytes = n * sizeof(value_type);
    Impl::tag_alloc( bytes, tags{} );
    pointer ptr = nullptr;
    posix_memalign( (void**)(&ptr), alignment, bytes);
    return ptr;
  }

  static void deallocate( pointer ptr, size_type n )
  {
    const size_t bytes = n * sizeof(value_type);
    Impl::tag_dealloc( bytes, tags{} );
    free( ptr );
  }
};

} // namespace Uintah

#endif //UINTAH_CORE_UTIL_NEW_HPP
