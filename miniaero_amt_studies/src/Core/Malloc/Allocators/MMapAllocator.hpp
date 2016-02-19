// The MIT License (MIT)
//
// Copyright (c) 2016 Daniel Sunderland
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef UTILITIES_MMAP_ALLOCATOR_HPP
#define UTILITIES_MMAP_ALLOCATOR_HPP

#include <limits>
#include <cstddef>
#include <utility>

#include <sys/mman.h>  // for mmap, munmap, MAP_ANON, etc
#include <new> // for bad_alloc

namespace Allocators { namespace Impl {

inline
void * mmap_allocate( size_t num_bytes )
{

// mmap flags for private anonymous memory allocation
#if defined( MAP_ANONYMOUS ) && defined( MAP_PRIVATE )
  enum { MMAP_FLAGS = (MAP_PRIVATE | MAP_ANONYMOUS) };
#elif defined( MAP_ANON) && defined( MAP_PRIVATE )
  enum { MMAP_FLAGS = (MAP_PRIVATE | MAP_ANON) };
#else
  #error "ERROR: mmap cannot be used to allocate memory."
#endif

  enum { MMAP_PROTECTION = (PROT_READ | PROT_WRITE) };

  void *ptr = nullptr;
  if (num_bytes) {
    ptr = mmap( nullptr, num_bytes, MMAP_PROTECTION, MMAP_FLAGS, -1 /*file descriptor*/, 0 /*offset*/);
    if (ptr == MAP_FAILED) {
      ptr = nullptr;
      throw std::bad_alloc();
    }
  }

  return ptr;
}

inline
void mmap_deallocate( void * ptr, size_t num_bytes )
{
  munmap(ptr, num_bytes);
}

}} //namespace Allocators::Impl

namespace Allocators {

template <  typename T >
class MMapAllocator
{
public:

  using size_type       = size_t;
  using difference_type = ptrdiff_t;
  using value_type      = T;
  using pointer         = T*;
  using reference       = T&;
  using const_pointer   = const T*;
  using const_reference = const T&;

  template <class U>
  struct rebind
  {
    using other = MMapAllocator<U>;
  };

  MMapAllocator() {}
  MMapAllocator( const MMapAllocator & ) {}
  MMapAllocator & operator=( const MMapAllocator & ) { return *this; }

  static       pointer address(       reference x ) noexcept { return &x; }
  static const_pointer address( const_reference x ) noexcept { return &x; }

  static constexpr size_type max_size() { return std::numeric_limits<size_t>::max(); }

  template <class U, class... Args>
  static void construct (U* ptr, Args&&... args)
  {
    new ((void*)ptr) U( std::forward<Args>(args)... );
  }

  template <class U>
  static void destroy (U* ptr)
  {
    ptr->~U();
  }

  pointer allocate( size_type n, void * = nullptr)
  {
    return reinterpret_cast<pointer>(Impl::mmap_allocate( n * sizeof(value_type) ));
  }

  void deallocate( pointer ptr, size_type n )
  {
    Impl::mmap_deallocate( ptr, n * sizeof(value_type) );
  }

private:
};

} // namespace Allocators

#endif //UTILITIES_MMAP_ALLOCATOR_HPP
