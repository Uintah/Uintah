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

#ifndef UTILITIES_ALIGNED_ALLOCATOR_HPP
#define UTILITIES_ALIGNED_ALLOCATOR_HPP

#include <cstdlib>
#include <cstddef>

namespace Allocators {

template <  typename T, unsigned N>
class AlignedAllocator
{
  static_assert( !(N & N-1u), "Alignment must be a power of 2");

public:

  static constexpr unsigned alignment = N;

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
    using other = AlignedAllocator<U, N>;
  };

  AlignedAllocator() {}
  AlignedAllocator( const AlignedAllocator & ) {}
  AlignedAllocator & operator=( const AlignedAllocator & ) { return *this; }

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
    pointer ptr;
    posix_memalign( reinterpret_cast<void**>(&ptr), N, n * sizeof(value_type) );
    return ptr;
  }

  void deallocate( pointer ptr, size_type )
  {
    free( ptr );
  }

private:
};

} // namespace Allocators

#endif //UTILITIES_ALIGNED_ALLOCATOR_HPP

