#ifndef SCINEW_ALLOCATOR_H
#define SCINEW_ALLOCATOR_H

#include <limits>

namespace Uintah {

template <typename T>
class SCINewAllocator
{
public:
  using size_type = size_t;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;

  template <class U>
  struct rebind
  {
    using other = SCINewAllocator<U>;
  };

  static       pointer address(       reference x ) noexcept { return &x; }
  static const_pointer address( const_reference x ) noexcept { return &x; }

  static constexpr size_type max_size() { return std::numeric_limits<size_t>::max(); }

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
    char * buffer = scinew char[sizeof(T)*n];
    return reinterpret_cast<pointer>(buffer);
  }

  void deallocate( pointer p, size_type n )
  {
    char * buffer = reinterpret_cast<char*>(p);
    delete[] buffer;
  }

private:
};


} // namespace Uintah

#endif //SCINEW_ALLOCATOR_H
