#include <sys/mman.h>  // for mmap, munmap, MAP_ANON, etc
#include <new> // for bad_alloc

// mmap flags for private anonymous memory allocation
#if defined( MAP_ANONYMOUS ) && defined( MAP_PRIVATE )
  #define MMAP_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS)
#elif defined( MAP_ANON) && defined( MAP_PRIVATE )
  #define MMAP_FLAGS (MAP_PRIVATE | MAP_ANON)
#else
  #error "ERROR: mmap cannot be used to allocate memory."
#endif

// read write access to private memory
#define MMAP_PROTECTION (PROT_READ | PROT_WRITE)

namespace Lockfree { namespace Impl {

namespace {

constexpr size_t page_size = 4096;

volatile size_t g_mmap_excess = 0;

} // namespace


size_t mmap_excess()
{
  return g_mmap_excess;
}

void * mmap_allocate( size_t num_bytes )
{

  const size_t excess = page_size - (num_bytes & (page_size - 1u));
  __sync_fetch_and_add( &g_mmap_excess, excess );

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

void mmap_deallocate( void * ptr, size_t num_bytes )
{
  const size_t excess = page_size - (num_bytes & (page_size - 1u));
  __sync_fetch_and_sub( &g_mmap_excess, excess );

  munmap(ptr, num_bytes);
}

}} //namespace Lockfree::Impl
