
#include <Allocator.h>
#include <AllocPriv.h>

void* malloc(size_t size)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "Unknown - malloc");
}

void free(void* ptr)
{
    default_allocator->free(ptr);
}

void calloc(size_t)
{
    AllocError("calloc not finished");
}

void* realloc(void* p, size_t s)
{
    return default_allocator->realloc(p, s);
}

void* memalign(size_t, size_t)
{
    AllocError("memalign not finished");
    return 0;
}

void* valloc(size_t)
{
    AllocError("valloc not finished");
    return 0;
}
