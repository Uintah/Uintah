
#include <Allocator.h>
#include <AllocPriv.h>
#include <bstring.h>

extern "C" {
void* malloc(size_t size);
void free(void* ptr);
void* calloc(size_t, size_t);
void* realloc(void* p, size_t s);
void* memalign(size_t, size_t);
void* valloc(size_t);
};

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

void* calloc(size_t n, size_t s)
{
    size_t tsize=n*s;
    void* p=malloc(tsize);
    bzero(p, tsize);
    return p;
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
