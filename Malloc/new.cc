
#include <Allocator.h>
#include <AllocPriv.h>

void* operator new(size_t size)
{
    if(!default_allocator)
	MakeDefaultAllocator();
    return default_allocator->alloc(size, "unknown - operator new");
}

void operator delete(void* ptr)
{
    default_allocator->free(ptr);
}

void* operator new(size_t size, Allocator* a, char* tag)
{
    return a->alloc(size, tag);
}

void* operator new[](size_t size, Allocator* a, char* tag)
{
    return a->alloc(size, tag);
}

