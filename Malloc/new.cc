
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
    if(!default_allocator)
	MakeDefaultAllocator();
    default_allocator->free(ptr);
}

void* operator new(size_t size, Allocator* a, const char* tag)
{
    if(!a){
	if(!default_allocator)
	    MakeDefaultAllocator();
	a=default_allocator;
    }
    return a->alloc(size, tag);
}

void* operator new[](size_t size, Allocator* a, const char* tag)
{
    if(!a){
	if(!default_allocator)
	    MakeDefaultAllocator();
	a=default_allocator;
    }
    return a->alloc(size, tag);
}

