

#ifndef Malloc_Allocator_h
#define Malloc_Allocator_h 1

#include <unistd.h>

struct Allocator;
Allocator* MakeAllocator();
void DestroyAllocator(Allocator*);

extern Allocator* default_allocator;
void MakeDefaultAllocator();

void PrintTag(void*);

Allocator* DefaultAllocator();
void GetGlobalStats(Allocator*,
		    size_t& nalloc, size_t& sizealloc,
		    size_t& nfree, size_t& sizefree,
		    size_t& nfillbin,
		    size_t& nmmap, size_t& sizemmap,
		    size_t& nmunmap, size_t& sizemunmap,
		    size_t& highwater_alloc, size_t& highwater_mmap,
		    size_t& nlonglocks, size_t& nnaps,
		    size_t& bytes_overhead,
		    size_t& bytes_free,
		    size_t& bytes_fragmented,
		    size_t& bytes_inuse,
		    size_t& bytes_inhunks);
int GetNbins(Allocator*);
void GetBinStats(Allocator*, int binno, size_t& minsize, size_t& maxsize,
		 size_t& nalloc, size_t& nfree, size_t& ninlist);

void AuditAllocator(Allocator*);
void DumpAllocator(Allocator*);

void* operator new(size_t, Allocator*, char*);
void* operator new[](size_t, Allocator*, char*);

#ifdef __GNUG__
#define scinew new
#else
#define scinew new(default_allocator, __FILE__)
#endif

#endif
