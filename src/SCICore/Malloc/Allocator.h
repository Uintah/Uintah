
/*
 *  Allocator.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Malloc_Allocator_h
#define Malloc_Allocator_h 1

#include <unistd.h>
#include <stdio.h> // ??? How else to get size_t defined?

namespace SCICore {
namespace Malloc {

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

} // End namespace Malloc
} // End namespace SCICore

void* operator new(size_t, SCICore::Malloc::Allocator*, char*);
void* operator new[](size_t, SCICore::Malloc::Allocator*, char*);

#define scinew new(SCICore::Malloc::default_allocator, __FILE__)

//
// $Log$
// Revision 1.1  1999/07/27 16:56:59  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:52:10  dav
// adding .h files back to src tree
//
// Revision 1.1  1999/05/05 21:05:21  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//

#endif
