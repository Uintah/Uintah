
/*
 *  AllocPriv.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <stdlib.h>
#include <stdio.h>

namespace SCICore {
namespace Malloc {

struct OSHunk;

struct Sentinel {
    unsigned int first_word;
    unsigned int second_word;
};

struct AllocBin;

struct Tag {
//    Allocator* allocator;
//    size_t size;
    AllocBin* bin;
    const char* tag;
    Tag* next;
    Tag* prev;
    OSHunk* hunk;
    size_t reqsize;
};

struct AllocBin {
    Tag* free;
    Tag* inuse;
    size_t maxsize;
    size_t minsize;
    int ninuse;
    int ntotal;
    size_t nalloc;
    size_t nfree;
};

struct Allocator {
    unsigned int the_lock;
    void initlock();
    inline void lock();
    void longlock();
    inline void unlock();

    void* alloc_big(size_t size, const char* tag);
    
    void* memalign(size_t alignment, size_t size, const char* tag);
    void* alloc(size_t size, const char* tag);
    void free(void*);
    void* realloc(void* p, size_t size);

    int strict;
    int lazy;
    FILE* trace_out;
    FILE* stats_out;
    OSHunk* hunks;

    AllocBin* small_bins;
    AllocBin* medium_bins;
    AllocBin big_bin;

    inline AllocBin* get_bin(size_t size);
    void fill_bin(AllocBin*);
    void get_hunk(size_t, OSHunk*&, void*&);

    void init_bin(AllocBin*, size_t maxsize, size_t minsize);

    void audit(Tag*, int);
    size_t obj_maxsize(Tag*);

    // Statistics...
    size_t nalloc;
    size_t nfree;
    size_t sizealloc;
    size_t sizefree;
    size_t nlonglocks;
    size_t nnaps;

    size_t nfillbin;
    size_t nmmap;
    size_t sizemmap;
    size_t nmunmap;
    size_t sizemunmap;

    size_t highwater_alloc;
    size_t highwater_mmap;

    size_t mysize;
};

void AllocError(char*);

} // End namespace Malloc
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/09/30 00:33:46  sparker
// Added support for memalign (got orphaned in move from SCIRun)
//
// Revision 1.3  1999/09/25 08:30:54  sparker
// Added support for MALLOC_STATS environment variable.  If set, it will
//   send statistics about malloc and a list of all unfreed objects at
//   program shutdown.  It isn't perfect yet - I need to figure out how
//   to make it run after the C++ dtors.
//
// Revision 1.2  1999/09/17 05:04:43  sparker
// Enhanced malloc tracing facility.  You can now set the environment
// variable MALLOC_TRACE to a filename, where the allocator will dump
// all calls to alloc/free.  It will also dump any unfreed objects
// and some statistics at the end of the program.  Setting MALLOC_TRACE
// to an empty string sends this information to stderr.
//
// Revision 1.1  1999/07/27 16:56:58  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:52:10  dav
// adding .h files back to src tree
//
// Revision 1.1  1999/05/05 21:05:20  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//

