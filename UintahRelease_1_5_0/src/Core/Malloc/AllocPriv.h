/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  AllocPriv.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 */

#include <sci_defs/thread_defs.h>
#include <sci_defs/malloc_defs.h>

#include <cstdlib>
#include <stdio.h>

#ifdef SCI_PTHREAD
#  include <pthread.h>
#else
#  ifdef __sgi
#    include <abi_mutex.h>
#  else
#    if !defined(SCI_NOTHREAD) && !defined(_WIN32)
#      error "No lock implementation for this architecture"
#    endif
#  endif
#endif

namespace SCIRun {

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
#ifdef USE_TAG_LINENUM
  int linenum;
#endif
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
#ifdef SCI_PTHREAD
   pthread_mutex_t the_lock;
#else
# ifdef __sgi
   abilock_t the_lock;
# endif
#endif
    void initlock();
    inline void lock();
    inline void unlock();
  void noninline_unlock();

#ifdef SCI_PTHREAD
    inline void rlock();
    // These (dont_lock et.al.) are added in an attempt to deal with some
    // bugs with current versions of glibc in linux.  If and when they get
    // resolved, this code should be removed.  The bug relates to mishandling
    // of mutex's accross fork calls.
    // This variable is initialized in initlock.
    //   James Bigler - 02/04/2003
    bool use_rlock;
    // Current thread that has the lock
    // I'm not sure what to initialize this to, but 0 seems a close enough bet
    pthread_t owner;
    bool owner_initialized;
    // Number of locks held by owner
    int lock_count;

#endif
  
    void* alloc_big(size_t size, const char* tag, int linenum);
    
    void* memalign(size_t alignment, size_t size, const char* tag);
    void* alloc(size_t size, const char* tag, int linenum);
#ifdef MALLOC_TRACE
#  include <MallocTraceOff.h>
#endif
    void free(void*);
    void* realloc(void* p, size_t size);
#ifdef MALLOC_TRACE
#  include <MallocTraceOn.h>
#endif

    int strict;
    int lazy;
    FILE* trace_out;
    FILE* stats_out;
    char* statsfile;
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

    size_t nfillbin;
    size_t nmmap;
    size_t sizemmap;
    size_t nmunmap;
    size_t sizemunmap;

    size_t highwater_alloc;
    size_t highwater_mmap;

    size_t mysize;

  size_t pagesize;
  bool dieing;
};

void AllocError(const char*);

} // End namespace SCIRun


