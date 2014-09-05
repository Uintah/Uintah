/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
 *  Copyright (C) 199? SCI Group
 */

#include <stdlib.h>
#include <stdio.h>

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

} // End namespace SCIRun


