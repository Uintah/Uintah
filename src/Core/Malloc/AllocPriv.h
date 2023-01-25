/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <Core/Parallel/MasterLock.h>

#include <sci_defs/malloc_defs.h>

#include <cstdlib>

#include <stdio.h>

namespace Uintah {

struct OSHunk;

struct Sentinel {
  unsigned int first_word;
  unsigned int second_word;
};

struct AllocBin;


struct Tag {
  AllocBin   * bin;
  const char * tag;
#ifdef USE_TAG_LINENUM
  int          linenum;
#endif
  Tag        * next;
  Tag        * prev;
  OSHunk     * hunk;
  size_t       reqsize;
};


struct AllocBin {
  Tag    * free;
  Tag    * inuse;
  size_t   maxsize;
  size_t   minsize;
  int      ninuse;
  int      ntotal;
  size_t   nalloc;
  size_t   nfree;
};


struct Allocator {

  void lock();

  void unlock();

  void* alloc_big( size_t size, const char* tag, int linenum );

  void* memalign( size_t alignment, size_t size, const char* tag );

  void* alloc( size_t size, const char* tag, int linenum );

  void free( void* );

  void* realloc( void* p, size_t size );

  inline AllocBin* get_bin( size_t size );

  void fill_bin( AllocBin* );

  void get_hunk( size_t, OSHunk*&, void*& );

  void init_bin( AllocBin*, size_t maxsize, size_t minsize );

  void audit( Tag*, int );

  size_t obj_maxsize( Tag* );


  int      strict;
  int      lazy;
  FILE   * stats_out;
  char   * stats_out_filename;
  OSHunk * hunks;

  AllocBin * small_bins;
  AllocBin * medium_bins;
  AllocBin   big_bin;


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

  bool dying;

  Uintah::MasterLock m_lock{};

};

void AllocError(const char*);

} // End namespace Uintah


