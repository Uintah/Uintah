/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
 *  AllocOS.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 */

#include <sci_defs/bits_defs.h>
#include <sci_defs/malloc_defs.h>

#include <Core/Malloc/AllocOS.h>
#include <Core/Malloc/AllocPriv.h>

#ifdef __APPLE__
#  include <sys/types.h>
#endif

#ifndef _WIN32
#  include <sys/mman.h>
#  include <unistd.h>
#endif

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fcntl.h>

#define ALIGN 16

#if defined(sun) || defined(__linux)
#  define MMAP_TYPE char
#else
#  define MMAP_TYPE void
#endif

namespace SCIRun {

#ifndef DISABLE_SCI_MALLOC
  static int devzero_fd=-1;
#endif

OSHunk* OSHunk::alloc(size_t size, bool returnable, Allocator* allocator)
{
#ifndef DISABLE_SCI_MALLOC
    unsigned long offset = sizeof(OSHunk)%ALIGN;
    if(offset != 0)
      offset = ALIGN-offset;
    size_t asize=size+sizeof(OSHunk)+offset;
    void* ptr;
    if(returnable){
       if(devzero_fd == -1){
	  devzero_fd=open("/dev/zero", O_RDWR);
	  if(devzero_fd == -1){
	     fprintf(stderr, "Error opening /dev/zero: errno=%d\n", errno);
	     abort();
	  }
       }
#ifdef SCI_64BITS
#  ifdef __sgi
       ptr=mmap64(0, asize, PROT_READ|PROT_WRITE, MAP_PRIVATE,
		  devzero_fd, 0);
#  else
       ptr=mmap(0, asize, PROT_READ|PROT_WRITE, MAP_PRIVATE,
		devzero_fd, 0);
#  endif
#else
       ptr=mmap(0, asize, PROT_READ|PROT_WRITE, MAP_PRIVATE,
		devzero_fd, 0);
#endif
    } else {
      void* align = sbrk(0);
      unsigned long offset = reinterpret_cast<unsigned long>(align)%ALIGN;
      if(offset){
	sbrk((long)(ALIGN-offset));
      }
      ptr = sbrk((long)asize);
    }

    OSHunk* hunk=(OSHunk*)ptr;
    if((long)ptr == -1){
#ifdef SCI_64BITS
       fprintf(stderr, "Error allocating memory (%lu bytes requested)\nmmap: errno=%d\n", asize, errno);
#else
       fprintf(stderr, "Error allocating memory (%u bytes requested)\nmmap: errno=%d\n", asize, errno);
#endif
       
       if(allocator){
#ifdef SCI_64BITS
	 fprintf(stderr, "Allocator was using %lu bytes.\n", allocator->sizealloc );
#else
	 fprintf(stderr, "Allocator was using %u bytes.\n", allocator->sizealloc );
#endif
	 // If the allocator is already dieing, we will just quit to try
	 // to avoid going into an infinite loop
	 if(allocator->dieing)
	   exit(1);

	 // Mark the allocator as dieing and unlock it so that allocations
	 // might succeed as we are shutting down
	 allocator->dieing = true;
	 allocator->noninline_unlock();
       }
       abort();
    }
    hunk->data=(void*)(hunk+1);
    if(offset){
      // Ensure alignment
      hunk->data = (void*)((char*)hunk->data+offset);
    }
    hunk->next=0;
    hunk->ninuse=0;
    hunk->len=size-offset;
    hunk->alloc_len=asize;
    hunk->returnable=returnable;
    return hunk;
#else
    return NULL;
#endif // DISABLE_SCI_MALLOC
}

#ifdef MALLOC_TRACE
#  include <MallocTraceOff.h>
#endif

void
OSHunk::free(OSHunk* hunk)
{
#ifndef DISABLE_SCI_MALLOC
   if(!hunk->returnable){
      fprintf(stderr, "Attempt to return a non-returnable memory hunk!\n");
      abort();
   }
    size_t len=hunk->alloc_len;

    if(munmap((MMAP_TYPE*)hunk, len) == -1){
	int i;
        for(i=0;i<10;i++){
#ifdef __sgi
	    sginap(10);
#endif
    	    if(munmap((MMAP_TYPE*)hunk, len) != -1)
		break;
        }
 	if(i==10){
	    fprintf(stderr, "Error unmapping memory\nmunmap: errno=%d\n", errno);
	    fprintf(stderr, "Unmap failed - leaking memory\n");
	    //abort();
	}
    }
#endif // DISABLE_SCI_MALLOC
}

#ifdef MALLOC_TRACE
#  include <MallocTraceOn.h>
#endif

} // End namespace SCIRun
