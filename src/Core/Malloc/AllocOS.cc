/*
 *  AllocOS.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Malloc/AllocOS.h>
#include <sys/mman.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#if defined(__sgi) || defined(__linux)
#include <unistd.h>
#endif
#include <sci_config.h>

#if defined(sun) || defined(__linux)
#define MMAP_TYPE char
#else
#define MMAP_TYPE void
#endif

namespace SCICore {
namespace Malloc {

static int devzero_fd=-1;


OSHunk* OSHunk::alloc(size_t size, bool returnable)
{
    size_t asize=size+sizeof(OSHunk);
    void* ptr;
    if(returnable){
       if(devzero_fd == -1){
	  devzero_fd=open("/dev/zero", O_RDWR);
	  if(devzero_fd == -1){
	     fprintf(stderr, "Error opening /dev/zero: errno=%d\n", errno);
	     abort();
	  }
       }
#ifdef SCI_64BIT
       ptr=mmap64(0, asize, PROT_READ|PROT_WRITE, MAP_PRIVATE,
		  devzero_fd, 0);
#else
       ptr=mmap(0, asize, PROT_READ|PROT_WRITE, MAP_PRIVATE,
		devzero_fd, 0);
#endif
    } else {
       ptr = sbrk((long)asize);
    }

    OSHunk* hunk=(OSHunk*)ptr;
    if((long)ptr == -1){
#ifdef SCI_64BIT
       fprintf(stderr, "Error allocating memory (%ld bytes requested)\nmmap: errno=%d\n", asize, errno);
#else
       fprintf(stderr, "Error allocating memory (%d bytes requested)\nmmap: errno=%d\n", asize, errno);
#endif
       abort();
    }
    hunk->data=(void*)(hunk+1);
    hunk->next=0;
    hunk->ninuse=0;
    hunk->len=size;
    hunk->returnable=returnable;
    return hunk;
}

void OSHunk::free(OSHunk* hunk)
{
   if(!hunk->returnable){
      fprintf(stderr, "Attempt to return a non-returnable memory hunk!\n");
      abort();
   }
    size_t len=hunk->len;
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
}

} // End namespace Malloc
} // End namespace SCICore

