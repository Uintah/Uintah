//static char *id="@(#) $Id$";

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
#ifdef __sgi
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

//
// $Log$
// Revision 1.6  2000/07/27 07:41:48  sparker
// Distinguish between "returnable" chunks and non-returnable chucks of memory
// Make malloc get along with SGI's MPI
//
// Revision 1.5  1999/08/23 06:30:37  sparker
// Linux port
// Added X11 configuration options
// Removed many warnings
//
// Revision 1.4  1999/08/19 05:30:56  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
// Revision 1.3  1999/08/18 18:56:17  sparker
// Use 64 bit mmap
// Incorporated missing Hashtable Pio function
// Fix bug in MAKE_PARALLELISM handling
// Got rid of lib32 in main/Makefile.in (for 64 bit)
//
// Revision 1.2  1999/08/17 06:39:30  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:58  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//
