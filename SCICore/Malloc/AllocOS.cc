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

#ifdef sun
#define MMAP_TYPE char
#else
#define MMAP_TYPE void
#endif

namespace SCICore {
namespace Malloc {

static int devzero_fd=-1;


OSHunk* OSHunk::alloc(size_t size)
{
    size_t asize=size+sizeof(OSHunk);
    if(devzero_fd == -1){
	devzero_fd=open("/dev/zero", O_RDWR);
	if(devzero_fd == -1){
	    fprintf(stderr, "Error opening /dev/zero: errno=%d\n", errno);
	    abort();
	}
    }
    void* ptr=mmap(0, asize, PROT_READ|PROT_WRITE, MAP_PRIVATE, devzero_fd, 0);
    if((long)ptr == -1){
	fprintf(stderr, "Error allocating memory (%d bytes requested)\nmmap: errno=%d\n", asize, errno);
	abort();
    }
    OSHunk* hunk=(OSHunk*)ptr;
    hunk->data=(void*)(hunk+1);
    hunk->next=0;
    hunk->ninuse=0;
    hunk->len=size;
    return hunk;
}

void OSHunk::free(OSHunk* hunk)
{
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
