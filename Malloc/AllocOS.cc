
#include <AllocOS.h>
#include <sys/mman.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

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
    if(munmap((void*)hunk, len) == -1){
	fprintf(stderr, "Error unmapping memory\nmunmap: errno=%d\n", errno);
	abort();
    }
}
