/* 
 * Copyright (c) 1990, 1995 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the Computer 
 * Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 * 	Utah $Hdr$
 */
/* 
 * File:	stack.c
 * Description: 
 * Author:	Leigh Stoller
 * 		Computer Science Dept.
 * 		University of Utah
 * Date:	15-Nov-90
 *
 */

#include "stack_internal.h"
#include <sys/types.h>
#include <sys/mman.h>

#include <stdio.h>

#ifndef HPUX
extern int getpagesize(void);
#else
#include <unistd.h>
static int getpagesize()
{
    return sysconf(_SC_PAGE_SIZE);
}
#endif

static int pagesize = 0;	/* Machine pagesize */

/*
 * Stack allocation/initialization code.
 */


/*
 * We preallocate some stacks to avoid wasting memory. The free list is
 * maintained by putting the "next" field at the base of the stack. Since
 * there is only 1 stack per cthread, there is no reason to put stacks
 * back on the free list; they will stay with the cthread.
 */
char *stack_free_list = 0;

#define stack_next(stack)	(*((int *)((stack) + STACKSIZE) - 1))

spin_lock_t stack_lock;


/*
 * Allocate a block of memory for stacks. Some OSs are picky about they use
 * mprotect, so we may need to mmap an anon region instead of using malloc.
 */
#ifdef HPUX
char *
getmem(int size)
{
	char 	*mem;
	
#ifdef HPUX
	int flags =  MAP_PRIVATE|MAP_ANONYMOUS|MAP_VARIABLE;
	int fd = -1;
#endif

	mem = (char *) mmap(NULL, size, PROT_READ|PROT_WRITE, flags,
			    fd, 0);

	if ((int) mem < 0) {
		perror("mmap");
		exit(1);
	}
	return(mem);
}
#else
char *
getmem(int size)
{
	return((char *) malloc(size));
}
#endif

/*
 * Lets preallocate some stacks. 
 */
void more_stacks(void)
{
	char *memblk = (char *) getmem((STACKSIZE * STACKCOUNT) + STACKSIZE);
	int  i;

	if (! memblk) cthread_panic("no memory for thread stacks\n" );

	/*
	 * Align base of the memory block, break it up into stacksized
	 * blocks, and generate the freelist.
	 */
	memblk = (char *) roundup((int) memblk, STACKSIZE);
	
	for (i = 0; i < STACKCOUNT; i++) {
		stack_next(memblk) = (int) stack_free_list;
		stack_free_list    = memblk;
		memblk	    	  += STACKSIZE;
	}
}


void stack_init(void)
{
    spin_lock_init(&stack_lock);
    pagesize = getpagesize();
    more_stacks();
}


/*
 * Protect the last page of the stack so that it cannot be written.
 */
static void 
setredzone(base)
	char *base;
{
  if (pagesize <= 0)
    {
      fprintf( stderr, "pagesize is bogus: %d  initialization error?\n", pagesize );
      abort();
    }
#ifndef IRIX4  
#ifdef STACK_GROWTH_UP
  if (mprotect(base + STACKSIZE - pagesize, pagesize, PROT_NONE) < 0)
#else
  if (mprotect(base, pagesize, PROT_READ) < 0)
#endif /* STACK_GROWTH_UP */
    {
      perror("mprotect error in setredzone");
      abort();
    }
#endif  
}

void stack_alloc(cthread_t t, stack_pointer_t *stack_base)
{
	char *base;

	spin_lock_yield(&stack_lock);
	if (! stack_free_list)   
	    more_stacks();
	base = stack_free_list;
	stack_free_list = (char *) stack_next(base);
	spin_unlock( &stack_lock );

	/*
	 * Store self pointer.
	 * Compute Top of Stack and store it into cthread structure.
	 */
#ifdef STACK_GROWTH_UP
	*((cthread_t *) base) = t;
	*stack_base = (void *) (base + sizeof(cthread_t *));
#else /* STACK_GROWTH_UP */
	*((cthread_t *) (base + STACKSIZE - sizeof(cthread_t *))) = t;
	*stack_base = (void *) (base + STACKSIZE - sizeof(cthread_t *));
#endif /* STACK_GROWTH_UP */

	setredzone(base);
}

/*
 * Self: Find the the pointer to the currently running cthread by looking
 * at the base of the stack.
 */
cthread_t
cthread_self()
{
	int x, y = (int) &x;
	cthread_t *addr;

#ifdef STACK_GROWTH_UP
	addr = (cthread_t *) (y & STACKMASK);
#else /* STACK_GROWTH_UP */
	addr = (cthread_t *)((y | STACKMASK) + 1)  -1;
#endif /* STACK_GROWTH_UP */
	return *addr;
}

