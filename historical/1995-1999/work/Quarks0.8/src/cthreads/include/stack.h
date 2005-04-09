/* 
 * Copyright (c) 1995 The University of Utah and
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

#ifndef STACK_H
#define STACK_H

#include "cthreads.h"

typedef void *stack_pointer_t;	/* the data type for a stored stack pointer */


/* Initialize the stack allocation package (to be called once) */
void stack_init(void);

/* Allocate one new stack */
void stack_alloc(cthread_t t, stack_pointer_t *stack_base);

/* Note there is no stack_free.  Once you allocate a stack to a
 * thread control block it should stay with the tcb.  You'll
 * probably keep a free list of tcbs (with associated stacks.)
 */

/* Stack.c also exports cthread_self(), defined in cthreads.h */

#endif  /* STACK_H */
