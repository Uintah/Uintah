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

#ifndef CONTEXT_SWITCH_H
#define CONTEXT_SWITCH_H

#include "lock.h"
#include "stack.h"


void cthread_switch (stack_pointer_t *cur_sp,  /* location to store old sp */
		     stack_pointer_t *new_sp,  /* SP to swap to */
		     spin_lock_t *lock	   /* spinlock to release atomically */
		     );

void cthread_prepare(cthread_t child,	      /* the thread we are preparing */
		     stack_pointer_t *child_sp, /* output: its stack pointer */
		     stack_pointer_t stackbase  /* input: base of stack      */
		    );

#endif /* CONTEXT_SWITCH_H */
