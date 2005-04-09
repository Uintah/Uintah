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


/*
 * Cthreads internal data structures
 */
#ifndef CTHREAD_INTERNALS_H
#define CTHREAD_INTERNALS_H

#include "cthreads.h"
#include "context_switch.h"
#include "lock.h"

struct cthread_queue {
    spin_lock_t lock;
    cthread_t head;
    cthread_t tail;
    int	      len;
};
typedef struct cthread_queue *cthread_queue_t;

#define queue_empty(q) (!(q)->len)

struct cthread {
/*   
 * The next link for queing the thread.  Mutual exclusion of this field is 
 * controled through the appropriate queue lock.  Special care is required
 * to ensure the thread is never being inserted into two different queues
 * at the same time.
 */
    cthread_t t_next;	 

     /*
      *  These are used by thread_body to invoke the users function 
      */
    cthread_fn_t func;		/* the function this thread runs */
    any_t  arg;			/* its argument */

    /*
     *  This is were all the thread context is held so thread switching
     *  can occur.
     */
    stack_pointer_t t_sp;	    /* current thread stack pointer */
    stack_pointer_t t_stackbase;	/* base of stack for thread */

    /*
     *  This has stuff has to do with the return value of the thread
     *  and the joining with other threads.
     */
    spin_lock_t result_lock;	/* for mutual exclusion to remaining data  */

    int detached;		/* TRUE if thread is detached */
    int done;			/* TRUE if thread is done */
    any_t result;		/* return value */
    cthread_t joiner;    	/* thread attempting to join */
};

struct mutex { 
    int held;			/* TRUE when mutex is held */
    struct cthread_queue waiters;	/* queue of waiters */
};


struct condition {
    struct cthread_queue waiters;
};


#endif /* CTHREAD_INTERNALS_H */
