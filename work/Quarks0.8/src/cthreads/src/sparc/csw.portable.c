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
 * C implementation of context switch code relying on setjmp/longjmp
 */
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <assert.h>

#include "cthreads.h"
#include "stack.h"
#include "lock.h"


static unsigned int saved_oldSP;
static unsigned int saved_mySP;


/* Imports */
  /* We use _setjmp and _longjmp because longjmp is expensive and
   * (at least on MIPS) longjmp insists we jump backward on the stack. 
   */
extern int _setjmp(jmp_buf);
extern void _longjmp(jmp_buf,int val);
extern void cthread_body(cthread_t);

/* ------------------------------------------------------------------------ */
/*
 * Machine dependent defines
 */

#ifdef hp300
#define SP 2			/* Stack pointer offset in jmpbuf */
#endif

#ifdef SPARC
#define SP 2			/* Stack pointer offset in jmpbuf */
#endif

#if defined(__hppa) && defined(__hp9000s800)
#define SP 1
#endif

/* ------------------------------------------------------------------------ */

/* 
 * Forward declarations:
 */
void thread_shell(int pass, unsigned int *oldSP );

/*
 * The context block for a thread.
 */
typedef 
struct context {
    jmp_buf jmpbuf;		/* state */
} *context_t;


/* 
 * The initialization code sets up intermediate_stack.  It is the
 * initial stack contents for a new thread.
 */
static unsigned int intermediate_stack[1024];
static jmp_buf base_jmpbuf;
static int growth;


static spin_lock_t *the_lock;	/* the lock to unlock when the new thread gets
                                   control 
                                 */


/*
Assumptions:
    After the longjmp, only the SP and FP registers contain live values 
        which point to the stack.  In particular, this routine's return 
        PC is not used.
    The compiler keeps its live variables (namely stack, proc, and 
        closure) in registers or in the copied stack frame.
Pros and cons of this approach:
    + no assembler needed
    + reasonably portable
    + avoids depending on the compiler to keep variables in registers
    + allows the check for return from proc
    - does not place proc at the root of the stack
*/

void
cthread_switch( stack_pointer_t *old, stack_pointer_t *new, spin_lock_t *lock )
{
    register context_t *old_context = (context_t *) old;
    register context_t *new_context = (context_t *) new;

    if (*old_context == 0)
	*old_context = (context_t) malloc(sizeof(struct context));

    if (_setjmp((*old_context)->jmpbuf) == 0)
    {
	/* Saving old context */
	the_lock = lock;	/* pass lock on to new guy */
	_longjmp((*new_context)->jmpbuf,2);
	cthread_panic("you can't get here");
    } else {
	/* Came from the longjmp */
   	    /* get lock from the guy who switched us in */
	if (the_lock) spin_unlock(the_lock);
    }
}


/*
 * Called from cthread_init to initialize top level context and
 * to initialize intermediate_stack.  The arugment is the original
 * stackpointer passed in from CRT0.
 */
void 
set_top_context(unsigned int *oldSP)
{
    register int pass;

    pass = _setjmp(base_jmpbuf);
    thread_shell(pass,oldSP);
}

void
thread_shell(int pass, unsigned int * oldSP)
{
    switch(pass)
    {
    case 0:
	/* On the SPARC we need to force the register windows back on
         * the stack before saving the stack contents.  A longjmp
         * takes care of that so we do one.  It's benign on other
         * architectures.
         */
	_longjmp(base_jmpbuf,1);
    case 1:  {

	/* first time through, save contents of stack in the 
         * intermediate stack for cthread_prepare to use.
         */
	register unsigned int *mySP = *((unsigned int **)base_jmpbuf + SP);
	register unsigned int *newSP;

	/* Compare oldSP and mySP to learn stack my frame size and growth 
	   direction. Copy the frame (plus pad) to intermediate_stack and
	   remember where that should go in the stacks */

	if (mySP == oldSP)  /* stack did not grow ! */
	    cthread_panic( "stack did not grow");

	else if (mySP < oldSP) { /* stack grows downward */
	    newSP = intermediate_stack;
	    saved_oldSP = (unsigned int) oldSP;
	    saved_mySP  = (unsigned int) mySP;
	    for (growth = 0; mySP <= oldSP; *newSP++ = *oldSP--, growth--);
	}
	else { /* stack grows upward */
	    newSP = intermediate_stack;
	    saved_oldSP = (unsigned int) oldSP;
	    saved_mySP  = (unsigned int) mySP;
	    for (growth = 0; mySP >= oldSP; *newSP++ = *oldSP++, growth++); 
	}
	return; 
      }
    default: {
	/* This is where new threads start their life */
	register cthread_t self = cthread_self();

	if (the_lock) spin_unlock(the_lock);
	cthread_body( self );
	cthread_panic( "cthread_body isn't suppose to return" );
      }
    }
}

#define NELTS(a) (sizeof(a) / sizeof(unsigned int *))

void 
cthread_prepare( cthread_t self, stack_pointer_t *sp, stack_pointer_t stack)
{
    context_t *c = (context_t *) sp;
    int dsp;
    unsigned int *src; unsigned int *dst;
    int i;

    if (*c == NULL) {
	*c = (context_t) malloc( sizeof(struct context) );
	if (! *c) cthread_panic( "no memory for thread context block\n" );
    }

    /*
     * Copy the state we prepared in determine_context into
     * the new threads stack and setup the important registers
     */
    memcpy ( (*c)->jmpbuf, base_jmpbuf, sizeof (jmp_buf));
    if (growth < 0)    { 
	src = intermediate_stack;
	dst = (unsigned int *) stack - 1;
	for (i = 0; i >= growth; i--)
	    *dst-- = *src++;
	dst += 2;
	dsp = (int) dst - (int) saved_mySP;

	dst = (unsigned int *) stack - 1;
	for(i=0; i>= growth; i--, dst--) /* patch up stack address on stack */
	    if (*dst >= saved_mySP && *dst <= saved_oldSP)
		*dst += dsp;
	dst = (unsigned int *) &((*c)->jmpbuf[0]);
	for(i=0; i<NELTS((*c)->jmpbuf); i++, dst++)
	    if (*dst >= saved_mySP && *dst <= saved_oldSP)
		*dst += dsp;
    }
    else {
	src =  intermediate_stack;
	dst = (unsigned int *) stack + 1;
	for (i =0; i <= growth; i++) 
	    *dst++ = *src++; 
	dsp = (int) dst - (int) ((unsigned int *)base_jmpbuf)[SP]; 
	dst = (unsigned int *) stack + 1;
	for (i =0; i <= growth; i++, dst++) 
	    if (*dst <= saved_mySP && *dst >= saved_oldSP)
		*dst += dsp;
	dst = (unsigned int *) &((*c)->jmpbuf[0]);
	for(i=0; i<NELTS((*c)->jmpbuf); i++, dst++)
	    if (*dst <= saved_mySP && *dst >= saved_oldSP)
		*dst += dsp;
    }
}



