/*
 * Copyright (c) 1991,1992,1994 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the
 * Computer Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 */

#include "asm.h"

/*
	.space  $PRIVATE$
	.subspa	$GLOBAL$
	.export	$global$
$global$
*/

	.space	$TEXT$
	.subspa $UNWIND_START$,QUAD=0,ALIGN=8,ACCESS=0x2c,SORT=56
	.export $UNWIND_START
$UNWIND_START
	.subspa	$UNWIND_END$,QUAD=0,ALIGN=8,ACCESS=0x2c,SORT=72
	.export	$UNWIND_END
$UNWIND_END

	.subspa	$RECOVER_START$,QUAD=0,ALIGN=4,ACCESS=0x2c,SORT=73
	.export $RECOVER_START
$RECOVER_START
	.subspa	$RECOVER$MILLICODE$,QUAD=0,ALIGN=4,ACCESS=0x2c,SORT=78
	.subspa	$RECOVER$
	.subspa	$RECOVER_END$,QUAD=0,ALIGN=4,ACCESS=0x2c,SORT=88
	.export	$RECOVER_END
$RECOVER_END
	.subspa	$CODE$
	.subspa $FIRST$


/*
 * start(argc, argv, envp)
 * 	int argc;
 * 	char **argv;
 * 	char **envp;
 *
 * Initial execution of a task. We assume that our parent set us up with
 * a valid stack to run on.
 */
	.proc
	.callinfo SAVE_SP,FRAME=128
	.export	_start,entry
	.entry
_start
	/*
	 * Stash the args away in callee save registers.
	 */
	copy	arg0,r10
	copy	arg1,r11
	copy	arg2,r12
	copy	arg3,r13

	/*
	 * initialize the global data pointer dp
	 */
/*
	ldil    L%$global$,dp
	ldo     R%$global$(dp),dp
*/

	/*
	 * double word align the stack and allocate a stack frame to start out
	 * with.
	 */
/*
	ldo     128(sp),sp
	depi    0,31,3,sp
*/

	/*
	 * Call cthread_init() which allocates the first cthread
	 * for the task. It then allocates a stack, stores the self reference 
	 * pointer for us and returns the stack we should start with. This 
	 * stack is not aligned and we need to double word align it before 
	 * use. If the routine returns 0 for the stack then this is not
	 * the first task in the cthreads environment and we should stay
	 * on the current stack.
	 */
	.import cthread_init, code
	ldil	L%cthread_init,t1
	ldo	R%cthread_init(t1),t1

	.call
	blr     r0,rp
	bv,n    (t1)
	nop

	comb,=,n r0,ret0,no_cthread_stack

	/*
	 * we have a new stack to switch to, align it to a double word boundary
	 * and allocate a stack frame.
	 */
	ldo	7(ret0),ret0
	depi	0,31,3,ret0

	copy	ret0,sp
	ldo	48(sp),sp

no_cthread_stack

	/*
	 * Okay, get ready to call main (finally). First we have to initialize
	 * the "environ" pointer, set up argc/argv, and stash the magic
	 * PA-RISC flag (arg3).
	 */
	.import environ,data 

	copy	r10,arg0
	copy	r11,arg1
	copy	r12,arg2
	copy	r13,arg3

        addil   L%environ-$global$,dp
        stw     arg2,R%environ-$global$(r1)

	.import main
	ldil	L%main,t1
	ldo	R%main(t1),t1

	.call
	blr     r0,rp
	bv,n    (t1)
	nop

	/*
	 * Call cthread_exit() to do cleanup of any threads that are still
	 * executing. cthread_exit() will call exit().
	 */
	.import cthread_exit, code
	ldil	L%cthread_exit,t1
	ldo	R%cthread_exit(t1),t1

	copy	ret0,arg0
	.call
	blr     r0,rp
	bv,n    (t1)
	nop

	/*
	 * never returns
	 */
	.exit
	.procend

	.end
