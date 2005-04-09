/*
 * Copyright (c) 1990,1991,1994 The University of Utah and
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
/*
 * Context switch and cproc startup for hp800
 */

#include "asm.h"
#include "assym.s"

#define FR_SIZE 256	/* XXX */

	.space	$TEXT$
	.subspa	$CODE$

/*
 * void
 * cthread_switch(cur, next, lock)
 *	int *cur;
 *	int *next;
 * 	spin_lock_t *lock;
 *
 * Take the current thread of execution and save its context on its stack.
 * Place the pointer to the context into *cur. Then take the context pointed
 * to by *next and load it. Finally clear the lock.
 */

	.export	cthread_switch,entry
	.proc
	.callinfo
cthread_switch
	.enter

	ldo	FR_SIZE(sp),sp

	/*
	 * save all the callee save registers for the current thread. We
	 * will also save rp so we know how to get back. We will save and
	 * restore arg0 since we need to pass some state in this when we
	 * bootstrap initial threads.
	 */
	stw	r2,-FR_SIZE+CSW_R2(sp)
	stw	r3,-FR_SIZE+CSW_R3(sp)
	stw	r4,-FR_SIZE+CSW_R4(sp)
	stw	r5,-FR_SIZE+CSW_R5(sp)
	stw	r6,-FR_SIZE+CSW_R6(sp)
	stw	r7,-FR_SIZE+CSW_R7(sp)
	stw	r8,-FR_SIZE+CSW_R8(sp)
	stw	r9,-FR_SIZE+CSW_R9(sp)
	stw	r10,-FR_SIZE+CSW_R10(sp)
	stw	r11,-FR_SIZE+CSW_R11(sp)
	stw	r12,-FR_SIZE+CSW_R12(sp)
	stw	r13,-FR_SIZE+CSW_R13(sp)
	stw	r14,-FR_SIZE+CSW_R14(sp)
	stw	r15,-FR_SIZE+CSW_R15(sp)
	stw	r16,-FR_SIZE+CSW_R16(sp)
	stw	r17,-FR_SIZE+CSW_R17(sp)
	stw	r18,-FR_SIZE+CSW_R18(sp)
	stw	arg0,-FR_SIZE+CSW_R26(sp)

	mfsp	sr3,t1
	stw	t1,-FR_SIZE+CSW_SR3(sp)
	mfctl	cr11,t1
	stw	t1,-FR_SIZE+CSW_CR11(sp)

	/*
	 * Save Necessary FPU State
	 * XXX - assuming that this is a real floating point unit
	 */

	/*
	 * Real fpu so first disable any pending traps by saving fr0
	 * and then save the exception registers and floating point registers
	 */
	ldo	-FR_SIZE+CSW_FR0(sp),t1
	fstds,ma fr0,8(t1)
	fstds,ma fr1,8(t1)
	fstds,ma fr2,8(t1)
	fstds,ma fr3,8(t1)
	fstds,ma fr12,8(t1)
	fstds,ma fr13,8(t1)
	fstds,ma fr14,8(t1)
	fstds	 fr15,(t1)

	/*
	 * save the context pointer, we'll just save the bottom of the context
	 * since our stacks grow up this will be much easier.
	 */
	stw	sp,(arg0)


	/*
	 * now load up the new context
	 */
	ldw	(arg1),sp

	ldw	-FR_SIZE+CSW_R2(sp),r2
	ldw	-FR_SIZE+CSW_R3(sp),r3
	ldw	-FR_SIZE+CSW_R4(sp),r4
	ldw	-FR_SIZE+CSW_R5(sp),r5
	ldw	-FR_SIZE+CSW_R6(sp),r6
	ldw	-FR_SIZE+CSW_R7(sp),r7
	ldw	-FR_SIZE+CSW_R8(sp),r8
	ldw	-FR_SIZE+CSW_R9(sp),r9
	ldw	-FR_SIZE+CSW_R10(sp),r10
	ldw	-FR_SIZE+CSW_R11(sp),r11
	ldw	-FR_SIZE+CSW_R12(sp),r12
	ldw	-FR_SIZE+CSW_R13(sp),r13
	ldw	-FR_SIZE+CSW_R14(sp),r14
	ldw	-FR_SIZE+CSW_R15(sp),r15
	ldw	-FR_SIZE+CSW_R16(sp),r16
	ldw	-FR_SIZE+CSW_R17(sp),r17
	ldw	-FR_SIZE+CSW_R18(sp),r18
	ldw	-FR_SIZE+CSW_R26(sp),arg0

	ldw	-FR_SIZE+CSW_SR3(sp),t1
	mtsp	t1,sr3
	ldw	-FR_SIZE+CSW_CR11(sp),t1
	mtctl	t1,cr11

	/*
	 * Restore FPU State
	 * XXX - assuming that this is a real floating point unit
	 */

	/*
	 * Real fpu so restore status register last to avoid a potential
	 * trap.  At this interface we only need to restore the entry
	 * save floating point registers.
	 */

	ldo	-FR_SIZE+CSW_FR15(sp),t1
	fldds,ma -8(t1),fr15
	fldds,ma -8(t1),fr14
	fldds,ma -8(t1),fr13
	fldds,ma -8(t1),fr12
	fldds,ma -8(t1),fr3
	fldds,ma -8(t1),fr2
	fldds,ma -8(t1),fr1
	fldds    (t1),fr0

	ldo	-FR_SIZE(sp),sp

	/*
	 * finally clear the lock pointed to by arg2
	 */

	/*
	 * align the lock to a 16 byte boundary
	 */
	ldo	15(arg2),arg2
	depi	0,31,4,arg2

	/*
	 * clear the lock by setting it non-zero
	 */
	ldi	-1,t1
	stw	t1,(arg2)

	.leave
	.procend


/*
 * void
 * cthread_prepare(child, child_context, stack)
 * 	int child;
 *	int *child_context;
 *	int stack;
 *
 * This routine is called to start a child process and fake up the context
 * so that it looks like it was running once. We need to fake the context
 * switch so that it "returns" from cproc_switch() to cthread_body().
 */

	.export	cthread_prepare,entry
	.proc
	.callinfo
cthread_prepare
	.enter

	/*
	 * align the stack onto a quad word
	 */
	ldo	15(arg2),arg2
	depi	0,31,4,arg2

	/*
	 * now create a stack frame so that cthread_body thinks everything
	 * is ok. Then put some room on for the save state information.
	 */
	ldo	64(arg2),arg2
	ldo	FR_SIZE(arg2),arg2

	/*
	 * basically we will just be saving 0 into most registers except for
	 * rp and arg0 since we need to pass the child argument to 
	 * cthread_body().
	 */
	.import	cthread_body
	ldil	L%cthread_body,t1
	ldo	R%cthread_body(t1),t1

	stw	t1,-FR_SIZE+CSW_R2(arg2)
	stw	r0,-FR_SIZE+CSW_R3(arg2)
	stw	r0,-FR_SIZE+CSW_R4(arg2)
	stw	r0,-FR_SIZE+CSW_R5(arg2)
	stw	r0,-FR_SIZE+CSW_R6(arg2)
	stw	r0,-FR_SIZE+CSW_R7(arg2)
	stw	r0,-FR_SIZE+CSW_R8(arg2)
	stw	r0,-FR_SIZE+CSW_R9(arg2)
	stw	r0,-FR_SIZE+CSW_R10(arg2)
	stw	r0,-FR_SIZE+CSW_R11(arg2)
	stw	r0,-FR_SIZE+CSW_R12(arg2)
	stw	r0,-FR_SIZE+CSW_R13(arg2)
	stw	r0,-FR_SIZE+CSW_R14(arg2)
	stw	r0,-FR_SIZE+CSW_R15(arg2)
	stw	r0,-FR_SIZE+CSW_R16(arg2)
	stw	r0,-FR_SIZE+CSW_R17(arg2)
	stw	r0,-FR_SIZE+CSW_R18(arg2)

	/*
	 * arg0 is already conveniently what we need it to be
	 */
	stw	arg0,-FR_SIZE+CSW_R26(arg2)

	/*
	 * the new thread should not assume the contents of sr3 or cr11
	 */
	stw	r0,-FR_SIZE+CSW_SR3(arg2)
	stw	r0,-FR_SIZE+CSW_CR11(arg2)

	/*
	 * We don't want the new process to inherit any floating point faults
	 * from the parent. Load all the floating point status registers
	 * with 0. We should also clear all the floatint point registers
	 * so that we don't magically end up with NaN in one of them when 
	 * we context switch to the thread.
	 */
	ldo	-FR_SIZE+CSW_FR0(arg2),t1
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,4(t1)
	stwm	r0,(t1)

	/*
	 * copy the context pointer into child_context
	 */
	stw	arg2,(arg1)

	.leave
	.procend

/*
 * int
 * get_dp()
 *
 * Return the contents of the dp register.
 */

	.export	get_dp,entry
	.proc
	.callinfo
get_dp
	.enter
	copy	dp,ret0
	.leave
	.procend
