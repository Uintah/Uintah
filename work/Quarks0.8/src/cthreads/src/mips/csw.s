/*
 *  MIPS R2000/R3000 Context switch code for coroutines
 */


#include <regdef.h>
#include <asm.h>

/* Offsets on stack for various saved registers */

#define S0 0
#define S1 4
#define S2 8
#define S3 12
#define S4 16
#define S5 20
#define GP 24	
#define PC 28
#define S6 32
#define S7 36
#define A0 40
#define FP 44
#define F20 48
#define F21 52
#define F22 56
#define F23 60
#define F24 64
#define F25 68
#define F26 72
#define F27 76
#define F28 80
#define F29 84
#define F30 88
#define F31 92
#define MACHINE_STATE_SIZE (F31+4)


	.globl cthread_body

	.text
	.align	2
/*
 * Cthread switch
 *
 *  a0  -- address of location to save old stack pointer
 *  a1  -- address of new stack pointer
 *  a2  -- address of lock to clear
 */
LEAF(cthread_switch)
	/* Push machine state and save the stack pointer*/
	subu	sp, MACHINE_STATE_SIZE
	sw	s0, S0(sp)
	sw	s1, S1(sp)
	sw	s2, S2(sp)
	sw	s3, S3(sp)
	sw	s4, S4(sp)
	sw	s5, S5(sp)
	sw	s6, S6(sp)
	sw	s7, S7(sp)
	sw	a0, A0(sp)
	sw	fp, FP(sp) 
	sw	gp, GP(sp) 
	sw	ra, PC(sp)
	swc1	$f20, F20(sp)
	swc1	$f21, F21(sp)
	swc1	$f22, F22(sp)
	swc1	$f23, F23(sp)
	swc1	$f24, F24(sp)
	swc1	$f25, F25(sp)
	swc1	$f26, F26(sp)
	swc1	$f27, F27(sp)
	swc1	$f28, F28(sp)
	swc1	$f29, F29(sp)
	swc1	$f30, F30(sp)
	swc1	$f31, F31(sp)
	sw	sp, 0(a0)


	/* Restore stack pointer, clear lock,  and pop state */
	lw	sp, 0(a1)
	sw	zero, 0(a2)
	lw	s0, S0(sp)
	lw	s1, S1(sp)
	lw	s2, S2(sp)
	lw	s3, S3(sp)
	lw	s4, S4(sp)
	lw	s5, S5(sp)
	lw	s6, S6(sp)
	lw	s7, S7(sp)
	lw	a0, A0(sp)
	lw	fp, FP(sp) 
	lw	gp, GP(sp) 
	lw	ra, PC(sp)
	lwc1	$f20, F20(sp)
	lwc1	$f21, F21(sp)
	lwc1	$f22, F22(sp)
	lwc1	$f23, F23(sp)
	lwc1	$f24, F24(sp)
	lwc1	$f25, F25(sp)
	lwc1	$f26, F26(sp)
	lwc1	$f27, F27(sp)
	lwc1	$f28, F28(sp)
	lwc1	$f29, F29(sp)
	lwc1	$f30, F30(sp)
	lwc1	$f31, F31(sp)
	addu	sp, MACHINE_STATE_SIZE

	j	ra
END(cthread_switch)

/*
 *  a0   -  cthread_t (passed as arg to cthread_body)
 *  a1   -  location to store new child stackpointer
 *  a2   -  stack base of child
 */
LEAF(cthread_prepare)
	/* Push machine state and save the stack pointer*/
	subu	a2, MACHINE_STATE_SIZE  
	subu	a2, MACHINE_STATE_SIZE
	sw	a0, A0(a2)
	sw	gp, GP(a2) 
	sw	fp, FP(a2) 
	la	t0, cthread_body+16
	sw	t0, PC(a2)
	sw	a2, 0(a1)
	j	ra
END(cthread_prepare)




