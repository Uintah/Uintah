/*
 *  MIPS R2000/R3000 Cthread startup code
 */

#define PIC

#include <regdef.h>
#include <asm.h>

	.lcomm	__Argv	4
	.lcomm	__Argc	4
	.lcomm	__original_sp 4
	.comm	_environ 4
	.comm	errno	4
	.text
	.align	2

	.set noreorder
	.option pic2
/*
 * I must confess: I dissasembled the /usr/lib/crt1.o file as a starting
 * point.
 */

LEAF(start)
	SETUP_GPX(t0)
	.set noreorder
	lw	a0, 0(sp)
	addiu	a1,sp,4
	addiu	a2,a1,4
	sll	v0,a0,2
	sw	a0,__Argc
	sw	a1,__Argv
	addu	a2,a2,v0
	sw	a2,_environ
	subu	sp,24
	
	sw	gp,16(sp)
	sw	zero,20(sp)
	move	s8,zero
	jal	__istart
	nop
	jal	__readenv_sigfpe
	nop

	sw	sp,__original_sp
	jal	cthread_init
	nop
	beq	v0,zero,noway
	nop
	move	sp,v0           /* new stack */
	subu	sp,128
noway:
	lw	a0,__Argc
	lw	a1,__Argv
	lw	a2,_environ
	jal	main		/* Do the main thing... */
	nop
	lw	t0, __original_sp
	nop
	lw	gp, 16(t0)
	move	a0, v0		/* Get return value... */
	jal	cthread_exit
	nop
	break   0

END(start)

LEAF(moncontrol)
	j	ra
END(moncontrol)

/*
LEAF(_mcount)
	j	ra
END(_mcount)
*/
	
LEAF(_sprocmonstart)
	j	ra
END(_sprocmonstart)
