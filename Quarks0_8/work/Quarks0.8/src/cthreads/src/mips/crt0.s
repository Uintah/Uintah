/*
 *  MIPS R2000/R3000 Cthread startup code
 */


#include <regdef.h>
#include <asm.h>

	.lcomm	__Argv	4
	.lcomm	__Argc	4
	.comm	_environ 4
	.comm	errno	4
	.text
	.align	2
/*
 * I must confess: I dissasembled the /usr/lib/crt1.o file as a starting
 * point.
 */
LEAF(start)
	la	gp,_gp
	lw	a0, 0(sp)
	addiu	a1,sp,4
	addiu	a2,a1,4
	sll	v0,a0,2
	sw	a1,__Argv
	addu	a2,a2,v0
	subu	sp,24
	sw	a2,_environ
	sw	zero,20(sp)
	move	s8,zero
	/* jal	_istart */
	sw	a0,__Argc
	jal	__readenv_sigfpe

	move	a0,zero
	jal	setchrclass

	jal	cthread_init
	beq	v0,zero,noway
	move	sp,v0           /* new stack */
	subu	sp,128
noway:
	lw	a0,__Argc
	lw	a1,__Argv
	lw	a2,_environ
	jal	main

	move	a0,v0
	jal	cthread_exit     /* better not return from this */
	break	0
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
