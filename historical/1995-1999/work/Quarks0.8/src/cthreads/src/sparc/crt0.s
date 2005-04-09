!
! Copyright (c) 1995 The University of Utah and
! the Computer Systems Laboratory (CSL).  All rights reserved.
!
! Permission to use, copy, modify and distribute this software is hereby
! granted provided that (1) source code retains these copyright, permission,
! and disclaimer notices, and (2) redistributions including binaries
! reproduce the notices in supporting documentation, and (3) all advertising
! materials mentioning features or use of this software display the following
! acknowledgement: ``This product includes software developed by the
! Computer Systems Laboratory at the University of Utah.''
!
! THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
! IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
! ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
!
! CSL requests users of this software to return to csl-dist@cs.utah.edu any
! improvements that they make and grant CSL redistribution rights.
!

	.data
	.global _environ
_environ:
	.long	0


	.text
	.globl start
start:
!
!   get arguments to main
!
		ld	[%sp + 64], %l0
		add	%sp, 68, %l1
		sll	%l0, 2,	%l2
		add	%l2, 4,	%l2
		add	%l1, %l2, %l2
		sethi	%hi(_environ), %l3
		st	%l2, [%l3+%lo(_environ)]
!
! initialize cthreads
!
		andn    %sp, 7, %sp 		! round stack ptr to 8 byte 
                call    _cthread_init, 0
		nop
                cmp     %o0, 0
                be	2f
                nop
!
!  Use stack returned from cthread_init
!
                mov     %o0, %sp
		andn    %sp, 7, %sp
		sub     %sp, 128, %sp       ! allow space for stackframe
2:
!
! call main
		mov	%l0,%o0
		mov	%l1,%o1
		mov	%l2,%o2
		call	_main, 3
                nop

!
! in case main returns
!
                call    _cthread_exit, 1
                nop
!
!  it is an error if cthread_exit returns
!
		call	_abort
                nop




