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

! sparc/csw.s
!
! Low level Context switch code for SPARC's

	.text

! Suspend the current thread and resume the next one.
!
!	void
!	cthread_switch(cur, next, lock)
!		int *cur;
!		int *next;

	.globl	_cthread_switch
_cthread_switch:

        save    %sp, -(18*4), %sp              ! Give me a window

	ta	3                              ! Save current windows

        mov     %i2, %o0                       ! Save lock pointer
	st	%fp, [%sp + 16*4]              ! store frame pointer
	st      %i7, [%sp + 17*4]              ! store return address
        st      %sp, [%i0]                     ! store previous stack

        ld      [%i1], %o1                     ! load new stack ptr into %o1
	ld	[%o1+(8*4)],  %i0              ! restore first arg 
        ld      [%o1+(16*4)], %fp              ! Restore frame pointer
        ld      [%o1+(17*4)], %i7              ! Restore return address
	mov	%o1, %sp		       ! NOW update the stack ptr

        st      %g0, [%o0]                     ! Clear the lock

	ret
        restore



! Prepare the stack with a default return location.
!
!  void
!  cthread_prepare ( cthread_t child,
!		    stack_pointer_t *child_sp,
!		    stack_pointer_t stackbase )

        .globl _cthread_prepare
_cthread_prepare:
	andn   %o2, 7, %o2              ! round to an 8 byte alignment

        sub    %o2, 16*4, %o2           ! Create a stack frame
        st     %g0, [%o2+(14*4)]        ! and null FP
	st     %g0, [%o2+(15*4)]        ! dummy return address

        sub    %o2, 16*4, %o3           ! Create a second stack frame
        sethi  %hi(_cthread_body), %o4
        or     %lo(_cthread_body), %o4, %o4   ! Get pointer to function
        add    %o4, -8, %o4             ! Add offset
        st     %o0, [%o3+(8*4)]         ! Store self
        st     %o2, [%o3+(16*4)]        ! fp is earlier frame
        st     %o4, [%o3+(17*4)]        ! return address
        retl                            ! Return from leaf
        st     %o3, [%o1]               ! Store new stack pointer DELAY SLOT

