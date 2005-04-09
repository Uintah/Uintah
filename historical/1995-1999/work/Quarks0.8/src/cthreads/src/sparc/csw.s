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
!
!  Reference: 
!     Register Windows and User-Space Threads on the SPARC
!     David Keppel
!     University of Washington Department of Computer Science and Engineering
!     Techreport #91-08-01
!
#include <sun4/asm_linkage.h>
#include <sun4/trap.h>

	.text

! Suspend the current thread and resume the next one.
!
!	void
!	cthread_switch(cur, next, lock)
!		int *cur;
!		int *next;

	.globl	_cthread_switch
_cthread_switch:
	st	%fp,[%sp + (16*4)]		! Save frame pointer and
	st	%o7,[%sp + (17*4)]		!   return address
	st	%sp,[%o0]		  	! Save old stack pointer

					   	! Trap to SunOS to save
	ta	ST_FLUSH_WINDOWS               	!  the current windows
						!  on the old stack

	sub	%sp, WINDOWSIZE, %sp    	! Allocate a third kernel
						! window save area (kswa) on
						! the old stack.

						! This is necessary so an 
						!  interrupt wouldn't write the
						!  registers back to either the
						!  old or new kwsa while in the
						!  middle of the stack 
						!  transition

						! We can only safely switch to
						!  the new stack when all the
						!  registers are reloaded

	ld	[%o1],%o0			! %o0 points to the new stack
	ldd	[%o0 + (0*4)], %l0 		! Restore all the registers
	ldd	[%o0 + (2*4)], %l2 		!  double loads are used since
	ldd	[%o0 + (4*4)], %l4 		!  they are denser.
	ldd	[%o0 + (6*4)], %l6 
	ldd	[%o0 + (8*4)], %i0
	ldd	[%o0 + (10*4)], %i2 
	ldd	[%o0 + (12*4)], %i4 
	ldd	[%o0 + (14*4)], %i6
	ld	[%o0 + (16*4)], %fp
	ld	[%o0 + (17*4)], %o7
	mov	%o0, %sp			! NOW update the stack pointer
	retl			
        st      %g0, [%o2]                      ! Clear the lock (DELAY SLOT)


! Prepare the stack with a default return location.
!
!  void
!  cthread_prepare ( cthread_t child,
!		    stack_pointer_t *child_sp,
!		    stack_pointer_t stackbase )

        .globl _cthread_prepare
_cthread_prepare:




	andn   %o2, 7, %o2              ! round to an 8 byte alignment

        sub    %o2, WINDOWSIZE, %o2     ! Create a stack frame
        st     %g0, [%o2+(14*4)]        ! and null FP
	st     %g0, [%o2+(15*4)]        ! dummy return address

        sub    %o2, WINDOWSIZE, %o3           ! Create a second stack frame
        sethi  %hi(_cthread_body), %o4
        or     %lo(_cthread_body), %o4, %o4   ! Get pointer to function
        st     %o0, [%o3+(8*4)]         ! Store self
        st     %o2, [%o3+(16*4)]        ! fp is earlier frame
        st     %o4, [%o3+(17*4)]        ! return address
        retl                            ! Return from leaf
        st     %o3, [%o1]               ! Store new stack pointer DELAY SLOT


