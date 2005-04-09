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

! sparc/lock.s
!
! Mutex implementation for sparc (based on UofU hp300 implementation).

! int
! spin_try_lock(m)
!   int m;		(= int *m for our purposes)
!
	.align 4
	.globl	_spin_try_lock
_spin_try_lock:

        ldstub [%o0+%g0], %o1  ! according to pg. 138 of the sparc arch. manual
                           ! this zero extends to 32 bits.
        cmp %o1, 0

        be __yes
        nop                ! Delay slot

	jmpl %o7+8, %g0    ! Return to the caller
        or %g0, %g0, %o0   ! Delay slot

__yes:
        jmpl %o7+8, %g0
        or %g0, 1, %o0     ! Delay slot

! void
! spin_unlock(m)
!   int m;		(= int *m for our purposes)
!
	.align 4
	.global	_spin_unlock
_spin_unlock:
	jmpl %o7+8, %g0          ! Return to the caller
	st %g0, [%o0]            ! The actual zero goes into delay slot.


