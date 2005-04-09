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
#include "asm.h"

	.space	$TEXT$
	.subspa $CODE$

/*
 * int
 * spin_try_lock(s)
 * 	spin_lock_t *s;
 * 
 * Try to acquire a lock, return 1 if successful, 0 if not.
 */


	.export	spin_try_lock,entry
	.proc
	.callinfo
spin_try_lock
	.enter

	/*
	 * align the lock to a 16 byte boundary
	 */
	ldo	15(arg0),arg0
	depi	0,31,4,arg0

	/*
	 * attempt to get the lock
	 */
	ldcws	(arg0),ret0

	/*
	 * if ret0 is 0 then we didn't get the lock and we return 0,
	 * else we own the lock and we return 1.
	 */
	subi,=	0,ret0,r0
	ldi	1,ret0

	.leave
	.procend


/*
 * void
 * spin_unlock(s)
 *	spin_lock_t *s;
 *
 * Release a lock.
 */
	.export	spin_unlock,entry
	.proc
	.callinfo
spin_unlock
	.enter

	/*
	 * align the lock to a 16 byte boundary
	 */
	ldo	15(arg0),arg0
	depi	0,31,4,arg0

	/*
	 * clear the lock by setting it non-zero
	 */
	ldi	-1,t1
	stw	t1,(arg0)

	.leave
	.procend
