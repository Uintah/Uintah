/* 
 * Copyright (c) 1995 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the Computer 
 * Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 * 	Utah $Hdr$
 */

#ifndef HPUX
/*
 * HISTORY
 * $Log$
 * Revision 1.1  1995/05/28 07:04:28  sparker
 * Imported Quarks
 *
 * Revision 1.1  90/11/17  16:58:24  stoller
 * Initial revision
 * 
 * Revision 2.2  90/11/05  14:38:00  rpd
 *      Created.
 *      [90/11/01            rwd]
 * 
 */

#ifndef _MACHINE_CTHREADS_H_
#define _MACHINE_CTHREADS_H_

/* On the 68K, a spinlock is just an int.  We use the tas instruction  */
typedef int spin_lock_t;

/* 
 *  All these operations receive a pointer to the spinlock 
 */

#define spin_lock_init(s) *(s)=0
#define spin_lock_locked(s) (*(s) != 0)

/* Operations implemented as functions */
int spin_try_lock( spin_lock_t *lock );

/* Operations Implemented as macros */
#define spin_unlock(the_lock)  *(the_lock)=0
#define spin_lock(the_lock) { while(! spin_try_lock(the_lock)) ; }

/* This one is like spin_lock but yields the processor between trys */
#define spin_lock_yield(the_lock) { \
  while(! spin_try_lock(the_lock)) \
      cthread_yield(); \
}



#endif /* _MACHINE_CTHREADS_H_ */

/*
#else
*/
#endif  

#if defined(hppa)

#ifndef _LOCK_
#define _LOCK_

typedef struct spin_lock_t {
	int _ldcws[4];
} spin_lock_t;

/*
 * we'll always assign a block of 16 bytes to the spinlock and then assume
 * that the real lock is the one aligned on a 16 byte boundary
 */
#define	_align_spin_lock(s) \
  ((int *)(((int) (s) + sizeof(spin_lock_t) - 1) & ~(sizeof(spin_lock_t) - 1)))

#define spin_lock_init(s) (*_align_spin_lock(s) = -1) 
#define spin_lock_locked(s) (*_align_spin_lock(s) == 0)

     
/* Operations implemented as functions */
extern int spin_try_lock( spin_lock_t *lock );
extern void spin_unlock( spin_lock_t *lock );

     /* Operations Implemented as macros */
#define spin_lock(the_lock) { while(! spin_try_lock(the_lock)) ; }

/* This one is like spin_lock but yields the processor between trys */
#define spin_lock_yield(the_lock) { \
  while(! spin_try_lock(the_lock)) \
      cthread_yield(); \
}

#endif /* _LOCK_ */
#endif 
