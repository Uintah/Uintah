/*              Quarks distributed shared memory system.
 * 
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
 *	Author: Dilip Khandekar, University of Utah CSL
 */
/**************************************************************************
 *
 * util.h: utilities
 *
 *************************************************************************/

#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdio.h>

#include "lock.h"       /* cthreads */
#include "cthreads.h"   /* cthreads */

/* Quarks header file */
#include "types.h"

/* interface with cthreads */
typedef spin_lock_t  Spinlock;
typedef any_t        Any;
typedef cthread_fn_t Threadfn;

extern Id Qknodeid;

#define ILOOP_MAX  1000

#ifndef IRIX
#ifdef CHECKASSERT
#define ASSERT(cond) \
    if ( !(cond) ) { \
       fprintf(stderr, "Assertion failed at line %d in file %s.\n", \
               __LINE__, __FILE__); \
       abort(); \
       }
#else
#define ASSERT(cond)
#endif
#define PANIC(message) \
    { \
	fprintf(stderr, "***PANIC (proc %d): %s\n", Qknodeid, (message)); \
	fprintf(stderr, "Line %d, File %s.\n", __LINE__, __FILE__); \
        graceful_exit(); \
	}

#define SETBIT(w, b)       ((w) |= (1 << (b)))
#define RESETBIT(w, b)     ((w) &= ~(1 << (b)))
#define TESTBIT(w, b)      ((w) & (1 << (b)))
#define SETLBITS(b)        ((1 << (b)) - 1)
#define min(a, b)          (((a) < (b)) ? (a) : (b))
#else   /* IRIX */

extern void ASSERT_I(int, int, char *);
#define ASSERT(cond)  ASSERT_I((int) cond, __LINE__, __FILE__)
extern void PANIC_I(char *, int, char *);
#define PANIC(message) PANIC_I(message, __LINE__, __FILE__)

#define SETBIT(w, b)       w|=(1<<b)
#define RESETBIT(w, b)     w&=~(1<<b)
#define TESTBIT(w, b)      (w&(1<<b))
#define SETLBITS(b)        (1<<b)-1
#define min(a, b)          (a<b)?a:b
#endif

extern void mumble(char *);
extern void mem_zero(char *, int);
extern void mem_copy(char *, char *, int);

extern void disable_signals();
extern void enable_signals();

extern void iloopck_init();
extern void iloopck();
extern void graceful_exit();

extern void Qk_basic_init(int);
extern void Qk_basic_shutdown();

#endif /* _UTIL_H_ */
