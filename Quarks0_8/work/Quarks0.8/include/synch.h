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
 * synch.h: synchronization primitives.
 *
 *************************************************************************/

#ifndef _SYNCH_H_
#define _SYNCH_H_

#include "config.h"
#include "types.h"
#include "list.h"
#include "thread.h"

#define LOCKTABLESIZE        MAX_LOCKS
#define BARRIERTABLESIZE     MAX_BARRIERS

/* waiter entry in the waiters queue for lock */
typedef struct Waiter {
    Threadid      th;
    Id            al_seqno;
} Waiter;

#if 0
/* current this is unused */
typedef struct Lockstat {
    int num_lacq;  /* number of acquires at local node */ 
    int num_racq;  /* number of times it was acquired from remote node */
    int num_erel;   /* number of times it was released (empty) */
    int num_lrel;   /* number of times it was released (linked) */
    int num_req;   /* number of acquire requests coming in */
    int queue_length;  /* cumulative release time Q length */
    int num_forw;  /* number of times a req was forwarded */
    int hold_proc[NUM_PROCS];
} Lockstat;
#endif

typedef struct Locallock {    /* entry in the table kept by agents */
    Id      ID;
    int     held;
    Id      lockPO;
    int     acq_req_pending;
    List    *waiters_queue;
    mutex_t mutex;
    Id      creator;   /* true if self is the creator of the lock */
#if 0
    /* The following are not used right now */
    int     to_proc[NUM_PROCS];
    Lockstat lockstat;
#endif
} Locallock;

typedef struct Localbarrier {
    Id      ID;
    Id      creator;
} Localbarrier;

extern Locallock *get_locallock(Id lockid);

#endif  /* _SYNCH_H_ */

