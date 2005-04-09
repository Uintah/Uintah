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
 * thread.h: threads
 *
 *************************************************************************/

#ifndef _THREAD_H_
#define _THREAD_H_

#include "lock.h"      /* cthreads */
#include "cthreads.h"  /* cthreads */


/* Quarks header files: */
#include "types.h"
#include "list.h"
struct Message;

#define MAX_PROCS   32
#define MAX_THREADS  8

/* server */
#define DSM_SERVER_THREAD  1
/* clients */
#define DSM_THREAD         1

typedef struct Clientid {
    Hostaddr  host;       /* example: IP address */
    Processid pid;        /* example: pid        */
} Clientid;


typedef struct ltable_t {
    cthread_t ctid;
    struct Message   *send_buffer;
    struct Message   *in_reply;
    List      *in_msg;   /* incoming messages list */
} ltable_t;

typedef struct rtable_t {
    struct Message   *last_reply;
    Id        reply_done_seqno;
    Id        proc_msg;
    int       discflag;
} rtable_t;

extern Threadid Qk_thread_fork(Threadfn, Any);

extern void init_lock(Spinlock *);
extern int  slock_try_lock(Spinlock *);
extern void slock_unlock(Spinlock *);
extern void thread_yield();
extern Threadid thread_self();
extern void     init_thread_tables();
extern int  rtable_index(Threadid);
extern void rtable_setdiscflag(Id);
extern void rtable_clean_nentry(Id);

#define construct_threadid(nodeid, tid)  ((nodeid << 16) | tid)
#define Node(tid)   (tid >> 16)
#define Thread(tid) (tid & 0x0000ffff)

extern ltable_t ltable[MAX_THREADS];
extern rtable_t rtable[MAX_PROCS*MAX_THREADS];
extern int      Qkmaster;
extern int      Qkchild;

#endif  /* _THREAD_H_  */
