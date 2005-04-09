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
 * thread.c: thread handling
 *
 *************************************************************************/

/* Quarks header files */
#include "buffer.h"
#include "thread.h"
#include "util.h"

ltable_t ltable[MAX_THREADS];

/* the entries are as follows:
   entry for proc 0, thread 0
   entry for proc 0, thread 1
            ...
   entry for proc N, thread M-2
   entry for proc N, thread M-1
*/
rtable_t   rtable[MAX_PROCS*MAX_THREADS];  
static int tables_init = 0;
Id         Qknodeid = INVALID_ID;
int        Qkmaster=0;   /* Is this node the master node */
int        Qkchild=0;    /* Is this node a child node    */

int    num_children_forked=0;
int    num_children_terminated=0;

void slock_init(Spinlock *l)
{
    spin_lock_init(l);
}

int slock_locked(Spinlock *l)
{
    return spin_lock_locked(l);
}

int slock_try_lock(Spinlock *l)
{
    return spin_try_lock(l);
}

void slock_unlock(Spinlock *l)
{
    spin_unlock(l);
}

void slock_lock(Spinlock *l)
{
    spin_lock(l);
}

void slock_lock_yield(Spinlock *l)
{
    spin_lock_yield(l);
}

void thread_yield()
{
    cthread_yield();
}

Threadid thread_self()
{
    int i;
    cthread_t cid;
#ifdef HPUX
    extern cthread_t idle_thread_id;
#endif

    ASSERT(tables_init);

    if (Qknodeid == INVALID_ID) return 0;

    cid = cthread_self();
#ifdef HPUX
    if (cid == idle_thread_id) return 0;
#endif

    for (i=0; i<MAX_THREADS; i++)
	if (cid == ltable[i].ctid)
	    return construct_threadid(Qknodeid, i);

    PANIC("Thread not found");
    return INVALID_ID;
}

void init_thread_tables()
{
    int i;
    Message *msg;

    for (i=0; i<MAX_THREADS; i++)
    {
	ltable[i].ctid        = 0;
	ltable[i].send_buffer = 0;
	ltable[i].in_reply    = 0;
	ltable[i].in_msg      = 0;
    }
    for (i=0; i<MAX_THREADS*MAX_PROCS; i++)
    {
	rtable[i].last_reply       = 0;
	rtable[i].reply_done_seqno = 0;
	rtable[i].proc_msg         = 0;
    }
    tables_init = 1;
    /* make main a thread */
    ltable[0].ctid = cthread_self();
    msg = ltable[0].send_buffer = new_buffer();
    MSG_SET_FAMILY(msg, MSG_FAMILY_SYNCH);
    MSG_SET_TYPE(msg, MSG_TYPE_MESSAGE);

    ltable[0].in_reply = 0;
    ltable[0].in_msg = new_list();
}
    
Threadid quarks_thread_fork(Threadfn func, Any arg)
{
    int i;
    Message *msg;

    if (!tables_init)
    {
	PANIC("Thread tables not initialized");
    }

    for (i=1; i<MAX_THREADS; i++)  /* entry 0 is for main thread */
	if (!ltable[i].ctid) break;
    if (i >= MAX_THREADS)
	PANIC("Thread table full. Increase MAX_THREADS");

    msg = ltable[i].send_buffer = new_buffer();
    MSG_SET_FAMILY(msg, MSG_FAMILY_SYNCH);
    MSG_SET_TYPE(msg, MSG_TYPE_MESSAGE);
    msg->op = 0;
    msg->seqno = msg->numfrags = msg->fragno = 0;
    msg->length = 0;

    ltable[i].in_msg = new_list();
    disable_signals(); 
    ltable[i].ctid = cthread_fork((cthread_fn_t) func, (any_t) arg); 
    enable_signals(); 
    cthread_detach(ltable[i].ctid); 
    return construct_threadid(Qknodeid, i);
}

void Qk_wait_for_threads()
{
    /* 
     * waits till all child threads finish.
     */

    ASSERT(Qknodeid == 1); /* Not every one can wait! */
    while (num_children_terminated < num_children_forked)
	thread_yield();
}

int rtable_index(Threadid th)
{
    Id  nodeid = Node(th);
    Id  tid    = Thread(th);
    int entry  = 0;

    ASSERT(tables_init);
    entry  = nodeid*MAX_THREADS;
    entry += tid;

    return entry;
}

void rtable_setdiscflag(Id nodeid)
{
    int nstart = nodeid*MAX_THREADS;
    int i;

    for (i=0; i<MAX_THREADS; i++)
	rtable[nstart+i].discflag = 1;
}

void rtable_clean_nentry(Id nodeid)
{
    int nstart = nodeid*MAX_THREADS;
    int i;

    for (i=0; i<MAX_THREADS; i++)
    {
	if (rtable[nstart+i].last_reply)
	    free_buffer(rtable[nstart+i].last_reply);
	rtable[nstart+i].reply_done_seqno = 0;
	rtable[nstart+i].proc_msg = 0;
	rtable[nstart+i].discflag = 0;
    }

}



