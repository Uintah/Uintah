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
 * synch.c: Synchronization primitives
 *
 *************************************************************************/

#include "synch.h"
#include "message.h"
#include "protocol.h"
#include "ptable.h"

static Locallock    lltable[LOCKTABLESIZE];
static Localbarrier lbtable[BARRIERTABLESIZE];

static mutex_t DUQ_lock;
static int DUQ_lock_init = 0;

static void init_llentry(int i)
{
    lltable[i].ID = INVALID_ID;
    lltable[i].held = 0;
    lltable[i].lockPO = INVALID_ID;
    lltable[i].acq_req_pending = 0;
    if (lltable[i].waiters_queue)
	free(lltable[i].waiters_queue);
    if (lltable[i].mutex)
	mutex_free(lltable[i].mutex);
    lltable[i].creator = INVALID_ID;
}

static void init_lbentry(int i)
{
    lbtable[i].ID = INVALID_ID;
    lbtable[i].creator = INVALID_ID;
}

void init_lltable()
{
    int i;

    for (i=0; i<LOCKTABLESIZE; i++)
	init_llentry(i);
}

void init_lbtable()
{
    int i;

    for (i=0; i<BARRIERTABLESIZE; i++)
	init_lbentry(i);
}
 
static int add_lltable(Id lockid)
{
    static int next = 0;
    int id;

    id = next++;
    ASSERT(lltable[id].ID == INVALID_ID);
    lltable[id].ID = lockid;
    lltable[id].held = 0;
    lltable[id].acq_req_pending = 0;
    lltable[id].waiters_queue = new_list();
    lltable[id].mutex = mutex_alloc();
    return id;
}

static int add_lbtable(Id barrierid)
{
    static int next = 0;
    int id;

    id = next++;
    ASSERT(lbtable[id].ID == INVALID_ID);
    lbtable[id].ID = barrierid;
    return id;
}

static int lltable_index(Id lockid)
{
    int i;

    for (i=0; i<LOCKTABLESIZE; i++)
	if (lltable[i].ID == lockid)
	    return i;

    return -1;
}

static int lbtable_index(Id barrierid)
{
    int i;

    for (i=0; i<BARRIERTABLESIZE; i++)
	if (lbtable[i].ID == barrierid)
	    return i;

    return -1;
}

Locallock *get_locallock(Id lockid)
{
    Locallock *lk;
    int       llindex;

    if ((llindex = lltable_index(lockid)) == -1)
	PANIC("Request for unknown lock");
    lk = &lltable[llindex];
    return lk;
}

static int pack_list(Message *msg, List *list)
{
    Waiter *rec;
    Threadid th;
    int qlen=0;

    while (! list_empty(list))
    {
	rec = (Waiter *) list_rem_head(list);
	MSG_INSERT(msg, rec->th);
	MSG_INSERT(msg, rec->al_seqno);
	free(rec);
	qlen++;
    }
    ASSERT(list_empty(list));
    return qlen;
}

static void unpack_list(char *buf, List *list, int num)
{
    /* buf:  points to the start of packed data
     * list: the list to be constructed
     * num:  number of list items in "buf"
     */

    int i;
    Threadid th, self = thread_self();
    Waiter *rec;
    unsigned long *data;
    
    ASSERT(num >= 0);
    data = (unsigned long *) buf;
    for (i=0; i<num; i++)
    {
	rec = (Waiter *) malloc(sizeof(Waiter));
	rec->th = (Id) *data++;
	rec->al_seqno = *data++;
	if (Node(rec->th) != Qknodeid)
	    list_append(list, (void *) rec);
	else
	    free(rec);
    }
}
    
void Qk_acquire(Id lockid)
{
    Threadid  th;
    Locallock *lk;
    int       llindex;

    int       numretries=0;
    int       try=1, queued=0;
    Message   *msg, *reply;
    static unsigned long al_seqno=1;
    unsigned long seqno;


    if ((llindex = lltable_index(lockid)) == -1)
	PANIC("Lock does not exist");
    lk = &lltable[llindex];

    ASSERT(Thread(thread_self()) != DSM_THREAD);

    mutex_lock(lk->mutex);
    if (lk->lockPO == Qknodeid)
    {
	/* XXX: The following assertions assume that there is one
	 * application thread. Will have to modify for multi-threaded
	 * Quarks.
	 */
	ASSERT(! lk->acq_req_pending);
	ASSERT(! lk->held);
	lk->held = 1; 
	mutex_unlock(lk->mutex);
	return;
    }

    seqno = al_seqno++;
    lk->acq_req_pending = 1;
    MSG_INIT(msg, MSG_OP_ACQUIRE_LOCK);
    MSG_INSERT(msg, lockid);        /* lockid        */
    MSG_INSERT(msg, thread_self()); /* req_th        */
    MSG_INSERT(msg, seqno);         /* seqno for acquire lock */
    
    while (1)
    {
	if (try && !queued)	
	    asend_msg(construct_threadid(lk->lockPO, DSM_THREAD), msg);

	mutex_unlock(lk->mutex);
	reply = receive(10*timeout_time);
	mutex_lock(lk->mutex);

	if (!reply)
	{
	    /* timed out */
	    try = 1;
	    printf("Retry in lock acquire... \n");
	    if (++numretries > MAX_RETRIES)
		PANIC("Cannot reach the lock owner");
	    continue;
	}
	
	if (extract_long(reply->data, FIRST) != lockid)
	{
	    handle_spurious_message(reply, 0);
	    try=0;
	    continue;
	}	
	if (reply->op == MSG_OP_LOCK_GRANTED) 
	{	
	    if (extract_long(reply->data, SECOND) < seqno)
	    {
		printf("Got an old acquire lock response. Dropping it\n");
		handle_spurious_message(reply, 0);
	    }
	    else
		break;
	}
	else
	{
	    if (reply->op == MSG_OP_LOCK_REQ_QUEUED)
	    {
		try = 0;
		queued = 1;
		reply_msg(reply);
	    }
	    else
	    {
		handle_spurious_message(reply, 0);
		try = 0;
	    }
	}
    }


    /* We finally got the lock */
    ASSERT(reply->op == MSG_OP_LOCK_GRANTED);
    ASSERT(msg_family(reply) == MSG_FAMILY_SYNCH);

    unpack_list(reply->data+8, lk->waiters_queue, 
		(reply->length-8)/sizeof(Waiter));
    MSG_CLEAR(reply);
    reply_msg(reply);

    lk->lockPO = Qknodeid;
    lk->acq_req_pending = 0;
    lk->held = 1;

    mutex_unlock(lk->mutex);
}


static int send_updates(unsigned long *copyset, Message *diff_msg)
{
    Message *diff_reply;
    int num_sent = 0;
    unsigned long cp = *copyset;
    unsigned long newcp, done=0;
    unsigned long prev_copyset;
    int xfer_own = 0;
    Id  pid = 1;
    Threadid th;
/*    fprintf(stdout, "Sending updates. Sent to "); fflush(stdout);  */

    RESETBIT(cp, Qknodeid);
    SETBIT(done, Qknodeid);
    while (cp)
    {
	if (!TESTBIT(cp, pid)) 
	{
	    pid = (pid == 31) ? 1 : pid+1;
	    continue; 
	}
	RESETBIT(cp, pid);
	SETBIT(done, pid);
	if (pid == Qknodeid) continue;
	
	th = construct_threadid(pid, DSM_THREAD);

	diff_reply = ssend_msg(th, diff_msg);
/*	fprintf(stdout, "%d ", pid); fflush(stdout);  */
	num_sent++;
	if (! diff_reply)
	    PANIC("Could not send diffs");

	if (extract_long(diff_reply->data, FIRST) == DIFF_NACK)
	{
/*	    fprintf(stdout, "Received a DIFF_NACK\n");  */
	    RESETBIT(*copyset, pid);
	    *(unsigned long *) (diff_msg->data+4) = *copyset;
	    free_buffer(diff_reply);
	    continue;
	}
	if (extract_long(diff_reply->data, FIRST) == DIFF_NACK_XFER_OWN)
	{
/*	    fprintf(stdout, "Received a DIFF_NACK_XFER_OWN\n");  */
	    RESETBIT(*copyset, pid);
	    *(unsigned long *) (diff_msg->data+4) = *copyset;
	    xfer_own = 1;
	    free_buffer(diff_reply);
	    continue;
	}
	if (extract_long(diff_reply->data, FIRST) != DIFF_ACK)
	    PANIC("Spurious message received");
	newcp = extract_long(diff_reply->data, SECOND);
	/* XXX Shouldn't we modify the copyset field in the message? */
	cp = cp | (newcp & ~done);
	*copyset = *copyset | (newcp & ~done);

	free_buffer(diff_reply);
    }
/*    fprintf(stdout, " DONE !!\n");  */

    return xfer_own;
}

static void lockDUQ()
{
    if (! DUQ_lock_init)
	DUQ_lock = mutex_alloc();

    mutex_lock(DUQ_lock);
}

static void unlockDUQ()
{
    mutex_unlock(DUQ_lock);
}


static void PurgeDUQ()
{
    int     size;
    Message *diff_msg;
    Ptentry *pte;
    Address gva_addr;

    lockDUQ();    
    while (! list_empty(DUQ))
    {
	pte = (Ptentry *) list_rem_head(DUQ);
	if ((pte->access == PAGE_MODE_RW) ||
	    (pte->access == PAGE_MODE_RO))
	{
	    lock_pte(pte);
	    
	    if (pte->twin_present)
	    {
		ASSERT(pte->copyset != add_copyset(0, Qknodeid));
		
		MSG_INIT(diff_msg, MSG_OP_PAGE_DIFF);
		gva_addr = lva2gva(pte->addr);
		MSG_INSERT(diff_msg, gva_addr);
		MSG_INSERT(diff_msg, pte->copyset);
		encode_object((unsigned long *) pte->addr,
			      (unsigned long *) pte->twin_addr,
			      (unsigned long *) (diff_msg->data+8),
			      &size);
		/* size is in words, not bytes */
		diff_msg->length += size*4;
		
/*
		free(pte->twin_addr);
		pte->twin_addr = 0; 
		pte->twin_present = 0;
		set_access(pte, PAGE_MODE_RO);
		pte->in_copyset = 0;
*/

		pte->sending_updates = 1;
		unlock_pte(pte);

		if (size)
		{
		    if (send_updates(&pte->copyset, diff_msg))
			pte->powner = Qknodeid;
		}

		lock_pte(pte);
/*		pte->copyset |= pte->in_copyset; */
		free(pte->twin_addr);
		pte->twin_addr = 0; 
		pte->twin_present = 0;
		set_access(pte, PAGE_MODE_RO);
		pte->sending_updates = 0;
		pte->in_copyset = 0;
		unlock_pte(pte);
	    }
	    else
	    {
		set_access(pte, PAGE_MODE_RO);
		unlock_pte(pte);
	    }
	}
    }
    unlockDUQ();
}

void Qk_release(Id lockid)
{
    Locallock *lk;
    int       llindex;
    int       qlen=0;
    Waiter    *rec;
    Threadid  th;
    Id        al_seqno;
    Id        node;
    Message   *msg, *reply;

    if ((llindex = lltable_index(lockid)) == -1)
	PANIC("Lock unknown");

    lk = &lltable[llindex];

    PurgeDUQ();

    mutex_lock(lk->mutex);
    if (! list_empty(lk->waiters_queue))
    {
	rec = (Waiter *) list_rem_head(lk->waiters_queue);

	th = rec->th;
	al_seqno = rec->al_seqno;
	free(rec);

	node = Node(th);
	ASSERT(node != Qknodeid);
	
	lk->lockPO = node;

	MSG_INIT(msg, MSG_OP_LOCK_GRANTED);	
	MSG_INSERT(msg, lockid);
	MSG_INSERT(msg, al_seqno);
	qlen = pack_list(msg, lk->waiters_queue);
	reply = ssend_msg(th, msg);
	free_buffer(reply); 
    }
#ifdef OPT_LOCK
    else
    {
	int max=0, total=0;
	ProcID which_proc = 0;
	for (int i=0; i<NUM_PROCS; i++)
	{
	    if (lk->to_proc[i] > max) {max = lk->to_proc[i]; which_proc = i;}
	    total += lk->to_proc[i];
	}
	if (which_proc != 0)
	{
	    double percent = (double) max / (double) total;
	    if ((total > 1) && (percent > LOCK_GIVEUP_THRESHOLD))
	    {
		giveup_lock(lk, which_proc);
		/* Above must have set held to 0, and released lk->mutex */
		return;
	    }
	}
    }
#endif

    lk->held = 0;
    mutex_unlock(lk->mutex);
}



Id Qk_newlock()
{
    Message *request, *reply;
    Id      allocid;
    int     llindex;

    MSG_INIT(request, MSG_OP_ALLOC_LOCK);
    reply = ssend_msg(DSM_SERVER_THREAD, request);
    allocid = (Id) extract_long(reply->data, FIRST);

    if (allocid == INVALID_ID)
	PANIC("Could not allocate a lock");
    
    free_buffer(reply);

    llindex = add_lltable(allocid);
    lltable[llindex].lockPO = Qknodeid;
    lltable[llindex].creator = Qknodeid;
    return allocid;
}

void quarks_define_lock(Id lockid, Id owner)
{
    int llindex;

    llindex = add_lltable(lockid);
    lltable[llindex].lockPO = owner;
    lltable[llindex].creator = INVALID_ID;
}

static void freelock(Id lockid)
{
    Message *request, *reply;
    int llindex;

    llindex = lltable_index(lockid);
    init_llentry(llindex);

    MSG_INIT(request, MSG_OP_FREE_LOCK);
    MSG_INSERT(request, lockid);
    reply = ssend_msg(DSM_SERVER_THREAD, request);
    free_buffer(reply);
}

Id Qk_newbarrier()
{
    Message *request, *reply;
    Id      allocid;
    int     lbindex;

    MSG_INIT(request, MSG_OP_ALLOC_BARRIER);
    reply = ssend_msg(DSM_SERVER_THREAD, request);
    allocid = (Id) extract_long(reply->data, FIRST);
    
    if (allocid == 0xffffffff)
	PANIC("Could not allocate a barrier");

    free_buffer(reply);

    lbindex = add_lbtable(allocid);
    lbtable[lbindex].creator = Qknodeid;

    return allocid;
}

static void freebarrier(Id barrierid)
{
    Message *request, *reply;

    MSG_INIT(request, MSG_OP_FREE_BARRIER);
    MSG_INSERT(request, barrierid);
    reply = ssend_msg(DSM_SERVER_THREAD, request);
    free_buffer(reply);
}

void Qk_wait_barrier(Id ba_id, int num_crossers)
{
    Message *request, *reply;
    Message *msg;

    PurgeDUQ();

    ASSERT(num_crossers > 0);
    if (num_crossers == 1) return;

    MSG_INIT(request, MSG_OP_BARRIER_WAIT);
    MSG_INSERT(request, ba_id);
    MSG_INSERT(request, num_crossers);
    /* DBGCODE */
    reply = ssend_msg(DSM_SERVER_THREAD, request);
    if (extract_long(reply->data, FIRST) != MSG_OP_BARRIER_CROSSED)
    {
	ASSERT(extract_long(reply->data, FIRST) == MSG_OP_WAIT_FOR_BARRIER);
	/* Will have to wait for the barrier */
	free_buffer(reply);

	msg = receive(0);
	while (msg->op != MSG_OP_BARRIER_CROSSED)
	{
	    handle_spurious_message(msg);
	    msg = receive(0);
	}
	ASSERT(extract_long(msg->data, FIRST) == ba_id);
	MSG_CLEAR(msg);
	reply_msg(msg);
    }
    else
	free_buffer(reply);

}

void free_all_locks()
{
    int i;

    for (i=0; i<LOCKTABLESIZE; i++)
    {
	if ((lltable[i].ID != INVALID_ID) && 
	    (lltable[i].creator == Qknodeid))
	    freelock(lltable[i].ID);
    }
}

void free_all_barriers()
{
    int i;

    for (i=0; i<BARRIERTABLESIZE; i++)
    {
	if ((lbtable[i].ID != INVALID_ID) && 
	    (lbtable[i].creator == Qknodeid))
	    freebarrier(lbtable[i].ID);
    }
}


#define NUMLOCK_PERMSG ((MSG_DATA_SIZE-4-4)/4)
void marshall_locks(Threadid th)
{
    Message *msg, *reply;
    unsigned long *numlocks, *data;
    int i;


    MSG_INIT(msg, MSG_OP_MARSHALLED_LOCKS);
    numlocks = (unsigned long *) msg->data;
    data   = (unsigned long *) (msg->data + 4);
    *numlocks = 0;
    for (i=0; i<LOCKTABLESIZE; i++)
	if ((lltable[i].ID != INVALID_ID) &&
	    (lltable[i].creator == Qknodeid))
	{
	    *data++ = lltable[i].ID;
	    (*numlocks)++;
	    if (*numlocks >= NUMLOCK_PERMSG)
	    {
		msg->length = 4 + (*numlocks)*4;

		reply = ssend_msg(th, msg);
		free_buffer(reply);
		*numlocks = 0;
		data = (unsigned long *) (msg->data + 4);
	    }
	}
    msg->length = 4 + (*numlocks)*4;
    reply = ssend_msg(th, msg);
    free_buffer(reply);
    
}

void unmarshall_locks()
{
    unsigned long *data;
    Id lockid;
    int done  = 0;
    int i, numlocks;
    Message *msg;

    while (!done)
    {
	msg = receive(0);
	ASSERT(msg->op == MSG_OP_MARSHALLED_LOCKS);
	numlocks = *(int *) msg->data;
	if (numlocks < NUMLOCK_PERMSG)
	    done = 1;
	data = (unsigned long *) (msg->data+4);
	for (i=0; i<numlocks; i++)
	{
	    lockid = (Id) *data++;
	    quarks_define_lock(lockid, Node(msg->from_thread));
	}
	MSG_CLEAR(msg);
	reply_msg(msg);
    }
}

#if 0   
extern "C" ConditionID
Qk_newcondition()
{
    Message *request;
    MSG_INIT(request, MSG_OP_ALLOC_CONDITION);
    Message *reply = request->Send(DSM_SERVER_THREAD);
    ConditionID allocid = (ConditionID) ExtractLong(reply->data, FIRST);
    
    if (allocid == 0xffffffff)
	PANIC("Could not allocate a condition");

    free_buffer(reply);
    return allocid;
}

extern "C" void
Qk_wait_condition(ConditionID condid, LockID lockid)
{
    /* Wait for condition "condid" to be true. Release the lock
     * lockid while we are waiting. If (lockid == 0) then we don't
     * wait for any locks (0 is an invalid id for a lock). 
     */

    Message *request, *reply;
    int done;

    if (lockid) Qk_release(lockid);

    MSG_INIT(request, MSG_OP_CONDITION_WAIT);
    MSG_INSERT(request, condid);
    reply = request->Send(DSM_SERVER_THREAD);

    ASSERT(ExtractLong(reply->data, FIRST) == condid);
    if (reply->header->op == MSG_OP_WAIT_FOR_CONDITION)
    {
	free_buffer(reply);
	done = 0;
	while (!done)
	{
	    reply = Receive(0);
	    if ((reply->header->op != MSG_OP_CONDITION_TRUE) ||
		(ExtractLong(reply->data, FIRST) != condid))
		handle_spurious_message(reply, MSG_OP_CONDITION_TRUE);
	    else 
		done = 1;
	}
    }
    ASSERT(reply->header->op == MSG_OP_CONDITION_TRUE);
    
    if (lockid) Qk_acquire(lockid);
}

extern "C" void
Qk_signal(ConditionID condid)
{
    Message *request, *reply;

    PurgeDUQ();

    MSG_INIT(request, MSG_OP_CONDITION_SIGNAL);
    MSG_INSERT(request, condid);
    reply = request->Send(DSM_SERVER_THREAD);
    free_buffer(reply);
}

extern "C" void
Qk_broadcast(ConditionID condid)
{
    Message *request, *reply;

    PurgeDUQ();

    MSG_INIT(request, MSG_OP_CONDITION_BROADCAST);
    MSG_INSERT(request, condid);
    reply = request->Send(DSM_SERVER_THREAD);
    free_buffer(reply);
}


void 
print_lock_stats(FILE *fp)
{
    int i;
    LocalLock *lk;

    for (i=0; i<LOCKTABLESIZE; i++)
    {
	lk = local_lock_table[i];
	if (!lk) continue;
	fprintf(fp, "--------------------------------------------------\n");
	fprintf(fp, "LockID = %d\n", lk->ID);
	fprintf(fp, "    local_acq = %d, remote_acq = %d\n",
		lk->lockstat.num_lacq, lk->lockstat.num_racq);
	fprintf(fp, "    empty release = %d, linked release = %d\n",
		lk->lockstat.num_erel, lk->lockstat.num_lrel);
	fprintf(fp, "    num_req = %d, num_forw = %d\n",
		lk->lockstat.num_req, lk->lockstat.num_forw);
	fprintf(fp, "    avg queue length = %lf\n",
		(double)lk->lockstat.queue_length /
		(double)lk->lockstat.num_lrel);
    }
}
#endif


