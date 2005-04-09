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
 * dsm_thread.c: handles incoming DSM requests
 *
 *************************************************************************/

#include "types.h"
#include "region.h"
#include "thread.h"
#include "message.h"
#include "ptable.h"
#include "port.h"
#include "synch.h"

extern int num_children_terminated;

void handle_spurious_message(Message *msg)
{
    /* Some messages invariably get late. We just need to identify them
     * and drop them.
     */
    switch (msg->op)
    { 
    case MSG_OP_LOCK_REQ_QUEUED:
    case MSG_OP_PAGE_GRANTED:
	if (msg_family(msg) == MSG_FAMILY_SYNCH)
	{
	    msg->length = 0;
	    reply_msg(msg);
	}
	else if (msg_family(msg) == MSG_FAMILY_ASYNCH)
	    free_buffer(msg);
	else
	    PANIC("Unknown family of spurious message");
	break;
    default:
	printf("Received message %d \n", msg->op);
	PANIC("Dont know what to do with this spurious message");
	break;
    }
}

static void
handle_page_request(Message *request)
{
    /* a request for a page from other node 
     */
    Message  *reply;
    Address  gva_addr;     /* Global virtual address of the fault        */
    Threadid req_th;       /* Requesting thread                          */
    Id       gp_seqno;     /* seqno for weeding out duplicates          */
    int      action;       /* FOR_READ or FOR_WRITE                      */
    Address  lva_addr;
    Ptentry  *pte;
    Id       regionid;

    gva_addr = (Address) extract_long(request->data, FIRST);
    gp_seqno = (Id)      extract_long(request->data, SECOND);
    action   = (int)     extract_long(request->data, THIRD);
    req_th   = (Threadid) extract_long(request->data, FOURTH);
    free_buffer(request);
    

    lva_addr = gva2lva(gva_addr);
    regionid = gaddr2region(gva_addr);

    pte = get_pte_locked(regionid, lva_addr);
    (*pte->page_request_handler)(gva_addr, lva_addr, pte, req_th, 
				 gp_seqno, action);
    unlock_pte(pte);
}

static void 
forward_lock_req(Id lockid, Id req_th, Id seqno, Id forw_to)
{
    Message *msg;

    MSG_INIT(msg, MSG_OP_ACQUIRE_LOCK);
    MSG_INSERT(msg, lockid);
    MSG_INSERT(msg, req_th);
    MSG_INSERT(msg, seqno);
    asend_msg(construct_threadid(forw_to, DSM_THREAD), msg);
}


static void
handle_lock_request(Message *request)
{
    /* A request for a lock from other node.
     */
    Id lockid;
    Id req_th;
    Id seqno;
    Locallock *lk;
    Waiter *rec;
    Message *msg, *reply;

    lockid = (Id) extract_long(request->data, FIRST);
    req_th = (Id) extract_long(request->data, SECOND);
    seqno  = (Id) extract_long(request->data, THIRD);
    free_buffer(request);

    lk = get_locallock(lockid);

    mutex_lock(lk->mutex);
    if (lk->held || lk->acq_req_pending)
    {
	/* Should queue up the request */
	rec = (Waiter *) malloc(sizeof(Waiter));
	rec->th = req_th;
	rec->al_seqno = seqno;
	list_append(lk->waiters_queue, (void *) rec);

	MSG_INIT(msg, MSG_OP_LOCK_REQ_QUEUED);	
	MSG_INSERT(msg, lockid);
	reply = ssend_msg(req_th, msg);
	free_buffer(reply);
	mutex_unlock(lk->mutex);
	return;
    }
    
    if (lk->lockPO != Qknodeid)  
    {
	forward_lock_req(lockid, req_th, seqno, lk->lockPO);
	mutex_unlock(lk->mutex);
	return;
    }
    
    ASSERT(lk->lockPO == Qknodeid);
    ASSERT(!lk->held);

    /* lock is not held */
    MSG_INIT(msg, MSG_OP_LOCK_GRANTED);
    MSG_INSERT(msg, lockid);
    MSG_INSERT(msg, seqno);
    if (!list_empty(lk->waiters_queue))	
	PANIC("Lock waiters list not empty");
    lk->lockPO = Node(req_th);
    mutex_unlock(lk->mutex);
    
    reply = ssend_msg(req_th, msg);
    free_buffer(reply);
}

static void incorporate_diff(Message *msg)
{
    /* Incorporate the diffs into local pages. Care is to be taken
     * if the local pages have been invalidated due to update timeout. 
     * In that case, the update request is replied with a NACK.
     */
    Address       gva_addr;
    Address       lva_addr;
    unsigned long copyset;
    Id            req_node;
    Ptentry       *pte;
    Id            regionid;

    gva_addr = (Address) extract_long(msg->data, FIRST);
    copyset = extract_long(msg->data, SECOND);
    lva_addr = gva2lva(gva_addr);
    req_node = Node(msg->from_thread);

    regionid = gaddr2region(gva_addr);
    if (regionid == INVALID_ID)
    {
	MSG_CLEAR(msg);
	MSG_INSERT(msg, DIFF_NACK);
	reply_msg(msg);
	return;
    }

    pte = get_pte_locked(regionid, lva_addr);
/*    fprintf(stdout, "Diff received from %d\n", req_proc);   */
    if (pte->access == PAGE_MODE_NO)
    {
	MSG_CLEAR(msg);
/* 	fprintf(stdout, "DIFF BEING NACKED (page invalid) !!!!!!!!\n");  */
	MSG_INSERT(msg, DIFF_NACK);
	reply_msg(msg);
	unlock_pte(pte);
	return;
    }

#ifdef UPD_TIMEOUT
    if (pte->access == PAGE_MODE_RO)
	pte->useless_diffs++;
    if (pte->access == PAGE_MODE_RW)
	pte->useless_diffs = 0;

    ASSERT(pte->num_usdiffs > 0);
    if (pte->useless_diffs > pte->num_usdiffs)
    {
	/* Page has been lying around untouched for "too" long.
	 * Probably the node does not want it no more.
	 */
	set_access(pte, PAGE_MODE_NO);
	MSG_CLEAR(msg);
	if (pte->powner == Qknodeid)
	{
	    pte->powner = req_node;
	    MSG_INSERT(msg, DIFF_NACK_XFER_OWN);
	}
	else
	    MSG_INSERT(msg, DIFF_NACK);
	reply_msg(msg);
	unlock_pte(pte);
	return;
    }
#endif

    disable_signals();
    decode_object((unsigned long *) (msg->data+8),
		  pte, (msg->length - 8)/4);
    enable_signals();

    MSG_CLEAR(msg);
    MSG_INSERT(msg, DIFF_ACK);
    MSG_INSERT(msg, pte->copyset);
    reply_msg(msg);

#ifdef UPD_TIMEOUT
    /* XXX should send updates to anyone not mentioned in copyset */
#endif

    unlock_pte(pte);
}

typedef void (*FUNCPTR)();

static Any thread_stub(Any func)
{
    Message *msg, *reply;

    printf("Thread stub called, invoking the function\n");
    (*((FUNCPTR) func))();

    printf("Invoked function ends. Thread stub terminating.\n");

    MSG_INIT(msg, MSG_OP_THREAD_TERMINATED);
    MSG_INSERT(msg, thread_self());

    reply = ssend_msg(construct_threadid(1, DSM_THREAD), msg);
    free_buffer(reply);

    return (Any) 0;
}

static void create_thread(Message *request)
{
    Threadfn func;

    func = (Threadfn) extract_long(request->data, FIRST);
    MSG_CLEAR(request);
    reply_msg(request);
    
    unmarshall_regions();
    unmarshall_locks();
    
    quarks_thread_fork((Threadfn) thread_stub, (Any) func);
    printf("Thread created\n");
}

Any dsm_thread(Any arg)
{
    Message *request;
    Id      nodeid;
    int     value;
    int     terminate;

    printf("Dsm thread starting. id = 0x%x\n", thread_self());
    while (1)
    {
	request = receive(0);
	switch (request->op)
	{
	case MSG_OP_CONNECT:
	    nodeid = extract_long(request->data, FIRST);
	    accept_node(nodeid, (Port *) (request->data+4));
	    MSG_CLEAR(request);	
	    MSG_INSERT(request, 0);
	    reply_msg(request);
	    break;
	case MSG_OP_DISCONNECT:
	    disconnect_node_mark(Node(request->from_thread));
	    MSG_CLEAR(request);
	    MSG_INSERT(request, 0);
	    reply_msg(request);
	    break;
	case MSG_OP_DISCONNECT_COMMIT:
	    disconnect_node(Node(request->from_thread));
	    free_buffer(request);
	    break;
	case MSG_OP_NOP:
	    value = extract_long(request->data, FIRST);
	    printf("got %d from node %d\n", value, Node(request->from_thread));
	    fflush(stdout);
	    reply_msg(request);
	    break;

	case MSG_OP_GET_PAGE:
	    handle_page_request(request); /* will reply/free */
	    break;
	case MSG_OP_ACQUIRE_LOCK:
	    handle_lock_request(request); /* will reply/free */
	    break;
	case MSG_OP_PAGE_DIFF:
	    incorporate_diff(request);    /* will reply */
	    break;
	case MSG_OP_CREATE_THREAD:
	    create_thread(request);
	    break;
	case MSG_OP_THREAD_TERMINATED:
	    if (Qknodeid != 1)
		PANIC("I should not be getting this message");
	    MSG_CLEAR(request);
	    reply_msg(request);
	    num_children_terminated++;
	    break;
	case MSG_OP_SHUTDOWN:
	    terminate = (int) extract_long(request->data, FIRST);
	    MSG_CLEAR(request);
	    reply_msg(request);
	    Qk_shutdown(terminate);
	    break;
	default:
	    PANIC("Unknown request");
	}
    }

    return (Any) 0;
}

