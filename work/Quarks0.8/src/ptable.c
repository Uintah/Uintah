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
 * ptable.c: page table manipulations.
 *
 *************************************************************************/

#include <sys/types.h>
#include <sys/mman.h>

#include "ptable.h"
#include "config.h"
#include "message.h"
#include "thread.h"

static int page_req_timeout = TIMEOUT_TIME*10;

extern void handle_spurious_message(Message *);

/* MACHDEP: this routine is system dependent */
void set_access(Ptentry *pte, int mode)
{
    switch (mode)
    {
    case PAGE_MODE_RW:
	if (mprotect(pte->addr, PAGE_SIZE, PROT_READ|PROT_WRITE) < 0)
	    PANIC("mprotect failed");
	break;
    case PAGE_MODE_RO:
	if (mprotect(pte->addr, PAGE_SIZE, PROT_READ) < 0)
	    PANIC("mprotect failed");
	break;
    case PAGE_MODE_NO:
	if (mprotect(pte->addr, PAGE_SIZE, PROT_NONE) < 0)
	    PANIC("mprotect failed");
	break;
    default:
	PANIC("Illegal access mode");
    }

    pte->access = mode;
}

void set_protocol(Ptentry *pte, int prot)
{
    switch (prot)
    {
    case PROT_WRITE_INVALIDATE:
	pte->read_fault_handler = read_fault_handler_wi;
	pte->write_fault_handler = write_fault_handler_wi;
	pte->page_request_handler = page_request_handler_wi;
	break;
    case PROT_WRITE_SHARED:
	pte->read_fault_handler = read_fault_handler_ws;
	pte->write_fault_handler = write_fault_handler_ws;
	pte->page_request_handler = page_request_handler_ws;
	break;
    default:
	PANIC("Unknown protocol");
    }
    pte->protocol = prot;
}


void init_ptentry(Ptentry *pte, char *addr, Id powner, int access)
{
    pte->addr   = addr;
    pte->powner = powner;
    pte->access = access;
    pte->lock   = mutex_alloc();
    pte->copyset = add_copyset(0, powner);
    pte->in_copyset = 0;
    pte->sending_updates = 0;
    pte->inval_pending = 0;

    pte->num_usdiffs = NUSDIFF_MIN;
    pte->twin_present = 0;
    pte->twin_addr = 0;
    
    set_protocol(pte, DEFAULT_PROTOCOL);
}

int get_page(Ptentry *pte, Address gva_addr, int action)
{
    /* action is FOR_READ or FOR_WRITE */
    Message *msg, *reply;
    static Id page_seqno = 1;
    Id        seqno;
    int       try = 1;
    int       done = 0;
    Threadid  self = thread_self();

    seqno = page_seqno++;
    MSG_INIT(msg, MSG_OP_GET_PAGE);
    MSG_INSERT(msg, gva_addr);
    MSG_INSERT(msg, seqno);
    MSG_INSERT(msg, action);
    MSG_INSERT(msg, self);

    while (!done)
    {
	if (try)
	    asend_msg(construct_threadid(pte->powner, DSM_THREAD), msg);
	
	reply = receive(page_req_timeout);
	if (!reply)
	{
	    /* timed out */
	    mumble("Retry in getpage... ");
	    try = 1;
	    continue;
	}

	if (extract_long(reply->data, FIRST) != gva_addr)
	{
	    handle_spurious_message(reply);
	    try = 0;
	    continue;
	}
	
	switch(reply->op)
	{
	case MSG_OP_PAGE_GRANTED:
	    if (extract_long(reply->data, SECOND) < seqno)
		handle_spurious_message(reply);
	    else
		done = 1;
	    break;
	case MSG_OP_PAGE_DENIED:
	    MSG_CLEAR(reply);
	    reply_msg(reply);
	    return 0;
	    break;
	default:
	    handle_spurious_message(reply);
	    try = 0;
	    break;
	}
    }

    /* We finally got the page */

    pte->copyset = extract_long(reply->data, THIRD);
    set_access(pte, PAGE_MODE_RW);  /* to copy the contents */
    mem_copy(reply->data + 12, pte->addr, PAGE_SIZE);
    if (action == FOR_READ)
	set_access(pte, PAGE_MODE_RO);

    MSG_CLEAR(reply);
    reply_msg(reply);
    return 1;
}








