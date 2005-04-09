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
 * protocol.c: handlers for coherence protocols
 *
 *************************************************************************/

#include "protocol.h"
#include "types.h"
#include "ptable.h"
#include "config.h"
#include "message.h"
#include "util.h"

List *DUQ;

void
read_fault_handler_ws(Address gva_addr, Address lva_addr,
		      Ptentry *pte)
{
    ASSERT(pte->access == PAGE_MODE_NO);
    if (! get_page(pte, gva_addr, FOR_READ))
	return;

    
#ifdef UPD_TIMEOUT
    pte->num_usdiffs = min(pte->num_usdiffs*2, NUSDIFF_MAX);
    pte->useless_diffs = 0;
#endif

    set_access(pte, PAGE_MODE_RO);
}

void
write_fault_handler_ws(Address gva_addr, Address lva_addr,
				 Ptentry *pte)
{
    int was_invalid = (pte->access == PAGE_MODE_NO);

    if (pte->access == PAGE_MODE_NO)
	if (! get_page(pte, gva_addr, FOR_WRITE))
	    return;

    if (pte->copyset == add_copyset(0, Qknodeid))
	pte->powner = Qknodeid;
    else
    {
	pte->twin_addr = (char *) malloc(sizeof(char)*PAGE_SIZE);
	mem_copy(pte->addr, pte->twin_addr, PAGE_SIZE);
	list_append(DUQ, (void *) pte);
	pte->twin_present = 1;
    }

#ifdef UPD_TIMEOUT    
    if (was_invalid)
	pte->num_usdiffs = min(pte->num_usdiffs*2, NUSDIFF_MAX);
    pte->useless_diffs = 0;
#endif

    set_access(pte, PAGE_MODE_RW);
}

void forward_page_req(Address gva_addr, Threadid req_th,
		      Id gp_seqno, Id owner, int action)
{
    
    /* forward the request to probowner */
    Message *forward;
    MSG_INIT(forward, MSG_OP_GET_PAGE);
    MSG_INSERT(forward, gva_addr);
    MSG_INSERT(forward, gp_seqno);
    MSG_INSERT(forward, action);
    MSG_INSERT(forward, req_th);
    asend_msg(construct_threadid(owner, DSM_THREAD), forward);
}

void
page_request_handler_ws(Address gva_addr, Address lva_addr,
		      Ptentry *pte, Threadid req_th,
		      Id gp_seqno, int action)
{
    Message *msg, *reply;

    if (pte->access == PAGE_MODE_NO)
    {
	/* forward the request to probowner */
	ASSERT(pte->powner != Node(req_th));
	ASSERT(pte->powner != Qknodeid);
	forward_page_req(gva_addr, req_th, gp_seqno, pte->powner, action);
    }
    else
    {
	pte->copyset = add_copyset(pte->copyset, Node(req_th));
	if ((pte->access == PAGE_MODE_RW) && !pte->twin_present)
	    set_access(pte, PAGE_MODE_RO);
	
	MSG_INIT(msg, MSG_OP_PAGE_GRANTED);
	MSG_INSERT(msg, gva_addr);
	MSG_INSERT(msg, gp_seqno);
	MSG_INSERT(msg, pte->copyset);
	if ((pte->access == PAGE_MODE_RW) && pte->twin_present)
	    MSG_INSERT_BLK(msg, pte->twin_addr, PAGE_SIZE);
	else
	    MSG_INSERT_BLK(msg, pte->addr, PAGE_SIZE);

	reply = ssend_msg(req_th, msg);
	free_buffer(reply);
    }
}

static void invalidate_all_copies(Ptentry *pte, Address gva_addr)
{

}

void
read_fault_handler_wi(Address gva_addr, Address lva_addr,
				Ptentry *pte)
{
    if (! get_page(pte, gva_addr, FOR_READ))
	return;

    pte->copyset = add_copyset(pte->copyset, Qknodeid);
    set_access(pte, PAGE_MODE_RO);
}


void
write_fault_handler_wi(Address gva_addr, Address lva_addr,
				 Ptentry *pte)
{
    if ((pte->access == PAGE_MODE_RO) ||
	(pte->access == PAGE_MODE_RW))  /* got the page, just get ownership */
    {
	if (pte->powner != Qknodeid)
	    if (! get_page(gva_addr, FOR_WRITE))
		return;
    }
    else 
	if (! get_page(gva_addr, FOR_WRITE))
	    return;
    
    if (!pte->inval_pending)
    {
	PANIC("Inval_pending is not set");
    }
    invalidate_all_copies(pte, gva_addr);
    pte->inval_pending = 0;
    set_access(pte, PAGE_MODE_RW);
    pte->copyset = add_copyset(0, Qknodeid);
    
    pte->powner = Qknodeid;
}




void
page_request_handler_wi(Address gva_addr, Address lva_addr,
		      Ptentry *pte, Threadid req_th, Id gp_seqno,
                      int action)
{
    Message *msg, *reply;

    if (pte->inval_pending)
    {
	MSG_INIT(msg, MSG_OP_PAGE_DENIED);
	MSG_INSERT(msg, gva_addr);
	reply = ssend_msg(req_th, msg);
	free_buffer(reply);

	return;
    }
    if (pte->powner == Node(req_th))  
    {
	PANIC("This should not happen");
    }

    if (pte->powner != Qknodeid)  /* I am not owner */
    {
	/* a good optimization possible:
	 * if I have the page in read mode, but am not the owner
	 * and if the request is for read, then I can reply 
	 * with the page. No need to forward the request
	 */
	ASSERT(pte->powner != Node(req_th));

	forward_page_req(gva_addr, req_th, gp_seqno,
			 pte->powner, action);
	pte->powner = Node(req_th);

	return;
    }
    if (pte->powner == Qknodeid)
    {
	pte->copyset = add_copyset(pte->copyset, Node(req_th));
	if (action == FOR_WRITE)
	    pte->copyset = RESETBIT(pte->copyset, Qknodeid);


	MSG_INIT(msg, MSG_OP_PAGE_GRANTED);
	MSG_INSERT(msg, gva_addr);
	MSG_INSERT(msg, pte->copyset);
	MSG_INSERT(msg, gp_seqno);
	MSG_INSERT_BLK(msg, pte->addr, PAGE_SIZE);
	
	if (action == FOR_READ)
	{
	    set_access(pte, PAGE_MODE_RO);
	}
	if (action == FOR_WRITE)
	{
	    pte->copyset = 0;
	    set_access(pte, PAGE_MODE_NO);
	    pte->powner = Node(req_th);
	}
	
	reply = ssend_msg(req_th, msg);
	free_buffer(reply);
    }
}



