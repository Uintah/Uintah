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
 * message.c: reliable, in-order message sub-system on UDP
 *
 *************************************************************************/

#include <signal.h>

/* Quarks header files */
#include "message.h"
#include "types.h"
#include "thread.h"
#include "util.h"

/* external dependencies */
extern Message *new_buffer();
extern int  contable_index(Id);
extern Port *contable_port(int);
extern int  get_node_info(Id);
extern int  connect_node_request(Id);

static Id  sequence=2;  /* sequence number 1 is used for 
		         * the first message to bind with
			 * the server 
			 */
unsigned long timeout_time = TIMEOUT_TIME;



void asend_msg(Threadid to_thread, Message *msg)
{
    /* Asynchronous send. Does not wait for the reply.
     */
    int     ct_index;
    Message *save;
    Id      nodeid = Node(to_thread);
    
    if ((ct_index = contable_index(Node(to_thread))) == -1)
    {
	save = new_buffer();
	mem_copy((char *) msg+MSG_PACKET_OFFSET,
		 (char *) save+MSG_PACKET_OFFSET, 
		 sizeof(Message)-MSG_PACKET_OFFSET);
	if (!get_node_info(nodeid))
	    PANIC("cannot get node-info");
	if (!connect_node_request(nodeid))
	    PANIC("Cannot connect to node");
	ct_index = contable_index(nodeid);
	mem_copy((char *) save+MSG_PACKET_OFFSET,
		 (char *) msg+MSG_PACKET_OFFSET, 
		 sizeof(Message)-MSG_PACKET_OFFSET);
	free_buffer(save);
    }
    MSG_SET_FAMILY(msg, MSG_FAMILY_ASYNCH);
    msg->to_thread = to_thread;
    msg->from_thread = thread_self();
    send_packet(contable_port(ct_index), (char *) msg + MSG_PACKET_OFFSET, 
		msg->length + HEADER_SIZE);
    MSG_SET_FAMILY(msg, MSG_FAMILY_SYNCH);
}

static Message *get_reply(ltable_t *entry)
{
    /* Atomic operation to pick up reply from the "in_reply" slot
     * and making the slot 0.
     */
    Message *reply;
    
    disable_signals();
    reply = entry->in_reply;
    entry->in_reply = 0;
    enable_signals();
    
    return reply;
}



Message *ssend_msg(Threadid to_thread, Message *msg)
{
    /* Synchronous message send. Waits for a reply.
     */
    int         ct_index;
    Threadid    self;
    int         lt_index;
    Message 	*reply, *save;
    int 	num_retries=0;
    Id		nodeid = Node(to_thread);
    unsigned long start, end;

    if ((ct_index = contable_index(nodeid)) == -1)
    {
	save = new_buffer();
	mem_copy((char *) msg+MSG_PACKET_OFFSET,
		 (char *) save+MSG_PACKET_OFFSET, 
		 sizeof(Message)-MSG_PACKET_OFFSET);
	if (!get_node_info(nodeid))
	    PANIC("cannot get node-info");
	if (!connect_node_request(nodeid))
	    PANIC("Cannot connect to node");
	ct_index = contable_index(nodeid);
	mem_copy((char *) save+MSG_PACKET_OFFSET,
		 (char *) msg+MSG_PACKET_OFFSET, 
		 sizeof(Message)-MSG_PACKET_OFFSET);
	free_buffer(save);
    }

    ASSERT(msg->length <= MSG_DATA_SIZE);
    msg->to_thread = to_thread;
    self = msg->from_thread = thread_self();
    lt_index = Thread(self);

    msg->seqno = sequence++;

    while (1)
    {
	send_packet(contable_port(ct_index), (char *) msg+MSG_PACKET_OFFSET, 
		    msg->length + HEADER_SIZE);
	start = Qk_current_time();
	end   = start + timeout_time;

	if (num_retries > 0) 
	{
	    printf("Retry in ssend to node %d, msg op %d\n",
		   nodeid, msg->op);
	}
	
	/*
	 * Wait for timeoutTime milliseconds for the mail
	 * Give chance to other threads to run. May not be a good
	 * idea if the reply is needed asap.
	 */
	reply = 0;
	while (Qk_current_time() < end)
	{
	    thread_yield();
	    
	    if (reply = get_reply(&ltable[lt_index])) 
	    {
		if (reply->seqno == msg->seqno)
		    break;
		if (reply->seqno < msg->seqno) /* old reply */
		{
		    free_buffer(reply);
		    reply = 0;
		}
		else
		    if (reply->seqno > msg->seqno) /* not possible */
			PANIC("reply from the future");
	    }
	}
	if (reply) break;

	/* in case we timed out just as we received the message */
	if (reply = get_reply(&ltable[lt_index])) 
	{
	    if (reply->seqno == msg->seqno)
		break;
	    if (reply->seqno < msg->seqno) /* old reply */
	    {
		free_buffer(reply);
		reply = 0;
	    }
	    if (reply->seqno > msg->seqno) /* not possible */
		PANIC("reply from the future");
	}

	num_retries++;
	if (num_retries > MAX_RETRIES)
	{
	    printf("Too many retries in ssend...\n");
	    return (Message *) 0;
	}
    }

    ASSERT(reply->to_thread == self);
    ASSERT(reply->from_thread == msg->to_thread);
    if (reply->seqno != msg->seqno)
    {
	printf("reply->seqno = %d, msg->seqno = %d\n",
	       reply->seqno, msg->seqno);
	PANIC("seqno mismatch");
    }

    return reply;
}


Message *receive(int timeout)
{
    /* receive a message from the incoming message queue. If the queue 
     * is empty and timeout is specified as a positive integer, then 
     * the wait is terminated after that many milliseconds. Otherwise 
     * the thread waits until it receives a message.
     */
    
    Message  *msg;
    Threadid self = thread_self();
    int      lt_index = Thread(self);
    unsigned long start, end;

    if (timeout == 0)  /* indefinite polling */
    {
	while (list_empty(ltable[lt_index].in_msg)) thread_yield();
    }
    if (timeout > 0)
    {
	start = Qk_current_time();
	end   = start + timeout;
	while (Qk_current_time() < end)
	{
	    if (! list_empty(ltable[lt_index].in_msg)) break;

	    thread_yield();
	}
	if (list_empty(ltable[lt_index].in_msg))  /* timed out */
	{
	    return (Message *) 0;
	}
    }

    /* woken up due to message arrival */
    ASSERT(! list_empty(ltable[lt_index].in_msg));

    /* race condition possible, with list_append in asyncio_handler.
     * list_rem_head should be atomic. Due to atomic nature of the 
     * asyncio_handler, append is atomic.
     * Race1_label:  (match with similar label below)
     */
    disable_signals();
    msg = (Message *) list_rem_head(ltable[lt_index].in_msg);
    enable_signals();

    return msg;
}


void reply_msg(Message *msg)
{
    /* This reply is to the sender of the message.
     * The application only inserts data into the message.
     * The header is constructed here.
     */
    Message    *lr;
    Id         nodeid;
    int        ct_index;
    int        rt_index;
    Threadid   self = thread_self();

    ASSERT(msg_family(msg) == MSG_FAMILY_SYNCH);
    MSG_SET_TYPE(msg, MSG_TYPE_REPLY);
    ASSERT(msg->to_thread == self);
    msg->to_thread = msg->from_thread;
    msg->from_thread = self;

    nodeid = Node(msg->to_thread);
    if ((ct_index = contable_index(nodeid)) == -1)
	PANIC("unknown node");
    
    send_packet(contable_port(ct_index), (char *) msg+MSG_PACKET_OFFSET,
		msg->length + HEADER_SIZE);

    rt_index = rtable_index(msg->to_thread);
    if (lr = rtable[rt_index].last_reply) 
    {
	free_buffer(lr);
    }
    rtable[rt_index].last_reply = msg;
    rtable[rt_index].reply_done_seqno = msg->seqno;
    rtable[rt_index].proc_msg = 0;
}


/*-----------------------------------------------------------------------
               Functions related to async-thread
-----------------------------------------------------------------------*/

static Port from_port;     /* structure to hold port information of 
			    * incoming messages */

static void place_reply(int lt_index, Message *msg)
{
    /* place the reply message "msg" in the in_reply slot. Has to
     * take care of a message already existing in that slot.
     * The signals are disabled, since this routine is called
     * from the asyncio handler.
     */

    if ((ltable[lt_index].in_reply) == 0)  /* most common case */
    {
	ltable[lt_index].in_reply = msg;
	return;
    }
    else
    {
	if (ltable[lt_index].in_reply->seqno < msg->seqno)
	{
	    /* old reply in the slot. It must be discarded */
	    free_buffer(ltable[lt_index].in_reply);
	    ltable[lt_index].in_reply = msg;
	}
	else  
	    /* the slot contains fresh reply, discard the new arrival */
	    free_buffer(msg);
    }
}

void handle_asyncio(int sig, int code, struct sigcontext* scp, char* addr)
{
    Message      *msg, *lr;
    int          should_yield = 0;
    register int rt_index, lt_index;
    int          ct_index, size;

    /* XXX: we can setup the signal delivery such that the signals are 
     * automatically disabled when we enter the handler. That will
     * save us this system call. Since at the end of this handler
     * the signals are enabled, a yield() from the handler should 
     * not be a problem. Not done yet.
     */
    disable_signals();
    while (1) 
    {
	msg = new_buffer();
	size = get_packet(&from_port, (char *) msg + MSG_PACKET_OFFSET,
			  MSG_DATA_SIZE + HEADER_SIZE);

	if (size < 0) PANIC("get_packet failed");

	if (!size)
	{
	    free_buffer(msg);
	    break; 
	}
	
	ASSERT(size >= HEADER_SIZE);
	msg->length = size - HEADER_SIZE;

	lt_index = Thread(msg->to_thread);
	if (ltable[lt_index].ctid == 0)
	    PANIC("message for a non-existing thread");

	if (msg_family(msg) == MSG_FAMILY_ASYNCH)
	{
	    list_append(ltable[lt_index].in_msg, (void *) msg);
	    continue; 
	}
	
	rt_index = rtable_index(msg->from_thread);
	should_yield = (Thread(thread_self()) != lt_index);
	switch(msg_type(msg))
	{
	case MSG_TYPE_MESSAGE:

	    if (msg->seqno == rtable[rt_index].proc_msg) 
	    {
		/* Message is being processed */
		free_buffer(msg);
		printf("message %d from node %d repeated again.\n", 
		       msg->op, Node(msg->from_thread));
		continue;
	    }
	    if (msg->seqno == rtable[rt_index].reply_done_seqno)
	    {
		ASSERT(rtable[rt_index].last_reply);
		ct_index = contable_index(Node(msg->from_thread));
		ASSERT(ct_index != -1);
		lr = rtable[rt_index].last_reply;
		send_packet(contable_port(ct_index),
			    (char *) lr + MSG_PACKET_OFFSET,
			    lr->length + HEADER_SIZE);
		free_buffer(msg);
		continue; 
	    }
	    if (msg->seqno < rtable[rt_index].reply_done_seqno)
	    {
		/* an old message. just discard it. */
		free_buffer(msg);
		continue;
	    }
		
            if (rtable[rt_index].discflag)  /* disconnected */
		PANIC("message on a disconnected channel");

	    rtable[rt_index].proc_msg = msg->seqno;
	    
	    /* Race condition possible with list_rem_head in receive()
	     * Race1_label: (match with similar label above)
	     */
	    list_append(ltable[lt_index].in_msg, (void *) msg);

	    break;
	case MSG_TYPE_REPLY:
	    place_reply(lt_index, msg);
	    break;
	default:
	    PANIC("Invalid message type");
	}
    }

    /* XXX: should be done only if (should_yield == 1) */
    enable_signals();

    /* Actually should start the thread which is to receive it */
    if (should_yield) cthread_yield();
}


#ifdef MSGTEST
/**********
the following code is for testing the message sub-system
**********/

void communicate()
{
#define NUM_MSG 100000
#define NUM_PROC 6
    int i;
    Id other;
    Message *msg, *reply;

    for (i=0; i<NUM_MSG; i++)
    {
	MSG_INIT(msg, MSG_OP_NOP);
	MSG_INSERT(msg, i+1);
	while ((other = (random()%NUM_PROC)+1) == Qknodeid);
	reply = ssend_msg(construct_threadid(other, DSM_THREAD), msg);
	if (extract_long(reply->data, FIRST) != (i+1))
	    PANIC("Incorrect reply");
	free_buffer(reply);
    }
}
#endif
