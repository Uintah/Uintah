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
 * message_op.h: defines the message operations
 *
 *************************************************************************/

#ifndef _MESSAGE_OP_H_
#define _MESSAGE_OP_H_

/* misc messages */
#define MSG_OP_NOP                       51

/* client-server (dis)connection messages */
#define MSG_OP_GET_NODEID 		101
#define MSG_OP_FREE_NODEID              102
#define MSG_OP_FREE_NODEID_COMMIT       103
#define MSG_OP_REGISTER_NODE            104
#define MSG_OP_GET_NODEINFO             105

/* inter-client (dis)connection messages */
#define MSG_OP_CONNECT                  151
#define MSG_OP_DISCONNECT               152
#define MSG_OP_DISCONNECT_COMMIT	153

/* page requests */
#define MSG_OP_GET_PAGE                 201
#define MSG_OP_PAGE_GRANTED             202
#define MSG_OP_PAGE_DENIED              203

/* regions */
#define MSG_OP_CREATE_REGION            251
#define MSG_OP_OPEN_REGION              252
#define MSG_OP_CLOSE_REGION             253
#define MSG_OP_DESTROY_REGION           254

/* locks */
#define MSG_OP_ALLOC_LOCK               301
#define MSG_OP_ACQUIRE_LOCK             302
#define MSG_OP_LOCK_GRANTED             303
#define MSG_OP_LOCK_REQ_QUEUED          304
#define MSG_OP_FREE_LOCK                305

/* barriers */
#define MSG_OP_ALLOC_BARRIER            351
#define MSG_OP_BARRIER_WAIT             352
#define MSG_OP_BARRIER_CROSSED          353
#define MSG_OP_WAIT_FOR_BARRIER         354
#define MSG_OP_FREE_BARRIER             355

/* threads */
#define MSG_OP_CREATE_THREAD            401
#define MSG_OP_THREAD_TERMINATED        402
#define MSG_OP_MARSHALLED_REGIONS       403
#define MSG_OP_MARSHALLED_LOCKS         404

/* misc */
#define MSG_OP_PAGE_DIFF                901
#define MSG_OP_SHUTDOWN                 902

#endif  /* _MESSAGE_OP_H_ */
