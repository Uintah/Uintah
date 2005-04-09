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
 * message.h: defines the Message struct
 *
 *************************************************************************/

#ifndef _MESSAGE_H_
#define _MESSAGE_H_

/* Quarks header files */
#include "config.h"
#include "message_op.h"
#include "port.h"
#include "types.h"
struct Buffer;

/* message family */
#define MSG_FAMILY_SYNCH    1
#define MSG_FAMILY_ASYNCH   2

/* message type */
#define MSG_TYPE_MESSAGE    1
#define MSG_TYPE_REPLY	    2

#define MSG_TMASK 0x00ff
#define MSG_FMASK 0xff00
#define MSG_TBITS 0
#define MSG_FBITS 8

#define msg_family(m)   (((m)->type & MSG_FMASK) >> MSG_FBITS)
#define msg_type(m)     (((m)->type & MSG_TMASK) >> MSG_TBITS)

/* timeout time in milliseconds */
#define TIMEOUT_TIME      1000

#define MSG_DATA_SIZE       PAGE_SIZE+12
typedef struct Message {
    /* First two words are not sent over the wire */
    struct Buffer         *pool_ptr;                       /* 4 */
    struct Message *next;                                  /* 4 */

    /* All the following is sent */
    unsigned short type;   /* encodes family and type */   /* 2 */
    unsigned short op;                                     /* 2 */
    unsigned long  seqno;                                  /* 4 */
    unsigned short numfrags;                               /* 2 */
    unsigned short fragno;                                 /* 2 */
    Id             from_thread;                            /* 4 */
    Id             to_thread;                              /* 4 */
    unsigned long  length;                                 /* 4 */
    /* total number of bytes for the above header:           24 */
    char           data[MSG_DATA_SIZE];                 /* 4104 */
    /* total size of Message, which is sent over ntwk:     4128 */
} Message;

#define HEADER_SIZE         24
/* offset of start of packet to be sent */
#define MSG_PACKET_OFFSET   8
#define MAX_RETRIES         64

#define MSG_SET_FAMILY(m, fam) ((m)->type = ((m)->type & MSG_TMASK) | \
                                 (((fam) & 0xffff) << MSG_FBITS))
#define MSG_SET_TYPE(m, typ)   ((m)->type = ((m)->type & MSG_FMASK) | \
                                 (((typ) & 0xffff) << MSG_TBITS))

/* convenience message operations */
#define FIRST		0
#define SECOND		1
#define THIRD		2
#define FOURTH		3
#define extract_long(data, which)   (*(unsigned long *) ((data) + (which)*4))

extern void     asend_msg(Threadid, Message *);
extern Message *ssend_msg(Threadid, Message *);
extern Message *receive(int);
extern void     reply(Message *msg);


#ifndef IRIX
#define MSG_INIT(m, opp)  ({(m)=ltable[Thread(thread_self())].send_buffer; \
		           (m)->op = (opp); \
			   (m)->length = 0; } )
#define MSG_OP(m, opp)     ((m)->op = (opp))
#define MSG_INSERT(m, d)  ( \
        { *(unsigned long *) ((m)->data + (m)->length) = (unsigned long) (d); \
          (m)->length += 4; } )
#define MSG_INSERT_BLK(m, a, s)  ( { \
        mem_copy((char *) (a), \
                      (char *) ((m)->data + (m)->length), (s)); \
        (m)->length += (s); } )
#define MSG_CLEAR(m)      ((m)->length = 0)
#else
extern void MSG_INIT_I(Message **, unsigned short);
#define MSG_INIT(m, opp) MSG_INIT_I(&m, opp)
extern void MSG_OP(Message *, unsigned short);
extern void MSG_INSERT_I(Message *, unsigned long);
extern void MSG_INSERT_BLK_I(Message *, char *, int);
extern void MSG_CLEAR(Message *);
#define MSG_INSERT(m, d)  MSG_INSERT_I(m, (unsigned long) d)
#define MSG_INSERT_BLK(m, a, s) MSG_INSERT_BLK_I(m, (char *) a, s)
#endif


extern unsigned long timeout_time;


#if 0
done upto here ....

/* message family */
#define MSG_FAMILY_SYNCH 	0x0001
#define MSG_FAMILY_ASYNCH       0x0002
#define MSG_FAMILY_MP           0x0003
#define MSG_FAMILY_ASYNCH_BCAST 0x0004

/* message operation */
#include "message_op.h"

/* message type */

/* return values for messages */
#define INVALIDATED             0x0002
#define COPYSET_UPDATED         0x0003
#define DIFF_ACK                0x0004
#define DIFF_NACK               0x0005
#define DIFF_NACK_XFER_OWN      0x0006


extern Message* Receive(int);
extern void     async_init();
extern unsigned long async_thread(unsigned long arg);

extern int timeoutTime;
#endif /* #if 0 */


#endif  /* _MESSAGE_H_  */








