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
 * buffer.c: implements buffer management for message sub-system
 *
 *************************************************************************/

#include <stdlib.h>
#include "buffer.h"
#include "config.h"

static Buffer   *bpool=0;         /* Pool of pre-allocated buffers */
static int      bpool_ent;        /* how many are allocated        */
static int      next_buffer = 0;  /* next buffer to choose from    */

static void init_bpool()
{
    int      i;
    Message  *buf;
    Message  *msg;

    /* XXX Must make the pool of buffers expandable.
     * Cannot make it unless we do something about the
     * pointers into individual bpool entries 
     */
    bpool_ent = NUM_MSG_BUFFERS;
    bpool = (Buffer *) taballoc((Address) &bpool,
				(Size) sizeof(Buffer),
				&bpool_ent);

    buf = (Message *) malloc(bpool_ent*sizeof(Message));
    for (i=0; i<bpool_ent; i++)
    {	
	slock_init(&bpool[i].inuse);
	msg = bpool[i].buf = buf++;
	msg->next   = 0;
	msg->pool_ptr = &bpool[i];
    }
}

Message *new_buffer()
{
    int i;
    int start = next_buffer;
    int found = -1;
    int orig_size;
    Message *buf;

    if (!bpool)
	init_bpool();

    while (! slock_try_lock(&bpool[next_buffer].inuse))
    {
	next_buffer = 
	    (next_buffer == (bpool_ent - 1)) ? 0 : (next_buffer + 1);
	if (next_buffer == start)
	{
	    PANIC("All buffers full");
	    /* all buffers are full. Must expand. */
	    /* XXX see comment in init_bpool() */
	    orig_size = bpool_ent;
	    bpool = (Buffer *) tabexpand((Address) &bpool,
					 &bpool_ent);
	    buf = (Message *) malloc(orig_size*sizeof(Message));
	    for (i=orig_size; i<bpool_ent; i++)
	    {
		slock_init(&bpool[i].inuse);
		bpool[i].buf = buf++;
		bpool[i].buf->pool_ptr = &bpool[i];
	    }
	    next_buffer = orig_size;
	    break;
	}
    }
    found = next_buffer;
    next_buffer = 
	(next_buffer == (bpool_ent - 1)) ? 0 : (next_buffer + 1);

    return bpool[found].buf;
}

void free_buffer(Message *b)
{
    slock_unlock(&b->pool_ptr->inuse);
}
















