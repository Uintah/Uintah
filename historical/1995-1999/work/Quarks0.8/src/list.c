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
 * list.c: list processing
 *
 *************************************************************************/

#include "list.h"
#include "config.h"

static LBuffer  *lpool=0;         /* a pool of buffers used for listnodes */
static int      lpool_ent;        /* number of buffers in the pool        */
static int      next_lbuffer = 0; /* which buffer to allocate next        */


static void init_lpool()
{
    int      i;
    Listnode *buf;

    /* XXX Must make the pool of buffers expandable.
     * Cannot make it unless we do something about the
     * pointers into individual lpool entries 
     */
    lpool_ent = NUM_LIST_BUFFERS;
    lpool = (LBuffer *) taballoc((Address) &lpool, 
				  (Size) sizeof(LBuffer), 
				 &lpool_ent);
    buf = (Listnode *) malloc(lpool_ent*sizeof(Listnode));
    for (i=0; i<lpool_ent; i++)
    {	
	slock_init(&lpool[i].inuse);
	lpool[i].buf = buf++;
	lpool[i].buf->pool_ptr = &lpool[i];
    }
}

#define free_listnode(b)  (slock_unlock(&(b)->pool_ptr->inuse))


static Listnode *new_listnode()
{
    int i;
    int start = next_lbuffer;
    int found = -1;
    int orig_size;
    Listnode *buf;

    if (!lpool)
	init_lpool();

    while (! slock_try_lock(&lpool[next_lbuffer].inuse))
    {
	next_lbuffer = 
	    (next_lbuffer == (lpool_ent - 1)) ? 0 : (next_lbuffer + 1);
	if (next_lbuffer == start)
	{
	    PANIC("All listnode buffers full");

	    /* all listnode buffers full. Must expand. */
	    /* XXX see comment in init_lpool() */
	    orig_size = lpool_ent;
	    lpool = (LBuffer *) tabexpand((Address) &lpool,
					   &lpool_ent);
	    buf = (Listnode *) malloc(orig_size*sizeof(Listnode));
	    for (i=orig_size; i<lpool_ent; i++)
	    {
		slock_init(&lpool[i].inuse);
		lpool[i].buf = buf++;
		lpool[i].buf->pool_ptr = &lpool[i];
	    }
	    next_lbuffer = orig_size;
	    break;
	}
    }
    found = next_lbuffer;

    next_lbuffer = 
	(next_lbuffer == (lpool_ent - 1)) ? 0 : (next_lbuffer + 1);

    return lpool[found].buf;
}
   
List *new_list()
{
    List *l;

    l = (List *) malloc(sizeof(List));
    l->head = l->current = l->tail = 0;

    return l;
}	


/* The invocee of list_append() and list_rem_head() should ensure
 * that there is no race condition. The solution, which is employed
 * currently in Quarks, is to disable interrupts before calls to 
 * these routines. More lightweight methods are needed.
 */
int list_append(List *list, void *obj)
{
    Listnode *p = new_listnode();
    p->object = obj;
    p->next = 0;

    if (list->head == 0)
    {
	ASSERT(list->tail == 0);
	ASSERT(list->current == 0);
	list->head = list->tail = list->current = p;
    }
    else
    {
	list->tail->next = p;
	list->tail = p;
    }
}

void *list_rem_head(List *l)
{
    void *obj;

    if (l->head == 0)
    {
	ASSERT(l->tail == 0);
	ASSERT(l->current == 0);
	return 0;
    }

    if (l->head == l->tail)
    {
	obj = l->head->object;
	free_listnode(l->head);
	l->head = l->tail = l->current = 0;
	return obj;
    }

    obj = l->head->object;
    if (l->current == l->head) l->current = l->current->next;
    free_listnode(l->head);
    l->head = l->head->next;
    return obj;
}

void list_rem_item(List *l, void *item)
{
    /* This routine might screw up l->current */
    Listnode *ptr, *remnode;

    if (l->head == 0)
    {	
	PANIC("List is empty");
	return;
    }

    if (l->head->object == item)
    {
	list_rem_head(l);
	return;
    }

    ptr = l->head;
    while (ptr->next->object != item)
    {
	ptr = ptr->next;
	if (!ptr || !ptr->next)
	    PANIC("Corrupt list");
    }
    remnode   = ptr->next;
    if (ptr->next == l->tail)
    {
	l->tail = ptr;
	ptr->next = 0;
    }
    else
	ptr->next = ptr->next->next;
    
    free_listnode(remnode);
}

int list_empty(List *l)
{
#ifdef CHECKASSERT
    disable_signals();
    if ((l->head == 0) != (l->tail == 0))
	PANIC("Corrupt list");
    enable_signals();
#endif

    return (l->head == 0);
}


