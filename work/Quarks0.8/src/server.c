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
 * server.c: the shared memory server
 *
 *************************************************************************/

#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>

/* Quarks header files */
#include "server.h"
#include "connection.h"
#include "buffer.h"
#include "util.h"
#include "serv_intf.h"

static Sregion *regtable;    /* region table maintained by the server */
static int    regtable_ent;
static Id     auto_alloc_id = AUTO_ALLOC_START;

/* XXX: nid_table should be expandable table */
static Nodeinfo     nid_table[MAX_NODEID+1];
static int          nid_t_init=0;

/* for allocated locks and barriers */
static Slock        lid_table[MAX_LOCKS];
static Sbarrier     bid_table[MAX_BARRIERS];

void quarks_dsm_init()
{
    PANIC("This routine is just a place holder. It should not be called");
}

void quarks_dsm_shutdown()
{	
    PANIC("This routine is just a place holder. It should not be called");
}

static void init_nidtable()
{
    int id;

    for (id=0; id<=MAX_NODEID; id++)
    {
	nid_table[id].nodeid = INVALID_ID;
	nid_table[id].registered = 0;
    }
    nid_table[0].nodeid = 0;     /* the server */
    nid_table[0].registered = 1; /* irrelevant */
    nid_t_init = 1;
}

static void init_lidtable()
{
    int i;

    for (i=0; i<MAX_LOCKS; i++) 
    {
	lid_table[i].lockid  = INVALID_ID;
	lid_table[i].creator = INVALID_ID;
    }
}

static void init_bidtable()
{
    int i;

    for (i=0; i<MAX_BARRIERS; i++)
    {
	bid_table[i].barrierid = INVALID_ID;
	bid_table[i].numsynch = 0;
	bid_table[i].num_to_wait_for = 0;
	bid_table[i].waiters = 0;
    }
}

static void alloc_nodeid(Message *msg)
{
    /* msg is a request for allocation of a new nodeid. Search the
     * nid_table and find a new id to allocate. Also reply to the 
     * message. 
     * NOTE: we are inside asyncio_handler, so make it fast.
     */
    int  id;
    
    if (! nid_t_init)
	init_nidtable();

    for (id=1; id<=MAX_NODEID; id++)
	if (nid_table[id].nodeid == INVALID_ID)
	{
	    nid_table[id].nodeid = id;
	    nid_table[id].registered = 0;
	    break;
	}
    if (id > MAX_NODEID)
	for (id=1; id<=MAX_NODEID; id++)
	    if ((nid_table[id].nodeid != INVALID_ID) &&
		disconnected(nid_table[id].nodeid))
	    {
		disconnect_node(nid_table[id].nodeid);
		nid_table[id].nodeid = id;
		nid_table[id].registered = 0;
		printf("recycling nodeid %d\n", nid_table[id].nodeid);
		break;
	    }

    accept_node(nid_table[id].nodeid, (Port *) msg->data);
    msg->length = 4;
    *(unsigned long *) msg->data = (unsigned long) nid_table[id].nodeid;
    asend_msg(construct_threadid(nid_table[id].nodeid, 0), msg);
    free_buffer(msg);
}

static void register_node(Message *msg)
{
    /* The sender client is ready to accept messages. Register it.
     */

    Id nid = Node(msg->from_thread);
    nid_table[nid].registered = 1;
    MSG_CLEAR(msg);
    MSG_INSERT(msg, 0);
    reply_msg(msg);
}


static void dealloc_nodeid_mark(Message *msg)
{
    int i;
    Id nid = Node(msg->from_thread);

    if (nid_table[nid].nodeid != nid)
	PANIC("invalid nodeid in dealloc_nodeid_mark");

    MSG_CLEAR(msg);
    MSG_INSERT(msg, 0);
    reply_msg(msg);
	
    disconnect_node_mark(nid_table[nid].nodeid);
}

static void dealloc_nodeid(Message *msg)
{
    int i;
    Id nid = Node(msg->from_thread);

    if (nid_table[nid].nodeid != nid)
	PANIC("invalid nodeid in dealloc_nodeid");

    free_buffer(msg);
	
    disconnect_node(nid_table[nid].nodeid);
    nid_table[nid].nodeid = INVALID_ID;
    nid_table[nid].registered = 0;
}

static void give_nodeinfo(Message *msg)
{
    Id nid = extract_long(msg->data, FIRST);
    int ct_index = contable_index(nid);
    MSG_CLEAR(msg);
    if (ct_index == -1)  /* server does not know about this node */
    {
	MSG_INSERT(msg,1);
    }
    else
    {
	if (nid_table[nid].registered == 0)  /* not yet registered */
	{
	    MSG_INSERT(msg, 1);
	}
	else	
	{
	    MSG_INSERT(msg,0);
	    MSG_INSERT_BLK(msg, contable_port(ct_index), sizeof(Port));
	    fprintf(stderr, "sin=%x\n", contable_port(ct_index)->sin.sin_addr.s_addr);
	}
    }
    reply_msg(msg);
}


static void init_regtable()
{	
    int i;

    regtable = 0;
    regtable = (Sregion *) taballoc((Address) &regtable,(Size) sizeof(Sregion),
				   &regtable_ent);
    if (! regtable)
	PANIC("Can not allocate region table");
    
    for (i=0; i<regtable_ent; i++)
	regtable[i].regionid = INVALID_ID;
}

static void clean_regtable_entry(rt_index)
{
    free (regtable[rt_index].clients);
    regtable[rt_index].regionid = INVALID_ID;
}


static int regtable_index(Id regionid)
{
    int i;

    for (i=0; i<regtable_ent; i++)
    {
	if (regtable[i].regionid == regionid) 
	    return i;
    }
    return -1;
}

static int free_rtentry()
{
    int i,j;

    for (i=0; i<regtable_ent; i++)
	if (regtable[i].regionid == INVALID_ID)
	    return i;
    
    j = regtable_ent;
    /* region table is full. must expand */
    regtable = (Sregion *) tabexpand((Address) &regtable, &regtable_ent);

    for (i=j; i<regtable_ent; i++)
	regtable[i].regionid = INVALID_ID;

    return j;
}

static Address asalloc(Size size)
{
    /* Address space allocator. Currently is just allocates requests in a 
     * linear fashion. Hence we don't need a sophisticated asfree().
     */
    static Address base = 0xffffffff;
    static Address srvpage;
    Address addr;
    int fd;

    if (base == 0xffffffff)
    {
#ifdef HPUX
	base = (Address) mmap(0, PAGE_SIZE, PROT_READ|PROT_WRITE, 
		    MAP_PRIVATE|MAP_VARIABLE|MAP_ANONYMOUS, -1, 0);
#else
	if ((fd = open("/dev/zero", O_RDWR, 0)) < 0)
	    PANIC("Could not open /dev/zero");
	base = (Address) mmap(0, PAGE_SIZE, PROT_READ|PROT_WRITE, 
		    MAP_PRIVATE, fd, 0);
#endif
	if ((int) base == -1)
	{	
	    perror("mmap");
	    PANIC("mmap failed");
	}
	srvpage = base;
	base += PAGE_SIZE;
    }

    addr = base;
    base += size;
    return addr;
}

static void asfree(Address addr)
{
    /* XXX not implemented yet. Will be needed when we have a more 
     * elaborate address space allocator-deallocator. Right now
     * we just hand out address space chunks linearly. But for the
     * server to be a "true" server, something cleverer has to be 
     * done.
     */
}

static void create_reg(Message *msg)
{
    /* request to allocate a region from a client. regionid can have
     * these values:
     * 0 or INVALID_ID:    the client does not care which id the server 
     *                     allocates. The server must communicate the
     *                     allocated id back the client. Useful for SPLASH
     *                     programs.
     * < AUTO_ALLOC_START: the client requests that the region be
     *                     given this particular id. Useful for 
     *                     the data-sharing-between-application
     *                     paradigm.
     */
    Id   regionid;
    Size size;
    int  rt_index;

    regionid = (Id)   extract_long(msg->data, FIRST);
    size     = (Size) extract_long(msg->data, SECOND);

    size = (size%PAGE_SIZE) ? (size+PAGE_SIZE-(size%PAGE_SIZE)) : size;
    ASSERT(!(size%PAGE_SIZE));
    
    if ((regionid >= AUTO_ALLOC_START) && 
	(regionid != INVALID_ID))
	PANIC("Invalid regionid in create_reg");

    if ((regionid == INVALID_ID) || (regionid == 0))
	regionid = auto_alloc_id++;

    if (regtable_index(regionid) != -1)  /* region already exists */
    {
	/* An autoalloc region cannot already exist, so there will 
	 * never be a need to revert the increment of auto_alloc_id.
	 */
	MSG_CLEAR(msg);
	MSG_INSERT(msg, REG_EXISTS);
	reply_msg(msg);
	return;
    }
    rt_index = free_rtentry();
    if ((regtable[rt_index].address  = asalloc(size)) == 0xffffffff)
    {
	MSG_CLEAR(msg);
	MSG_INSERT(msg, REG_NO_MEM);
	reply_msg(msg);
	return;
    }

    /* region can infact be allocated */
    regtable[rt_index].regionid = regionid;
    regtable[rt_index].size     = size;
    
    regtable[rt_index].destroy = 0;
    regtable[rt_index].creator = Node(msg->from_thread);
    regtable[rt_index].clients = new_list();
    list_append(regtable[rt_index].clients,
		(void *) Node(msg->from_thread));

    MSG_CLEAR(msg);
    MSG_INSERT(msg, REG_ALLOCATED);
    MSG_INSERT(msg, regionid);
    MSG_INSERT(msg, size);
    MSG_INSERT(msg, regtable[rt_index].address);
    reply_msg(msg);
    
    printf("region allocated at 0x%x\n", regtable[rt_index].address);
}

static void open_reg(Message *msg)
{
    Id  regionid;
    int rt_index;

    regionid = (Id) extract_long(msg->data, FIRST);
    if ((rt_index = regtable_index(regionid)) == -1)
    {
	/* Region was not created */
	MSG_CLEAR(msg);
	MSG_INSERT(msg, REG_NOT_CREATED);
	reply_msg(msg);
	return;
    }
    if (regtable[rt_index].destroy)  
    {
	/* region is to be destroyed */
	MSG_CLEAR(msg);
	MSG_INSERT(msg, REG_DESTROYED);
	reply_msg(msg);
	return;
    }
    list_append(regtable[rt_index].clients, 
		(void *) Node(msg->from_thread));
    MSG_CLEAR(msg);
    MSG_INSERT(msg, REG_OPENED);
    MSG_INSERT(msg, regtable[rt_index].address);
    MSG_INSERT(msg, regtable[rt_index].size);
    MSG_INSERT(msg, regtable[rt_index].creator);
    reply_msg(msg);
}

static void close_reg(Message *msg)
{
    Id  regionid;
    int rt_index;

    regionid = (Id) extract_long(msg->data, FIRST);
    if ((rt_index = regtable_index(regionid)) == -1)
    {
	/* The region does not exist */
	MSG_CLEAR(msg);
	MSG_INSERT(msg, REG_NOT_CREATED);
	reply_msg(msg);
	return;
    }
    
    list_rem_item(regtable[rt_index].clients, 
		  (void *) Node(msg->from_thread));
    if (list_empty(regtable[rt_index].clients) && 
	regtable[rt_index].destroy)
	clean_regtable_entry(rt_index);

    MSG_CLEAR(msg);
    MSG_INSERT(msg, REG_CLOSED);
    reply_msg(msg);
}

static void destroy_reg(Message *msg)
{
    Id  regionid;
    int rt_index;
    
    regionid = (Id) extract_long(msg->data, FIRST);
    if ((rt_index = regtable_index(regionid)) == -1)
    {
	/* The region does not exist */
	MSG_CLEAR(msg);
	MSG_INSERT(msg, REG_NOT_CREATED);
	reply_msg(msg);
	return;
    }
    asfree(regtable[rt_index].address);
    list_rem_item(regtable[rt_index].clients, 
		  (void *) Node(msg->from_thread));
    if (!list_empty(regtable[rt_index].clients))
	regtable[rt_index].destroy = 1;
    else
	clean_regtable_entry(rt_index);

    MSG_CLEAR(msg);
    MSG_INSERT(msg, REG_DESTROYED);
    reply_msg(msg);
}


static void alloc_lock(Message *msg)
{
    int i;

    for (i=0; i<MAX_LOCKS; i++)
	if (lid_table[i].lockid == INVALID_ID)
	    break;

    if (i >= MAX_LOCKS)
	PANIC("Lock table full. Increase MAX_LOCKS");

    lid_table[i].lockid = i;
    lid_table[i].creator = Node(msg->from_thread);

    MSG_CLEAR(msg);
    MSG_INSERT(msg, lid_table[i].lockid);
    reply_msg(msg);
}

static void alloc_barrier(Message *msg)
{
    int i;

    for (i=0; i<MAX_BARRIERS; i++)
	if (bid_table[i].barrierid == INVALID_ID)
	    break;

    if (i >= MAX_BARRIERS)
	PANIC("Barrier table full. Increase MAX_BARRIERS");
    
    bid_table[i].barrierid = i;
    bid_table[i].creator = Node(msg->from_thread);
    bid_table[i].waiters = new_list();
    MSG_CLEAR(msg);
    MSG_INSERT(msg, bid_table[i].barrierid);
    reply_msg(msg);
}

static int bid_table_index(Id ba_id)
{
    return (int) ba_id;
}

static void wait_barrier(Message *msg)
{
    Id  ba_id;
    int num_crossers;
    Message *buf, *reply;
    Threadid th;
    int baindex;

    ba_id = (Id) extract_long(msg->data, FIRST);
    num_crossers = (int) extract_long(msg->data, SECOND);
    baindex = bid_table_index(ba_id);
    if (bid_table[baindex].num_to_wait_for == 0)
    {
	/* First arrival */
	bid_table[baindex].num_to_wait_for = num_crossers;
    }
    bid_table[baindex].numsynch++;
    if (bid_table[baindex].numsynch >= bid_table[baindex].num_to_wait_for)
    {
	/* everyone has synchronized */
	MSG_CLEAR(msg);
	MSG_INSERT(msg, MSG_OP_BARRIER_CROSSED);
	reply_msg(msg);

	MSG_INIT(buf, MSG_OP_BARRIER_CROSSED);
	MSG_INSERT(buf, ba_id);
	while (! list_empty(bid_table[baindex].waiters))
	{
	    th = (Threadid) list_rem_head(bid_table[baindex].waiters);
	    reply = ssend_msg(th, buf);
	    free_buffer(reply);
	}
	bid_table[baindex].numsynch=0;
	bid_table[baindex].num_to_wait_for=0;
    }
    else
    {
	list_append(bid_table[baindex].waiters, (void *) msg->from_thread);
	MSG_CLEAR(msg);
	MSG_INSERT(msg, MSG_OP_WAIT_FOR_BARRIER);
	reply_msg(msg);
    }
}

static void free_lock(Message *msg)
{
    Id lockid;
    int i;

    lockid = (Id) extract_long(msg->data, FIRST);
    for (i=0; i<MAX_LOCKS; i++)
	if (lid_table[i].lockid == lockid)
	{	
	    lid_table[i].lockid = INVALID_ID;
	    break;
	}
    MSG_CLEAR(msg);
    reply_msg(msg);
}

static void free_barrier(Message *msg)
{
    Id barrierid;
    int i;

    barrierid = (Id) extract_long(msg->data, FIRST);
    for (i=0; i<MAX_BARRIERS; i++)
	if (bid_table[i].barrierid == barrierid)
	{
	    bid_table[i].barrierid = INVALID_ID;
	    ASSERT(list_empty(bid_table[i].waiters));
	    free(bid_table[i].waiters);
	    bid_table[i].waiters = 0;
	    break;
	}
    MSG_CLEAR(msg);
    reply_msg(msg);
}


static void welcome()
{
    printf("\n\n");
    printf("*************************************\n");
    printf("*            Quarks 0.8             *\n");
    printf("*************************************\n");
    printf("\n\n");
}

Any dsm_server(Any arg)
{
    Message *request;

    init_nidtable();
    init_lidtable();
    init_bidtable();
    init_regtable();
    
    while (1)
    {
	request = receive(0);
	switch (request->op)
	{
	    /* nodeid management */
	case MSG_OP_GET_NODEID:
	    printf("Request for allocating nodeid\n");
	    alloc_nodeid(request);  /* also replies to the message */
	    break;
	case MSG_OP_FREE_NODEID:
	    printf("Request to deallocate nodeid %d\n", 
		   Node(request->from_thread));
	    dealloc_nodeid_mark(request); /* also replies to message */
	    break;
	case MSG_OP_FREE_NODEID_COMMIT:
	    dealloc_nodeid(request); /* also frees the message */
	    break;
	case MSG_OP_REGISTER_NODE:
	    register_node(request);  /* also replies to message */
	    break;
	case MSG_OP_GET_NODEINFO:
	    give_nodeinfo(request);  /* also replies to message */
	    break;

	    /* region management */
	case MSG_OP_CREATE_REGION:
	    create_reg(request);     /* also replies to message */
	    break;
	case MSG_OP_OPEN_REGION:
	    open_reg(request);	     /* also replies to message */
	    break;
	case MSG_OP_CLOSE_REGION:
	    close_reg(request);      /* also replies to message */
	    break;
	case MSG_OP_DESTROY_REGION:
	    destroy_reg(request);    /* also replies to message */
	    break;

	    /* Synchronization primitives */
	case MSG_OP_ALLOC_LOCK:
	    alloc_lock(request);     /* will reply to message */
	    break;
	case MSG_OP_ALLOC_BARRIER:   
	    alloc_barrier(request);  /* will reply to message */
	    break;
	case MSG_OP_BARRIER_WAIT:
	    wait_barrier(request);   /* will reply to message */
	    break;
	case MSG_OP_FREE_LOCK:
	    free_lock(request);
	    break;
	case MSG_OP_FREE_BARRIER:
	    free_barrier(request);
	    break;
	default:
	    PANIC("Unknown request");
	}
    }
    
    return (Any) 0;
}

char myhostname[256];

extern void idlethread();
main()
{
    sprintf(myhostname, "taz-fast.cs.utah.edu");
    welcome();
    quarks_basic_init(1);
    quarks_thread_fork(dsm_server, 0);  
}

