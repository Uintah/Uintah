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
 * region.c: region handling at client side.
 *
 *************************************************************************/

#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>


#include "region.h"
#include "config.h"
#include "ptable.h"
#include "message.h"
#include "serv_intf.h"

Region openreg_table[MAX_REGIONS];
static int ortable_init = 0;
int        devzero_fd = -1;
extern int    num_children_forked;
extern int    num_children_terminated;

static void init_orentry(int i)
{
    openreg_table[i].regionid = INVALID_ID;
    openreg_table[i].gva_base = 0;
    openreg_table[i].lva_base = 0;
    openreg_table[i].size     = 0;
    openreg_table[i].numpages = 0;
    openreg_table[i].ptable   = 0;
    openreg_table[i].var_addr = 0;
    if (openreg_table[i].lock == 0)
	openreg_table[i].lock = mutex_alloc();
    openreg_table[i].creator = INVALID_ID;
}

static void init_ortable()
{
    int i;

    for (i=0; i<MAX_REGIONS; i++)
    {
	openreg_table[i].lock = 0;
	init_orentry(i);
    }

    ortable_init = 1;
}

static int new_ortable_index()
{
    int i;

    if (!ortable_init) init_ortable();

    for (i=0; i<MAX_REGIONS; i++)
	if (openreg_table[i].regionid == INVALID_ID)
	    return i;

    return -1;
}

static int ortable_index_reg(Id regionid)
{
    /* Given a regionid, return the table index */
    int i;

    if (!ortable_init) init_ortable();

    /* XXX this must be made more efficient */
    for (i=0; i<MAX_REGIONS; i++)
	if (openreg_table[i].regionid == regionid)
	    return i;

    return -1;
}

static int ortable_index_gva(Address gva_addr)
{
    /* Given a GV address, return the table index */
    int orindex;

    if (!ortable_init) init_ortable();

    /* XXX this could be made more efficient */
    /* find out the region gva_addr belongs to */
    for (orindex=0; orindex < MAX_REGIONS; orindex++)
	if ((gva_addr >= openreg_table[orindex].gva_base) &&
	    (gva_addr < (openreg_table[orindex].gva_base + 
		     openreg_table[orindex].size)))
	    return orindex;

    return -1;
}

static int ortable_index_lva(Address lva_addr)
{
    /* Given a LV address, return the table index */
    int orindex;

    if (!ortable_init) init_ortable();

    /* This could be made more efficient */
    for (orindex=0; orindex < MAX_REGIONS; orindex++)
	if ((lva_addr >= openreg_table[orindex].lva_base) &&
	    (lva_addr < (openreg_table[orindex].lva_base + 
		     openreg_table[orindex].size)))
	    return orindex;

    return -1;
}


Ptentry *get_pte(Id regionid, Address lva_addr)
{
    int     pagenum;
    Ptentry *pte;
    int     orindex;

    if (!ortable_init) init_ortable();
    orindex = ortable_index_reg(regionid);

    pte = openreg_table[orindex].ptable;
    pagenum = (lva_addr - openreg_table[orindex].lva_base)/PAGE_SIZE;
    ASSERT(pagenum < openreg_table[orindex].numpages);

    return &(openreg_table[orindex].ptable[pagenum]);
}

Ptentry *get_pte_locked(Id regionid, Address lva_addr)
{
    Ptentry *pte;
    
    pte = get_pte(regionid, lva_addr);
    mutex_lock(pte->lock);
    return pte;
}

void lock_pte(Ptentry *pte)
{
    mutex_lock(pte->lock);
}

void unlock_pte(Ptentry *pte)
{
    mutex_unlock(pte->lock);
}


Id gaddr2region(Address gva_addr)
{
    int orindex;

    if ((orindex = ortable_index_gva(gva_addr)) == -1)
	PANIC("Unknown region");
    
    return openreg_table[orindex].regionid;
}

Id laddr2region(Address lva_addr)
{
    int orindex;

    if ((orindex = ortable_index_lva(lva_addr)) == -1)
	return INVALID_ID;

    return openreg_table[orindex].regionid;
}

Address gva2lva(Address gva_addr)
{
    int orindex;

    orindex = ortable_index_gva(gva_addr);
    return (gva_addr - openreg_table[orindex].gva_base +
	    openreg_table[orindex].lva_base);
}

Address lva2gva(Address lva_addr)
{
    int orindex;

    orindex = ortable_index_lva(lva_addr);
    return (lva_addr - openreg_table[orindex].lva_base +
	    openreg_table[orindex].gva_base);
}

static int region_exists(Id regionid)
{
    int orindex;

    if (regionid == INVALID_ID) return 0;

    if ((orindex = ortable_index_reg(regionid)) == -1)
	return 0;

    return 1;
}

static Address map_region(int orindex, Id owner, int access)
{
    /* openreg_table has the appropriate entry for the region.
     * just mmap the region and return the local address 
     */
    char    *addr, *base_addr;
    int     i;
    Ptentry *pte;
    int     prot;

#ifndef HPUX
    if (devzero_fd == -1)
	if ((devzero_fd = open("/dev/zero", O_RDWR, 0)) < 0)
	    PANIC("Cannot open /dev/zero");
#endif
    
    if (access == PAGE_MODE_RW) prot = PROT_READ|PROT_WRITE;
    if (access == PAGE_MODE_RO) prot = PROT_READ;
    if (access == PAGE_MODE_NO) prot = PROT_NONE;
#ifdef HPUX
    base_addr = mmap(0, openreg_table[orindex].size, prot,
		MAP_PRIVATE | MAP_VARIABLE | MAP_ANONYMOUS, -1, 0);
#else
    base_addr = mmap(0, openreg_table[orindex].size, prot,
		MAP_PRIVATE, devzero_fd, 0);
    /* DBGCODE */
    mprotect(base_addr, openreg_table[orindex].size, prot);
#endif
    if ((int)base_addr == -1)
    {
	perror("mmap");
	PANIC("mmap failed");
    }
    openreg_table[orindex].lva_base = (Address) base_addr;
    openreg_table[orindex].ptable = (Ptentry *) 
	malloc(sizeof(Ptentry)*openreg_table[orindex].numpages);

    pte = &openreg_table[orindex].ptable[0];
    addr = base_addr;
    for (i=0; i<openreg_table[orindex].numpages; i++)
    {
	init_ptentry(pte, addr, owner, access);
	pte++;
	addr += PAGE_SIZE;
    }

    i = *(int *) base_addr;

    return (Address) base_addr;
}

void Qk_create_region(Id regionid, Size size, unsigned long *var_addr)
{
    /* Returns the lva_base of the region. */
    Message *msg, *reply;
    Id      alocid;
    Size    alocsize;
    Address alocaddr;
    int     orindex;

    if (region_exists(regionid))
    {
	mumble("region already exists");
	*var_addr = 0;
	return;
    }
    MSG_INIT(msg, MSG_OP_CREATE_REGION);
    MSG_INSERT(msg, regionid);
    MSG_INSERT(msg, size);
    reply = ssend_msg(DSM_SERVER_THREAD, msg);
    switch (extract_long(reply->data, FIRST))
    { 
    case REG_ALLOCATED:
	orindex  = new_ortable_index();
	alocid   = (Id)      extract_long(reply->data, SECOND);
	alocsize = (Size)    extract_long(reply->data, THIRD);
	alocaddr = (Address) extract_long(reply->data, FOURTH);
	openreg_table[orindex].regionid = alocid;
	openreg_table[orindex].gva_base = alocaddr;
	openreg_table[orindex].size     = alocsize;
	openreg_table[orindex].numpages = alocsize/PAGE_SIZE;
	openreg_table[orindex].creator  = 1;
	openreg_table[orindex].var_addr = var_addr;
	free_buffer(reply);
	/* ptable and lva_base are set by map_region, 
	 * lock is already initialized. 
	 */
	printf("Create_region: id = 0x%x\n", alocid);
	*var_addr = map_region(orindex, Qknodeid, PAGE_MODE_RW);
	break;
    case REG_EXISTS:
	mumble("region already exists");
	*var_addr = 0;
	return;
    case REG_NO_MEM:
	mumble("region cannot be allocated. No memory");
	*var_addr = 0;
	return;
    default:
	PANIC("Invalid return code");
    }
}

Address Qk_open_region(Id regionid)
{
    Message *msg, *reply;
    Id      owner;
    int     orindex;

    if (region_exists(regionid))
    {
	mumble("region already opened");
	return openreg_table[regionid].lva_base;
    }
    MSG_INIT(msg, MSG_OP_OPEN_REGION);
    MSG_INSERT(msg, regionid);
    reply = ssend_msg(DSM_SERVER_THREAD, msg);
    switch (extract_long(reply->data, FIRST))
    {
    case REG_OPENED:
	orindex = new_ortable_index();
	openreg_table[orindex].regionid = regionid;
	openreg_table[orindex].gva_base = (Address)
	    extract_long(reply->data, SECOND);
	openreg_table[orindex].size = (Size)
	    extract_long(reply->data, THIRD);
	owner = extract_long(reply->data, FOURTH);
	openreg_table[orindex].numpages = 
	    openreg_table[orindex].size / PAGE_SIZE;
	openreg_table[orindex].creator  = 0;
	free_buffer(reply);
	printf("region 0x%x opened, the owner is %d\n", regionid, owner);
	return map_region(orindex, owner, PAGE_MODE_NO);
    case REG_NOT_CREATED:
	mumble("region should be created to be open");
	return 0;
    case REG_DESTROYED:
	mumble("region being destroyed");
	return 0;
    default:
	PANIC("Invalid return code");
    }
}

void Qk_close_region(Id regionid)
{
    Message *msg, *reply;
    int orindex;

    if (!region_exists(regionid))
    {
	mumble("Qk_close_region: region not open");
	return;
    }
    orindex = ortable_index_reg(regionid);
    free(openreg_table[orindex].ptable);
    init_orentry(orindex);

    MSG_INIT(msg, MSG_OP_CLOSE_REGION);
    MSG_INSERT(msg, regionid);
    reply = ssend_msg(DSM_SERVER_THREAD, msg);
    switch (extract_long(reply->data, FIRST))
    {
    case REG_CLOSED:
	free_buffer(reply);
	break;
    case REG_NOT_CREATED:
	/* the region does not exist at the server. 
	 * This is an error condition.
	 */
	free_buffer(reply);
	PANIC("Server does not know of this region");
    default:
	free_buffer(reply);
	PANIC("Invalid return code");
    }

}

void Qk_destroy_region(Id regionid)
{
    Message *msg, *reply;
    if (!ortable_init) init_ortable();

    /* The region may not exist in an open mode locally!!! */
    MSG_INIT(msg, MSG_OP_DESTROY_REGION);
    MSG_INSERT(msg, regionid);
    reply = ssend_msg(DSM_SERVER_THREAD, msg);
    switch(extract_long(reply->data, FIRST))
    {
    case REG_NOT_CREATED:
	mumble("Qk_destroy_region: region does not exist at server");
	free_buffer(reply);
	break;
    case REG_DESTROYED:
	free_buffer(reply);
	break;
    default:
	free_buffer(reply);
	PANIC("Invalid return code");
    }
}


void close_all_regions()
{
    int i;

    for (i=0; i<MAX_REGIONS; i++)
	if (openreg_table[i].regionid != INVALID_ID)
	{
	    if (openreg_table[i].creator == Qknodeid)
		Qk_destroy_region(openreg_table[i].regionid);
	    else
		Qk_close_region(openreg_table[i].regionid);
	}
}

#define NUMREG_PERMSG ((MSG_DATA_SIZE-4-8)/8)

static void marshall_regions(Threadid th)
{
    Message *msg, *reply;
    unsigned long *numreg, *data;
    int i;

    MSG_INIT(msg, MSG_OP_MARSHALLED_REGIONS);
    numreg = (unsigned long *) msg->data;
    data   = (unsigned long *) (msg->data + 4);
    *numreg = 0;
    for (i=0; i<MAX_REGIONS; i++)
	if ((openreg_table[i].regionid != INVALID_ID) &&
	    (openreg_table[i].creator == Qknodeid))
	{
	    *data++ = openreg_table[i].regionid;
	    *data++ = (unsigned long) openreg_table[i].var_addr;
	    (*numreg)++;
	    if (*numreg >= NUMREG_PERMSG)
	    {
		msg->length = 4 + (*numreg)*8;

		reply = ssend_msg(th, msg);
		free_buffer(reply);
		*numreg = 0;
		data = (unsigned long *) (msg->data + 4);
	    }
	}
    msg->length = 4 + (*numreg)*8;
    reply = ssend_msg(th, msg);
    free_buffer(reply);
}

void unmarshall_regions()
{
    unsigned long *data;
    Id regionid;
    unsigned long *var_addr;
    int done  = 0;
    int i, numreg;
    Message *msg;

    while (!done)
    {
	msg = receive(0);
	ASSERT(msg->op == MSG_OP_MARSHALLED_REGIONS);
	numreg = *(int *) msg->data;
	if (numreg < NUMREG_PERMSG)
	    done = 1;
	data = (unsigned long *) (msg->data+4);
	for (i=0; i<numreg; i++)
	{
	    regionid = (Id) *data++;
	    var_addr = (unsigned long *) *data++;
	    *var_addr = Qk_open_region(regionid);
	}
	MSG_CLEAR(msg);
	reply_msg(msg);
    }
}

int Qk_fork(void* func)
{
    static Id which_proc = 2;

    Message *request, *reply;
    Threadid th;

    MSG_INIT(request, MSG_OP_CREATE_THREAD);
    MSG_INSERT(request, func);
    ASSERT(Qknodeid == 1); /* Only master can create remote threads */
    
    th = construct_threadid(which_proc, DSM_THREAD);
    which_proc++;

    reply = ssend_msg(th, request);
    free_buffer(reply);

    marshall_regions(th);
    marshall_locks(th);

    num_children_forked++;
    return which_proc-1;
}

