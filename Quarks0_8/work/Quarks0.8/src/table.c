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
 * table.c: table management
 *
 *************************************************************************/

#include "table.h"
#include "thread.h"
#include "util.h"

static Table    table[MAX_TABLES];
static Boolean  table_initialized = 0;

static void init_table()
{
    int i;

    for (i=0; i<MAX_TABLES; i++)
    {
	table[i].name = 0;
	table[i].item_size = 0;
	table[i].num_allocated = 0;
	table[i].alloc_addr = 0;
	table[i].num_expansion = 0;
    }
    table_initialized = 1;
}

static int find_tab_entry(Address name)
{
    int i;

    for (i=0; i<MAX_TABLES; i++)
	if (table[i].name == name)
	    return i;

    return -1;
}

Address taballoc(Address name, Size size, int *nument)
{
    static int next = 0;
    char *thing;

    if (!table_initialized)
	init_table();
    if (find_tab_entry(name) >= 0)
	mumble("taballoc: entry already exists");

    table[next].name = name;
    table[next].item_size = size;
    table[next].num_allocated = (*nument) ? *nument : DEFAULT_ALLOCATION;
    thing = (char *) malloc(size*table[next].num_allocated);
    table[next].alloc_addr = (Address) thing;
    mem_zero((char *) table[next].alloc_addr, size*table[next].num_allocated);
    next++;

    if (next >= MAX_TABLES)
	PANIC("Table-table full. Increase MAX_TABLES.");

    *nument = table[next-1].num_allocated;
    return table[next-1].alloc_addr;
}

Address tabexpand(Address name, int *nument)
{
    int entry = -1;
    Address  old_addr;
    int      to_alloc;

    if (!table_initialized)
	init_table();
    if ((entry = find_tab_entry(name)) < 0)
    {
	mumble("tabexpand: entry does not exist");
	return 0;
    }

    /* XXX: use realloc */
    to_alloc = table[entry].item_size*table[entry].num_allocated*2;
    old_addr = table[entry].alloc_addr;
    table[entry].num_allocated *= 2;
    table[entry].alloc_addr = (Address) malloc(to_alloc);
    mem_zero((char *) table[entry].alloc_addr, to_alloc);
    mem_copy((char *) old_addr, (char *) table[entry].alloc_addr, to_alloc/2);
    free(old_addr);
    
    *nument = table[entry].num_allocated;
    table[entry].num_expansion++;
    if (table[entry].num_expansion > MAX_EXPANSIONS)
	mumble("table expanded too many times. possible runaway table");

    return table[entry].alloc_addr;
}
    
