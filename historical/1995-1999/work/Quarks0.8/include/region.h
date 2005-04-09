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
 * region.h: shared memory regions
 *
 *************************************************************************/

#ifndef _REGION_H_
#define _REGION_H_

#include "ptable.h"
#include "thread.h"

#define MAX_REGIONS 1024

typedef struct Region {
    Id                  regionid; /* the region id */
    Address             gva_base; /* global base virtual address */
    Address		lva_base; /* local base virtual address */
    Address             size;     /* size of the region          */
    int                 numpages; /* number of entries in ptable */
    Ptentry		*ptable;  /* list of pagetable entries */
    unsigned long       *var_addr; /* used for remote thread fork */
    mutex_t 		lock;     /* to lock the region table entry */
    Id                  creator;  /* 1 if self is the creator */
} Region;

extern Region openreg_table[MAX_REGIONS];

extern Ptentry *get_pte(Id, Address);
extern Ptentry *get_pte_locked(Id, Address);
extern void    lock_pte(Ptentry *);
extern void    unlock_pte(Ptentry *);
extern Id      gaddr2region(Address);
extern Id      laddr2region(Address);
extern Address gva2lva(Address);
extern Address lva2gva(Address);
extern Address create_region_req(Id, Size, unsigned long *);
extern Address open_region_req(Id);
extern void    close_region_req(Id);
extern void    destroy_region_req(Id);
extern void    unmarshall_regions();

#endif  /* _REGION_H_ */
