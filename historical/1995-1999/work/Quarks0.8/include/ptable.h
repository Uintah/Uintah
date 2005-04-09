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
 * ptable.h: page table
 *
 *************************************************************************/

#ifndef _PTABLE_H_
#define _PTABLE_H_

#include "types.h"
#include "protocol.h"
#include "thread.h"


/* access modes of a page */
#define PAGE_MODE_RW   1
#define PAGE_MODE_RO   2
#define PAGE_MODE_NO   3

/* action in get_page() */
#define FOR_READ       1
#define FOR_WRITE      2

#define NUSDIFF_MAX  16
#define NUSDIFF_MIN   4

typedef struct Ptentry {
    char  	*addr;     /* page address, used for mprotect/shmdt */
    Id          powner;
    unsigned short access;
    mutex_t     lock;

    unsigned long copyset; /* bit vector */
    unsigned long in_copyset;     /* Used to avoid data race in SendUpdates()*/
    int         sending_updates;  /* and incorporate_diff() */

    int         inval_pending;
    unsigned long lease_expiry_time;
    unsigned long inval_time;
    int         num_usdiffs;
    int         useless_diffs;
    
    int         twin_present;
    char        *twin_addr;
    int         protocol;
    fault_handler_t read_fault_handler;
    fault_handler_t write_fault_handler;
    page_req_handler_t page_request_handler;
} Ptentry;

extern void set_access(Ptentry *, int);
extern void set_protocol(Ptentry *, int);
extern void init_ptentry(Ptentry *, char *, Id, int);

#endif /* _PTABLE_H_ */
