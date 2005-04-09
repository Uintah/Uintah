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
 * server.h: shared memory server
 *
 *************************************************************************/

#ifndef _SERVER_H_
#define _SERVER_H_

#include <stdio.h>
#include "types.h"
#include "list.h"
#include "port.h"
#include "thread.h"
#include "message.h"

#define AUTO_ALLOC_START  0xffff0000
#define DSM_SERVER_THREAD 1

#define MAX_NODEID    16

typedef struct Sregion  {
    Id      regionid;
    Size    size;
    Address address;
    Id      creator;
    Boolean destroy;
    List    *clients;   /* List of "Clientid" items */
} Sregion;

typedef struct Nodeinfo {
    Id  nodeid;
    int registered;
} Nodeinfo;

typedef struct Slock {
    Id      lockid;
    Id      creator;
} Slock;

typedef struct Sbarrier {
    Id 	 barrierid;
    Id   creator;
    int  numsynch;      /* no of processes that have reached the barrier  */
    int  num_to_wait_for;  /* "quorum" required at a given border crossing */
    List *waiters; /* list of agents that have requested synchronization */
} Sbarrier;


#endif /* _SERVER_H_ */
