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
 * protocol.h: consistency protocols
 *
 *************************************************************************/

#ifndef _PROTOCOL_H_
#define _PROTOCOL_H_

#include "types.h"
#include "list.h"

#define PROT_WRITE_SHARED       1
#define PROT_WRITE_INVALIDATE   2
#define DEFAULT_PROTOCOL        PROT_WRITE_SHARED

#define DIFF_ACK           1
#define DIFF_NACK          2
#define DIFF_NACK_XFER_OWN 3

#define NUM_USDIFFS_MIN    8
#define add_copyset(cp, id)  ((cp) | (1 << (id)))

struct Ptentry;

typedef void (*fault_handler_t)(Address, Address, struct Ptentry *);
typedef void (*page_req_handler_t)(Address, Address, struct Ptentry *,
				   Threadid, Id, int);


extern void read_fault_handler_ws(Address, Address, struct Ptentry *);
extern void write_fault_handler_ws(Address, Address, struct Ptentry *);
extern void page_request_handler_ws(Address, Address, struct Ptentry *,
				    Threadid, Id, int);
extern void read_fault_handler_wi(Address, Address, struct Ptentry *);
extern void write_fault_handler_wi(Address, Address, struct Ptentry *);
extern void page_request_handler_wi(Address, Address, struct Ptentry *,
				    Threadid, Id, int);


extern List *DUQ;
#endif  /* _PROTOCOL_H_ */
