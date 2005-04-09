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
 * config.h: configuration parameters
 *
 *************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

#define PAGE_SIZE 4096
#define PAGE_MASK (~(PAGE_SIZE - 1))

#define PAGE_ALIGN(m)  ((m) & PAGE_MASK)

#define MAX_LOCKS    8192
#define MAX_BARRIERS    8

/* XXX should get rid of these hard limits someday.
 * They are table sizes. The table entries have pointers into 
 * them from outside, so expanding the table is not really
 * that easy.
 */
#define NUM_LIST_BUFFERS    128
#define NUM_MSG_BUFFERS      64


#endif  /* _CONFIG_H_ */
