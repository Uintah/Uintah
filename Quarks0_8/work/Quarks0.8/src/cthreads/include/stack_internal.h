/* 
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
 */

#ifndef STACK_INTERNAL_H
#define STACK_INTERNAL_H

#include "stack.h"
#include "lock.h"

#ifdef hppa
#define STACK_GROWTH_UP
#endif


/*
 * Stack definitions.
 */
#define STACKSIZE	(32*1024)
#define STACKCOUNT	20

#ifdef	STACK_GROWTH_UP
#define STACKMASK	~(STACKSIZE - 1)
#else	/* STACK_GROWTH_UP */
#define STACKMASK	(STACKSIZE - 1)
#endif	/* STACK_GROWTH_UP */

#define roundup(x, y) ((((x) + ((y) - 1)) / (y)) * (y))

#endif /* STACK_INTERNAL_H */
