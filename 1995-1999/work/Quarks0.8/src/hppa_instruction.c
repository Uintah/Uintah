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
 * hppa_instruction.c: PA-RISC specific.
 *
 *************************************************************************/

#include <stdio.h>
#include "instruction.h"

#ifndef hppa
THIS FILE IS ONLY FOR PA_RISC INSTRUCTION SET
#endif

/* This is currently under development */

static unsigned short *load_opcodes = 0;
static unsigned short *store_opcodes = 0;

static void init_opcodes()
{
}

int LOAD_INSTRUCTION(unsigned long inst)
{
    return 0;
}

int STORE_INSTRUCTION(unsigned long inst)
{
    return 1;
}
			      
