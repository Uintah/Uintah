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
 * sparc_instruction.c: sparc specific.
 *
 *************************************************************************/

#include <stdio.h>
#include "instruction.h"

#ifndef sparc
THIS FILE IS ONLY FOR SPARC INSTRUCTION SET
#endif

static unsigned short *load_opcodes = 0;
static unsigned short *store_opcodes = 0;

static void init_opcodes()
{
    int i,j,factor,sum;

    load_opcodes  = (unsigned short *) 
	malloc(sizeof(unsigned short)*num_load_opcodes);
    store_opcodes = (unsigned short *) 
	malloc(sizeof(unsigned short)*num_store_opcodes);

    for (i=0; i<num_load_opcodes; i++)
    {	
	factor = 1;
	sum = 0;
	for (j = OP_WIDTH-1; j >= 0; j--)
	{
	    sum += factor * ((ld_op_bits[i][j] == '1') ? 1 : 0);
	    factor = factor << 1;
	}
	load_opcodes[i] = sum;
    }
    for (i=0; i<num_store_opcodes; i++)
    {	
	factor = 1;
	sum = 0;
	for (j = OP_WIDTH-1; j >= 0; j--)
	{
	    sum += factor * ((st_op_bits[i][j] == '1') ? 1 : 0);
	    factor = factor << 1;
	}
	store_opcodes[i] = sum;
    }
}

int LOAD_INSTRUCTION(unsigned long inst)
{
    int i;

    unsigned long opcode =  ((inst & op1_mask) >> op1_shift) |
	                    ((inst & op2_mask) >> op2_shift);
    
    if (! load_opcodes) init_opcodes();

    for (i=0; i<num_load_opcodes; i++)
    {
	if (opcode == load_opcodes[i])
	{
	    return 1;
	}
    }
    return 0;
}

int STORE_INSTRUCTION(unsigned long inst)
{
    int i;

    unsigned short opcode = (unsigned short) 
	                    ((inst & op1_mask) >> op1_shift) |
	                    ((inst & op2_mask) >> op2_shift);
    
    if (! store_opcodes) init_opcodes();

    for (i=0; i<num_store_opcodes; i++)
    {
	if (opcode == store_opcodes[i])
	{
	    return 1;
	}
    }
    return 0;
}
			      



