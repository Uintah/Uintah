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
 * mips_instruction.c: mips specific.
 *
 *************************************************************************/
#include <stdio.h>
#include "instruction.h"

#ifndef mips
THIS FILE IS ONLY FOR MIPS 2000/3000 INSTRUCTION SET
#endif

int STORE_INSTRUCTION(unsigned long inst)
{
    return 1;
/*  XXX if we get hold of the correct PC, then the following code
    is ok. But we are not able to get the PC.

    unsigned long opcode = (inst & op_mask) >> op_shift;
    
    printf("Opcode = %d\n", opcode);
    if ((opcode >= 40) && (opcode <= 43)) return 1;	
    if (opcode == 46) return 1;
    return 0;
*/
}

int LOAD_INSTRUCTION(unsigned long inst)
{
    return !STORE_INSTRUCTION(inst);
}

