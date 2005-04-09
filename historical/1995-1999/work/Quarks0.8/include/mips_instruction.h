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
 * mips_instruction.h: mips specific.
 *
 *************************************************************************/

#ifndef _MIPS_INSTRUCTION_H_
#define _MIPS_INSTRUCTION_H_

#ifndef mips
THIS FILE IS ONLY FOR MIPS 2000/3000 INSTRUCTION SET
#endif

#define op_mask   0xfc000000
#define op_shift  26


static unsigned long load_opcodes[] = {
    32,  /* LB  */
    33,  /* LH  */
    34,  /* LWL */
    35,  /* LW  */
    36,  /* LBU */
    37,  /* LHU */
    38,  /* LWR */
};

static unsigned long store_opcodes[] = {
    40,  /* SB  */
    41,  /* SH  */
    42,  /* SWL */
    43,  /* SW  */
    46,  /* SWR */
};

#endif  /* _MIPS_INSTRUCTION_H_ */
