/*
 * Copyright (c) 1990,1991,1993,1994 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the
 * Computer Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 */
/*
 * generate assemby symbols
 */
#include <stdio.h>
#include "csw.h"

main()
{
	struct csw *csw = (struct csw *)0;

	printf("#define\tCSW_R2\t0x%x\n", &csw->r2);
	printf("#define\tCSW_R3\t0x%x\n", &csw->r3);
	printf("#define\tCSW_R4\t0x%x\n", &csw->r4);
	printf("#define\tCSW_R5\t0x%x\n", &csw->r5);
	printf("#define\tCSW_R6\t0x%x\n", &csw->r6);
	printf("#define\tCSW_R7\t0x%x\n", &csw->r7);
	printf("#define\tCSW_R8\t0x%x\n", &csw->r8);
	printf("#define\tCSW_R9\t0x%x\n", &csw->r9);
	printf("#define\tCSW_R10\t0x%x\n", &csw->r10);
	printf("#define\tCSW_R11\t0x%x\n", &csw->r11);
	printf("#define\tCSW_R12\t0x%x\n", &csw->r12);
	printf("#define\tCSW_R13\t0x%x\n", &csw->r13);
	printf("#define\tCSW_R14\t0x%x\n", &csw->r14);
	printf("#define\tCSW_R15\t0x%x\n", &csw->r15);
	printf("#define\tCSW_R16\t0x%x\n", &csw->r16);
	printf("#define\tCSW_R17\t0x%x\n", &csw->r17);
	printf("#define\tCSW_R18\t0x%x\n", &csw->r18);
	printf("#define\tCSW_R26\t0x%x\n", &csw->r26);
	printf("#define\tCSW_SR3\t0x%x\n", &csw->sr3);
	printf("#define\tCSW_CR11\t0x%x\n", &csw->cr11);
	printf("#define\tCSW_FR0\t0x%x\n", &csw->fr0);
	printf("#define\tCSW_FR1\t0x%x\n", &csw->fr1);
	printf("#define\tCSW_FR2\t0x%x\n", &csw->fr2);
	printf("#define\tCSW_FR3\t0x%x\n", &csw->fr3);
	printf("#define\tCSW_FR12\t0x%x\n", &csw->fr12);
	printf("#define\tCSW_FR13\t0x%x\n", &csw->fr13);
	printf("#define\tCSW_FR14\t0x%x\n", &csw->fr14);
	printf("#define\tCSW_FR15\t0x%x\n", &csw->fr15);
	printf("#define\tCSW_SIZE\t0x%x\n", (sizeof(struct csw)));

	return 0;
}
