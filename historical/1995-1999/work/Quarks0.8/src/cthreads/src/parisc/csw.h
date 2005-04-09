/*
 * Copyright (c) 1990,1994 The University of Utah and
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
 * structure to build context area for C-threads package
 */

struct csw {
	unsigned r2;
	unsigned r3;
	unsigned r4;
	unsigned r5;
	unsigned r6;
	unsigned r7;
	unsigned r8;
	unsigned r9;
	unsigned r10;
	unsigned r11;
	unsigned r12;
	unsigned r13;
	unsigned r14;
	unsigned r15;
	unsigned r16;
	unsigned r17;
	unsigned r18;
	unsigned r26;
	unsigned sr3;
	unsigned cr11;
	double fr0;
	double fr1;
	double fr2;
	double fr3;
	double fr12;
	double fr13;
	double fr14;
	double fr15;
};
