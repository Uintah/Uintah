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
 * irix.c: Required on our local Irix implementation.
 *
 *************************************************************************/

#include <stdio.h>
#include "message.h"
#include "thread.h"

void MSG_INIT_I(Message **m, unsigned short opp)
{
    *m = ltable[Thread(thread_self())].send_buffer;
    (*m)->op = opp;
    (*m)->length = 0;
}

void MSG_OP(Message *m, unsigned short opp)
{
    m->op = opp;
}

void MSG_INSERT_I(Message *m, unsigned long d)
{
    *(unsigned long *) (m->data + m->length) = d;
    m->length += 4;
}

void MSG_INSERT_BLK_I(Message *m, char *a, int s)
{
    mem_copy(a, (char *) m->data + m->length, s);
    m->length += s;
}

void MSG_CLEAR(Message *m)
{
    m->length = 0;
}

void ASSERT_I(int cond, int line, char *file)
{
#ifdef CHECKASSERT    
    if (! cond)
    {
	fprintf(stderr, "Assertion failed at line %d in file %s.\n",
		line, file);
	abort();
    }
#endif
}

void PANIC_I(char *message, int line, char *file)
{
    fprintf(stderr, "***PANIC (proc %d): %s\n", Qknodeid, message);
    fprintf(stderr, "Line %d, File %s.\n", line, file);
    graceful_exit();
}

void printfloat(FILE *fp, double num)
{
    double frac, frac10;
    int tmp;
    int numdec = 0;
#define NUM_DEC_PT  3

    tmp = (int) num;
    fprintf(fp, "%d.", tmp);
    frac = num - tmp;
    while (frac > 0)
    {
	frac10 = frac*10;
	tmp = (int) frac10;
	fprintf(fp,"%d", tmp);
	frac = frac10 - tmp;
	numdec++;
	if (numdec >= NUM_DEC_PT) break;
    }
    fflush(fp);
}

