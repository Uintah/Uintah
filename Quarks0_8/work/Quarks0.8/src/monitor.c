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
 * monitor.c: To display the ability to share across applications.
 *
 *************************************************************************/

#include <stdio.h>
#include "Quarks.h"

#define SIZE 34
#define ELEMENT_DOUBLE


#ifdef ELEMENT_INT
typedef int element_t;
#else
typedef double element_t;
#endif

static element_t *sh_region;
extern void printfloat(FILE *, double);

static void myprint(element_t val)
{
#ifdef ELEMENT_INT 
    printf("%d ", val);
#else
#ifdef IRIX
    printfloat(stdout, val);
#else
    printf("%5.1lf ", val); 
#endif
#endif
    fflush(stdout);
}

static element_t readB(int row, int col)
{
    int mindex = row*SIZE + col;
    element_t value;

    value = *(sh_region + mindex);
    return value;
}

static void monitor(Id regionid)
{
    int i,j;
    element_t val;
    char answer[3];

    printf("Monitor: opening the region\n");
    sh_region = (element_t *) Qk_open_region(regionid);
    while(1)
    {
	for (i=0; i<SIZE; i++)
	{
	    for (j=0; j<SIZE; j++)
	    {
		val = readB(i, j);
		myprint(val);
	    }
	    printf("\n");
	}
	printf("Continue? [y/n] "); fflush(stdout);
	scanf("%s", answer);
	if (answer[0] == 'n') break;
    }
}

main(int argc, char *argv[])
{
    Qk_initialize();
    monitor(1);
    Qk_shutdown(1);
}

