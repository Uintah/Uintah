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
 * qtest.c: test for locks and shared regions.
 *
 *************************************************************************/

#include <stdio.h>

#include "Quarks.h"

static int numobj;
static int numcycles;

static int numprocs;
int    *mat;
struct global {
    Id ba;
    Id *lk;
} *gl;

static long myrandom()
{
#ifdef HPUX
    return (long) rand();
#else
    return (long) random();
#endif
}

void compute()
{
    int i, j;
    int ind1, ind2, val, temp, sum;

    printf("Compute thread begins. \n");
    Qk_wait_barrier(gl->ba, numprocs);

    for (i=0; i<numcycles; i++)
    {
	ind1 = myrandom()%numobj;
	ind2 = myrandom()%numobj;
	val  = myrandom()%10;

	if (ind2 > ind1) {temp = ind1; ind1 = ind2; ind2 = temp;}
	Qk_acquire(gl->lk[ind1]);
	if (ind1 != ind2) Qk_acquire(gl->lk[ind2]);
	mat[ind1] = mat[ind1] - val;
	mat[ind2] = mat[ind2] + val;
	if (ind1 != ind2) Qk_release(gl->lk[ind2]);
	Qk_release(gl->lk[ind1]);

	if ((i>0) && (i%10 == 0)) fprintf(stderr, "Done %d cycles\n", i);
    }

    Qk_wait_barrier(gl->ba, numprocs);

    printf("Done computation\n");
    sum = 0;
    for (i=0; i<numobj; i++)
    {
	sum += mat[i];
    }
	
    fprintf(stderr, "************* Resulting sum = %d\n", sum);

}

static void initialize()
{
    int  i;
    Id   ba;

    Qk_create_region(1, numobj*sizeof(int),
		     (unsigned long *) &mat);
    Qk_create_region(2, sizeof(Id)*(numobj+2),
		     (unsigned long *) &gl);
    gl->ba = Qk_newbarrier();
    gl->lk = ((Id *) gl) + 2;
    for (i=0; i<numobj; i++)
    {
	gl->lk[i] = Qk_newlock();
	mat[i] = 0;
    }

    for (i=1; i<numprocs; i++)
	Qk_fork(compute);
}

main(int argc, char *argv[])
{
    if (argc < 4)
    {
	fprintf(stderr, "Usage: %s numprocs numobj numcycles\n", argv[0]);
	exit(0);
    }
    numprocs = (int) atoi(argv[1]);
    numobj   = (int) atoi(argv[2]);
    numcycles= (int) atoi(argv[3]);

    Qk_initialize();

    if (Qkchild) return;

    initialize();
    compute();
    
    Qk_wait_for_threads();
    Qk_shutdown(0);
}
