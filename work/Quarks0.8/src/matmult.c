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
 * matmult.c: matrix multiplication - example applications for Quarks.
 *
 *************************************************************************/

#include <stdio.h>

#include "Quarks.h"

#define ELEMENT_INT

#ifdef ELEMENT_INT
typedef unsigned long element_t;
#endif
#ifdef ELEMENT_DOUBLE
typedef double element_t;
#endif

static long myrandom()
{
#ifdef HPUX
    return (long) rand();
#else
    return (long) random();
#endif
}

extern void matmult(void);

/* The shared memory regions */
element_t *matA, *matB, *matC;

/* The shared memory region containing the synchronization primitives */
struct global {
    Id ba;
} *gl;

/* Other global variables of the program, that are initialized via
 * command line arguments */
int       matsize, numproc;


static int start_row, end_row, extra_rows;

element_t newval()
{
    element_t v;
    static int maxval = 10;

    v = (element_t) (myrandom() % maxval);   
    v += (element_t) ((double) (myrandom()%maxval)/(double) maxval);
    return v;
}

void write_elem(element_t *m, int i, int j, element_t val)
{
    int offset = i*matsize + j;
    *(m+offset) = val;
}

element_t read_elem(element_t *m, int i, int j)
{
    element_t val;

    int offset = i*matsize + j;
    val = *(m+offset);
    return val;
}

void init_matrices()
{
    int i,j;

    for (i=0; i<matsize; i++)
	for (j=0; j<matsize; j++)
	{
	    write_elem(matA, i, j, newval());
	    write_elem(matB, i, j, newval());
	    write_elem(matC, i, j, 0);
	}
}


void initialize()
{
    /* Initialize the shared memory regions and the 
     * synchronization primitives 
     */
    int i;

    Qk_create_region(1, matsize*matsize*sizeof(element_t),
		     (unsigned long *) &matA);
    Qk_create_region(2, matsize*matsize*sizeof(element_t),
		(unsigned long *) &matB);
    Qk_create_region(3, matsize*matsize*sizeof(element_t),
		(unsigned long *) &matC);

    init_matrices();

    if (numproc > 1)
    {
	Qk_create_region(4, sizeof(struct global), 
		    (unsigned long *) &gl);

	gl->ba = Qk_newbarrier();
	
	/* Create remote threads */
	for (i=0; i < (numproc-1); i++)
	    Qk_fork(matmult);
    }
}

void verify_result()
{
    /* Verifies the resulting matrix, by computing the overall sum
     * and the row-wise sum of the elements of the matrix. The sums
     * should be identical to that obtained by running the program 
     * with one process.
     */

    int i,j;
    element_t sum;
    element_t val;

    printf("Verifying the result\n");
    sum = 0;

    printf("The row-wise sum of the result matrix:\n");
    for (i=0; i<matsize; i++)
    {
	sum = 0;
	for (j=0; j<matsize; j++)
	{
	    val  = read_elem(matC, i, j);
	    sum += val;
	}
#ifdef ELEMENT_INT
	printf("%d\n", sum);
#endif
#ifdef ELEMENT_DOUBLE
	printf("%.2lf\n", sum);
#endif
    }
    
    printf("---------> done <--------\n");
}

void matmult(void)
{
    /* The matrix multiplication function. Slave threads start
     * execution here.
     */

    int       i,j,k;
    element_t prod;

    /* Assumes that matrices are square, with size a multiple
     * of the number of processes.
     */
    extra_rows = matsize%numproc;
    
    start_row = ((matsize-extra_rows)/numproc)*(Qknodeid-1) + extra_rows;
    if (Qknodeid == 1) start_row = 0;
    end_row   = ((matsize-extra_rows)/numproc)*Qknodeid - 1 + extra_rows;
    printf("Process %d started. Working on rows %d thru %d\n", 
	   Qknodeid, start_row, end_row);

    if (numproc > 1) Qk_wait_barrier(gl->ba, numproc);
    for (i=start_row; i<=end_row; i++)
	for (j=0; j<matsize; j++)
	{
	    prod = 0;
	    for (k=0; k<matsize; k++)
		prod += read_elem(matA, i, k)*read_elem(matB, k, j);
	    write_elem(matC, i, j, prod);
	}

    /* Wait for everyone to finish. Memory will become consistent
     * only after the processes cross the barrier.
     */
    if (numproc > 1) Qk_wait_barrier(gl->ba, numproc);

    verify_result();
}


main(int argc, char *argv[])
{
    int  i;
    char *addr;

    if (argc < 3)
    {
	fprintf(stderr,"usage: %s numproc size\n", argv[0]);
	fflush(stderr);
	exit(0);
    }
    numproc = (int) atoi(argv[1]);
    matsize = (int) atoi(argv[2]);

    Qk_initialize();
    if (Qkchild) return;

    initialize(); 
    matmult();

    if (numproc > 1) Qk_wait_for_threads();
    Qk_shutdown(0);
}







