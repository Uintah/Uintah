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
 * sor.c: succesive over relaxation - example applications for Quarks.
 *
 *************************************************************************/

#include <stdio.h>
#include <signal.h>
#include "Quarks.h"

#define ELEMENT_DOUBLE

#ifdef ELEMENT_INT 
typedef int element_t;
#else
typedef double element_t;
#endif

extern void printfloat(FILE *, double);

#define SIZE 	        34
#define NUM_ITER      1000
#define init_val      ((element_t) 100)

static int numprocs;
static element_t *sh_region;
struct global {
    Id ba;
} *gl;


static int equal(element_t a, element_t b)
{
#define EPS 0.01

#ifdef ELEMENT_INT
    return a==b;
#else
    return ( ((a-b) < EPS) || ((b-a) < EPS) );
#endif
}

void writeB(int row, int col, element_t value)
{
    int mindex = row*SIZE + col;
    
    *(sh_region + mindex) = value;
}

element_t readB(int row, int col)
{
    int mindex = row*SIZE + col;
    element_t value;

    value = *(sh_region + mindex);
    return value;
}

element_t *sh_addr(int row, int col)
{
    int mindex = row*SIZE + col;
    return (sh_region + mindex);
}

int sor()
{
    int start_row, end_row;
    int start_col, end_col;
    int numrows, extrarows;
    element_t *temp, *prev_temp;
    element_t *temp_h, *prev_temp_h;
    element_t val;
    int       iter;
    int NW,SW,SE,NE;
    int mismatch=0;
    element_t thing, sum, dval;
    FILE *fp;
    int row, col;

    temp_h       = (element_t *) malloc(sizeof(element_t)*(SIZE+1));
    prev_temp_h  = (element_t *) malloc(sizeof(element_t)*(SIZE+1));

    /* double word align the allocated region */
    temp_h++; prev_temp_h++;
    temp_h = (element_t *) ((unsigned long) temp_h & ~7);
    prev_temp_h = (element_t *) ((unsigned long) prev_temp_h & ~7);
    
    numrows = (SIZE-2)/numprocs;
    extrarows = (SIZE-2)%numprocs;

    start_row = (Qknodeid - 1)*numrows + 1;
    end_row = (Qknodeid)*numrows;

    start_col = 1;
    end_col   = SIZE-2;

    iter = (Id) gl->ba;

    if (numprocs > 1) Qk_wait_barrier(gl->ba, numprocs);
    printf("begining computation\n");
    for (iter=0; iter<NUM_ITER; iter++)
    {
	for (row=start_row; row <= end_row; row++)
	{
	    temp = temp_h;
	    for (col=start_col; col <= end_col; col++)
	    {
		dval   = readB(row-1, col);
		dval  += readB(row, col-1);
		dval  += readB(row+1, col);
		dval  += readB(row, col+1);
		*temp  = dval / 4;
		temp++;
	    }
	    prev_temp = prev_temp_h;
	    temp = temp_h;
	    for (col=start_col; col <= end_col; col++)
	    {
		if (row > start_row)
		    writeB(row-1, col, *prev_temp);
		if (row == end_row)
		    writeB(row, col, *temp);
		*prev_temp = *temp;
		prev_temp++; temp++;
	    }
	}

	if (numprocs > 1) Qk_wait_barrier(gl->ba, numprocs);
	if (iter%10 == 0) printf("Done iter %d\n", iter);
    }

    if (numprocs > 1) Qk_wait_barrier(gl->ba, numprocs);
    printf("Verifying the result\n");

    sum=0;
    if (Qkmaster)
    {
	fp = fopen("sor1.out", "w");
    }
    for (row=0; row<SIZE; row++)
    {
	for (col=0; col<SIZE; col++)
	{
	    thing = readB(row, col);
	    if (Qkmaster)
	    {
#ifdef ELEMENT_INT
		fprintf(fp, "%d ", thing); 
#else
#ifdef IRIX
	        printfloat(fp, thing);
		fprintf(fp, "   ");
#else
	        fprintf(fp, "%5.1lf ", thing);
#endif
#endif
	    }
	    sum += thing;
	}
	if (Qkmaster) fprintf(fp, "\n"); 
    }
    if (Qkmaster) fclose(fp);
#ifdef ELEMENT_INT
    printf("Sum = %d\n", sum);
#else
#ifdef IRIX
    printfloat(stdout, sum);
#else
    printf("Sum = %lf\n", sum);
#endif
#endif

    for (row=0; row<SIZE/2; row++)
	for (col=0; col<SIZE/2; col++)
	{
	    NW = readB(row, col);
	    SW = readB(SIZE-row-1, col);
	    SE = readB(SIZE-row-1, SIZE-col-1);
	    NE = readB(row, SIZE-col-1);
	    if (!(equal(NE,SW) && equal(NW,SE) && equal(NW, NE)))
	    {
		mismatch++;
	    }
	}
    printf("Mismatches = %d\n", mismatch);
}


initialize()
{	
    int i;
    int row, col;

    Qk_create_region(1, SIZE*SIZE*sizeof(element_t),
		     (unsigned long *) &sh_region);
    Qk_create_region(2, sizeof(*gl), 
		     (unsigned long *) &gl);

    gl->ba = Qk_newbarrier();
    printf("Initializing the grid... "); fflush(stdout);
    for (row=0; row<SIZE; row++)
    {
	writeB(row,0,init_val);
	writeB(row,SIZE-1,init_val);
    }
    for(col=1;col<SIZE-1;col++)
    {	
	writeB(0,col,init_val);
	writeB(SIZE-1,col,init_val);
    }
    for (row=1;row<SIZE-1;row++)
	for(col=1;col<SIZE-1;col++)
	    writeB(row,col,0);
    printf("done\n");

    for (i=0; i<(numprocs-1); i++)
	Qk_fork(sor);

}

void
main(int argc, char *argv[])
{
    int i;

    if (argc < 2)
    {
	printf("usage %s numprocs\n", argv[0]);
	exit(0);
    }
    numprocs = (int) atoi(argv[1]);

    if ((SIZE-2)%numprocs)
    {
	printf("Sorry, size of matrix does not correspond to numprocs\n");
	exit(0);
    }
    Qk_initialize();
    if (Qkchild) return;

    initialize();
    sor();
    if (numprocs > 1) Qk_wait_for_threads();
    Qk_shutdown(0);
}


#if 0
/* code not used */
void printmatrix()
{
    int i,j;
    int *r = sh_region;

    for (i=0; i<SIZE; i++)
    {
	for (j=0; j<SIZE; j++)
	    printf("%2d ", *r++);
	printf("\n");

    }
}

void printrow()
{
    int i,j;
    int *r;

    r = sh_region + SIZE;
    printf("row #1:\n");
    for (i=0; i<SIZE; i++)
	printf("%2d ", *r++);
    printf("\n------------------------------\n");

    r = sh_region + (SIZE-2)*SIZE;
    printf("Last but one row: \n");
    for (i=0; i<SIZE; i++)
	printf("%2d ", *r++);
    printf("\n");
}
#endif
