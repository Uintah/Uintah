/*--------------------------------------------------------------------------
 * File: mydriver.h
 *
 * Header file for the test driver for semi-structured matrix interface.
 *
 * Revision history:
 * 19-JUL-2005   Oren   Created to allow rhs(), diffusion() etc. in a separate
 *                      file.
 *--------------------------------------------------------------------------*/
#ifndef __MYDRIVER_H__
#define __MYDRIVER_H__

/*================== Library includes ==================*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <vector>

using std::vector;

/*================== Macros ==================*/

#define DEBUG      1        // Debug flag
#define MAX_DIMS   3        // Maximum number of dimensions
#define NUM_VARS   1        // # variable types; we use only cell-centered

/*================== Global variables ==================*/

extern int         numDims;  // Actual number of dimensions
extern int         MYID;     // The same as myid (this proc's id)
//extern char boundaryTypeString[3][256];

/*================== Type definitions ==================*/

typedef int    Index[MAX_DIMS];    // Subscript in d-dimensional space
typedef vector<double> Location;   // Data node location in d-dim. space

/*================== Function delcarations ==================*/

/* mydriver.cc */

/*================== Class delcarations ==================*/

#endif
