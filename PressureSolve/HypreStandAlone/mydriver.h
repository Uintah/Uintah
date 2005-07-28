/*--------------------------------------------------------------------------
 * File: mydriver.h
 *
 * Header file for the test driver for semi-structured matrix interface.
 * The declarations in this file are common for most of the cc files in this
 * code.
 *
 * Revision history:
 * 19-JUL-2005   Oren   Created to allow rhs(), diffusion() etc. in a separate
 *                      file.
 *--------------------------------------------------------------------------*/
#ifndef __MYDRIVER_H__
#define __MYDRIVER_H__

/*================== Standard Library includes ==================*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <vector>
using std::vector;

/*================== Macros ==================*/

#define DEBUG      1              // Debug flag
#define MAX_DIMS   3              // Maximum number of dimensions
#define NUM_VARS   1              // # variable types; we use only cell-centered

/*================== Type definitions ==================*/

typedef unsigned short Counter;   // Non-negative integer/counter
typedef int Index[MAX_DIMS];      // Subscript in d-dimensional space
typedef vector<double> Location;  // Data node location in d-dim. space
enum Side {                       // Left or right boundary in each dimension
  Left = -1,
  Right = 1
};

/*================== Our Library includes ==================*/

#include "IntMatrix.h"

/*================== Global variables ==================*/

extern int         MYID;          // The same as myid (this proc's id)

/*================== Class delcarations ==================*/

#endif // __MYDRIVER_H__
