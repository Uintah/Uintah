/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*#############################################################################
  # Macros.h - Global macros.
  #============================================================================
  # This h-file includes useful types, constants and macros (e.g.,min and max),
  # which are convenient to re-define for uniformity of compilation under
  # different operating systems.
  #
  # See also: Timer.
  #
  # Revision history:
  # -----------------
  # 25-DEC-2004   Added comments.
  ###########################################################################*/

#ifndef _MACROS_H
#define _MACROS_H

/*================== Standard Library includes ==================*/
#ifndef MPIPP_H
#  define MPIPP_H
#endif
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cassert>

/*------------- Useful constants -------------*/

#define DRIVER_DEBUG      1                 // Debug flag
#define NUM_VARS          1                 // # variable types; we use only cell-centered
#ifndef PI                                  // Pi to 32 digits
#define PI 3.14159265358979323846264338327950L
#endif

/*------------- Type definitions and forward declarations -------------*/

class DebugStream;
template<class VAR> class Vector;
typedef unsigned short Counter;            // Non-negative integer/counter
typedef Vector<double> Location;           // Data node location in d-dim space

/*------------- Useful macros -------------*/

//#define MIN(a,b) (((a) < (b)) ? (a) : (b))    // Minimum of two numbers
//#define MAX(a,b) (((a) > (b)) ? (a) : (b))    // Maximum of two numbers
//#define ABS(a)   (((a) > 0) ? (a) : (-(a)))   // Absolute value of a number
#define DELETE_PTR(x) { if (x) delete x; x = 0; }
#define DELETE_BRACKET(x) { if (x) delete [] x; x = 0; }

/*------------- Global variables, must be declared in main file -------------*/

extern int MYID;     // The same as this proc's myid, but global
extern DebugStream dbg;
extern DebugStream dbg0;

#endif /* _MACROS_H */
