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
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <values.h>
using std::string;
using std::cout;
using std::setw;

/*------------- Useful constants -------------*/

#define DRIVER_DEBUG      1                 // Debug flag
#define NUM_VARS          1                 // # variable types; we use only cell-centered
#ifndef PI                                  // Pi to 32 digits
#define PI 3.14159265358979323846264338327950L
#endif

/*------------- Type definitions -------------*/

template<class VAR> class Vector;
typedef unsigned short Counter;            // Non-negative integer/counter
typedef Vector<double> Location;           // Data node location in d-dim space
enum Side {                                // Left/right boundary in each dim
  Left = -1,
  Right = 1,
  NA = 3
};

inline Side& operator++(Side &s)
{
  return s = Side(s+2);
}

/*------------- Useful macros -------------*/

#define MIN(a,b) (((a) < (b)) ? (a) : (b))    // Minimum of two numbers
#define MAX(a,b) (((a) > (b)) ? (a) : (b))    // Maximum of two numbers
#define ABS(a)   (((a) > 0) ? (a) : (-(a)))   // Absolute value of a number
#define DELETE_PTR(x) { if (x) delete x; x = 0; }
#define DELETE_BRACKET(x) { if (x) delete [] x; x = 0; }

/*------------- Global variables, must be declared in main file -------------*/

extern int     MYID;     /* The same as this proc's myid, but global */

#endif /* _MACROS_H */
