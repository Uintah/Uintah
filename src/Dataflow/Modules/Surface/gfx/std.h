#define GFX_DEF_FMATH

#ifndef GFX_STD_INCLUDED // -*- C++ -*-
#define GFX_STD_INCLUDED

/************************************************************************

  Standard base include file for all gfx-based programs.  This defines
  various common stuff that is used elsewhere.

  $Id$

 ************************************************************************/


#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream.h>

//
// Define the real (ie. default floating point) type
//
#ifdef GFX_REAL_FLOAT
typedef float real;
#else
#define GFX_REAL_DOUBLE
typedef double real;
#endif

//
// Define the boolean type and true/false constants.
// 

#define GFX_NO_BOOL
#ifndef GFX_NO_BOOL
typedef int bool;
const int false = 0;
const int true  = 1;
#endif

// Define True/False as synonyms for true/false
#ifndef GFX_NO_BOOL_MACROS
#  ifndef True
#    define True true
#    define False false
#  endif
#endif

// Handle platforms which don't define M_PI in <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//
// Define min/max.
// These are defined as inlined template functions because that's a more
// C++ish thing to do.
//
#ifndef GFX_NO_MINMAX
#  ifndef min
template<class T>
inline T min(T a, T b) { return (a < b)?a:b; }

template<class T>
inline T max(T a, T b) { return (a > b)?a:b; }
#  endif
#endif

//
// For the old school, we also have MIN/MAX defined as macros.
//
#ifndef MIN
#define MIN(a,b) (((a)>(b))?(b):(a))
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif


#ifndef ABS
#define ABS(x) (((x)<0)?-(x):(x))
#endif

#ifdef GFX_DEF_FMATH
#define fabsf(a) ((float)fabs((double)a))
#define cosf(a) ((float)cos(double)a)
#define sinf(a) ((float)sin(double)a)
#endif

#ifndef FEQ_EPS
#define FEQ_EPS 1e-6
#define FEQ_EPS2 1e-12
#endif
inline bool FEQ(double a,double b,double eps=FEQ_EPS) { return fabs(a-b)<eps; }
inline bool FEQ(float a,float b,float eps=FEQ_EPS) { return fabsf(a-b)<eps; }

#ifndef GFX_NO_AXIS_NAMES
enum Axis {X=0, Y=1, Z=2, W=3};
#endif




#define fatal_error(s)  report_error(s,__FILE__,__LINE__)

#ifdef assert
#  undef assert
#endif
#define  assert(i)  (i)?((void)NULL):assert_failed(# i,__FILE__,__LINE__)

#ifdef SAFETY
#  define AssertBound(t) assert(t)
#else
#  define AssertBound(t)
#endif

//
// Define the report_error and assert_failed functions.
//
inline void report_error(char *msg,char *file,int line)
{
    cerr << msg << " ("<<file<<":"<<line<<")"<<endl;
    exit(1);
}

inline void assert_failed(char *text,char *file,int line)
{
    cerr << "Assertion failed: {" << text <<"} at ";
    cerr << file << ":" << line << endl;
    abort();
}

inline void assertf(int test, char *msg,
                    char *file=__FILE__, int line=__LINE__)
{
    if( !test )
    {
        cerr << "Assertion failed: " << msg << " at ";
        cerr << file << ":" << line << endl;
        abort();
    }
}

// GFX_STD_INCLUDED
#endif
