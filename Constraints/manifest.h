
/*
 *  manifest.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef _H_MANIFEST_VALUES
#define _H_MANIFEST_VALUES

#include <stdlib.h>
#include <math.h>
#include <Classlib/Assert.h>
#include <iostream.h>


typedef double Real;
typedef unsigned int Index;

inline Real
RealAbs ( Real x )
{
   return fabs(x);
}

inline int
RealRound ( Real x )
{
   return (int)(floor (x + 0.5));
}

inline Real
RealMin ( Real x1, Real x2 )
{
   return ((x1 < x2) ? x1 : x2);
}

inline Real
RealMax ( Real x1, Real x2 )
{
   return ((x1 > x2) ? x1 : x2);
}


enum boolean { false=0, true=1 };
inline boolean
NOT ( boolean b )
{
   return ((b == true) ? false : true);
}


/*const Real Pi = 3.14159;

inline Real
DegreesToRadians ( Real x )
{
   return x * Pi / 180.0;
}

inline Real
RadiansToDegrees ( Real x )
{
   return x * 180.0 / Pi;
}
*/
const Real Epsilon = 0.000006;


#define NOT_FINISHED(what) cerr << what << ": Not Finished " << __FILE__ << " (line " << __LINE__ << ") " << endl
#define ERROR_HERE(what) cerr << "Error: " << what << " in " << __FILE__ << " (line " << __LINE__ << ") " << endl

static void Error(char *errorString)
{
    cerr << "Error: " <<  errorString << "." << endl;
    exit(-1);
}


#endif
