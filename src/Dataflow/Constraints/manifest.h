
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
#include <Util/Assert.h>
#include <iostream.h>

namespace PSECommon {
namespace Constraints {

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


#define NOT_FINISHED(what) cerr << what << ": Not Finished " << __FILE__ << " (line " << __LINE__ << ") " << endl
#define ERROR_HERE(what) cerr << "Error: " << what << " in " << __FILE__ << " (line " << __LINE__ << ") " << endl

} // End namespace Constraints
} // End namespace PSECommon

#endif
