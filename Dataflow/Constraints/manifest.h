
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

namespace SCIRun {

typedef double Real;
typedef unsigned int Index;

inline PSECORESHARE Real
RealAbs ( Real x )
{
   return x<0?-x:x;
}


} // End namespace SCIRun

#endif
