
/*
 *  MinMax.h: Various implementations of Min and Max
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Math_MinMax_h
#define SCI_Math_MinMax_h 1

#include <SCICore/share/share.h>

namespace SCICore {
namespace Math {

// 2 Integers
inline SCICORESHARE int Min(int d1, int d2)
{
    return d1<d2?d1:d2;
}

inline SCICORESHARE int Max(int d1, int d2)
{
    return d1>d2?d1:d2;
}

// 2 doubles
inline SCICORESHARE double Max(double d1, double d2)
{
  return d1>d2?d1:d2;
}

inline SCICORESHARE double Min(double d1, double d2)
{
  return d1<d2?d1:d2;
}

// 3 doubles
inline SCICORESHARE double Min(double d1, double d2, double d3)
{
    double m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline SCICORESHARE double Max(double d1, double d2, double d3)
{
    double m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 integers
inline SCICORESHARE int Min(int d1, int d2, int d3)
{
    int m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline SCICORESHARE int Max(int d1, int d2, int d3)
{
    int m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

} // End namespace Math
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:34  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:03  mcq
// Initial commit
//
// Revision 1.4  1999/07/01 16:44:23  moulding
// added SCICORESHARE to enable win32 SCICORESHAREd libraries (dll's)
//
// Revision 1.3  1999/05/06 19:56:19  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:23  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//

#endif /* SCI_Math_MinMax_h */
