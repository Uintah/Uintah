/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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


#ifndef __MinMax_h
#define __MinMax_h

namespace SemotusVisum {

// 2 Integers
inline  int Min(int d1, int d2)
{
    return d1<d2?d1:d2;
}

inline  int Max(int d1, int d2)
{
    return d1>d2?d1:d2;
}

// 2 Integers
inline  long Min(long d1, long d2)
{
    return d1<d2?d1:d2;
}

inline  long Max(long d1, long d2)
{
    return d1>d2?d1:d2;
}

// 2 doubles
inline  double Max(double d1, double d2)
{
  return d1>d2?d1:d2;
}

inline  double Min(double d1, double d2)
{
  return d1<d2?d1:d2;
}

// 3 doubles
inline  double Min(double d1, double d2, double d3)
{
    double m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline  double Max(double d1, double d2, double d3)
{
    double m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 integers
inline  int Min(int d1, int d2, int d3)
{
    int m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline  int Max(int d1, int d2, int d3)
{
    int m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

// 3 integers
inline  long  Min(long  d1, long  d2, long  d3)
{
    long  m=d1<d2?d1:d2;
    m=m<d3?m:d3;
    return m;
}

inline  long  Max(long  d1, long  d2, long  d3)
{
    long  m=d1>d2?d1:d2;
    m=m>d3?m:d3;
    return m;
}

} // End namespace SCIRun


#endif /* SCI_Math_MinMax_h */
