//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : CubicHermitian.cc
//    Author : Martin Cole
//    Date   : Wed Apr 28 10:21:01 2004

#include <Core/Basis/CubicHermitian.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {

// specialization code
template <>
double
CubicHermitianBase<Point>::distance(const Point &p0, const Point &p1) const
{
  Vector v = p1 - p0;
  return v.length2();
}

//! f(u) =C'(u) dot (C(u) - P)
//! the distance from P to C(u) is minimum when f(u) = 0
template <>
double calc_next_guess<Point>(const double xi, const Point &val,
			      const Point &interp, const Point &deriv, 
			      const Point &deriv2)
{
  //C(u) == interpolate(xi, cd);
  //C'(u) == derivate(xi, cd);
  double num = Dot(deriv, (interp - val)); 
  double den = Dot(deriv, (interp - val)) + (deriv.asVector().length2());
  return xi - (num / den);
}



} //end namespace SCIRun
