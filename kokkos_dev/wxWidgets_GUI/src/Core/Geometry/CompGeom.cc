/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  CompGeom.cc
 *
 *  Written by:
 *   Allen Sanderson
 *   SCI Institute
 *   University of Utah
 *   August 2005
 *
 *  Copyright (C) 2005 SCI Group
 *
 */

#include <Core/Geometry/CompGeom.h>

#define TOLERANCE_MIN 1.0e-12

namespace SCIRun {

double
distance_to_line2(const Point &p, const Point &a, const Point &b)
{
  Vector m = b - a;
  Vector n = p - a;
  if (m.length2() < TOLERANCE_MIN) {
    return n.length2();
  }
  else {
    const double t0 = Dot(m, n) / Dot(m, m);
    if (t0 <= 0) return (n).length2();
    else if (t0 >= 1.0) return (p - b).length2();
    else return (n - m * t0).length2();
  }
}


void
distance_to_line2_aux(Point &result,
                      const Point &p, const Point &a, const Point &b)
{
  Vector m = b - a;
  Vector n = p - a;
  if (m.length2() < TOLERANCE_MIN) {
    result = a;
  }
  else
  {
    const double t0 = Dot(m, n) / Dot(m, m);
    if (t0 <= 0) 
    {
      result = a;
    }
    else if (t0 >= 1.0)
    {
      result = b;
    }
    else
    {
      Vector offset = m * t0;
      result = a + offset;
    }
  }
}



void
closest_point_on_tri(Point &result, const Point &P,
                     const Point &A, const Point &B, const Point &C)
{
  const Vector E0 = B - A;
  const Vector E1 = C - A;

  const Vector D = A - P;
  const double a = Dot(E0, E0);
  const double b = Dot(E0, E1);
  const double c = Dot(E1, E1);
  const double d = Dot(E0, D);
  const double e = Dot(E1, D);
  //const double f = Dot(D, D);

  const double det = a*c - b*b;
  double s = b*e - c*b;
  double t = b*d - a*e;

  if (s < 0.0) s = 0.0;
  if (t < 0.0) t = 0.0;
  if (s + t < det)
  {
    const double idet = 1.0 / det;
    s *= idet;
    t *= idet;

    result = A + E0 * s + E1 * t;
  }
  else
  {
    distance_to_line2_aux(result, P, B, C);
  }
}




double
RayPlaneIntersection(const Point &p,  const Vector &dir,
		     const Point &p0, const Vector &pn)
{
  // Compute divisor.
  const double Vd = Dot(dir, pn);

  // Return no intersection if parallel to plane or no cross product.
  if (Vd < TOLERANCE_MIN)
    return 1.0e24;

  const double D = - Dot(pn, p0);

  const double V0 = - (Dot(pn, p) + D);
    
  return V0 / Vd;
}


#define EPSILON 1.0e-6


bool
RayTriangleIntersection(double &t, double &u, double &v, bool backface_cull,
                        const Point &orig,  const Vector &dir,
                        const Point &p0, const Point &p1, const Point &p2)
{
  const Vector edge1 = p1 - p0;
  const Vector edge2 = p2 - p0;

  const Vector pvec = Cross(dir, edge2);
  
  const double det = Dot(edge1, pvec);

  if (det < EPSILON && (backface_cull || det > -EPSILON))
    return false;

  const double inv_det = 1.0 / det;
    
  const Vector tvec = orig - p0;

  u = Dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0)
    return false;

  const Vector qvec = Cross(tvec, edge1);

  v = Dot(dir, qvec) * inv_det;
  if (v < 0.0 || u+v > 1.0)
    return false;

  t = Dot(edge2, qvec) * inv_det;

  return true;
}


}

