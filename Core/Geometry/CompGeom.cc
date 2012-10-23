/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 *
 */

#include <Core/Geometry/CompGeom.h>
#include <iostream>

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
closest_point_on_tri(Point &result, const Point &orig,
                     const Point &p0, const Point &p1, const Point &p2)
{
  const Vector edge1 = p1 - p0;
  const Vector edge2 = p2 - p0;

  const Vector dir = Cross(edge1, edge2);

  const Vector pvec = Cross(dir, edge2);
  
  const double inv_det = 1.0 / Dot(edge1, pvec);

  const Vector tvec = orig - p0;
  double u = Dot(tvec, pvec) * inv_det;

  const Vector qvec = Cross(tvec, edge1);
  double v = Dot(dir, qvec) * inv_det;

  if (u < 0.0)
  {
    distance_to_line2_aux(result, orig, p0, p2);
  }
  else if (v < 0.0)
  {
    distance_to_line2_aux(result, orig, p0, p1);
  }
  else if (u + v > 1.0)
  {
    distance_to_line2_aux(result, orig, p1, p2);
  }
  else
  {
    result = p0 + u * edge1 + v * edge2;
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


bool
closest_line_to_line(double &s, double &t,
                     const Point &a0, const Point &a1,
                     const Point &b0, const Point &b1)
{
  const Vector u = a1 - a0;
  const Vector v = b1 - b0;
  const Vector w = a0 - b0;

  const double a = Dot(u, u);
  const double b = Dot(u, v);
  const double c = Dot(v, v);
  const double d = Dot(u, w);
  const double e = Dot(v, w);
  const double D = a*c - b*b;

  if (D < EPSILON)
  {
    s = 0.0;
    t = (b>c?d/b:e/c);
    return false;
  }
  else
  {
    s = (b*e - c*d) / D;
    t = (a*e - b*d) / D;
    return true;
  }
}


void
uniform_sample_triangle(Point &p, const Point &p0,
                        const Point &p1, const Point &p2,
                        MusilRNG &rng)
{
  // Fold the quad sample into a triangle.
  double u = rng();
  double v = rng();
  if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
  
  // Compute the position of the random point.
  p = p0+((p1-p0)*u)+((p2-p0)*v);
}


void
uniform_sample_tetrahedra(Point &p, const Point &p0, const Point &p1,
                          const Point &p2, const Point &p3,
                          MusilRNG &rng)
{
  double t = rng();
  double u = rng();
  double v = rng();

  // Fold cube into prism.
  if (t + u > 1.0)
  {
    t = 1.0 - t;
    u = 1.0 - u;
  }

  // Fold prism into tet.
  if (u + v > 1.0)
  {
    const double tmp = v;
    v = 1.0 - t - u;
    u = 1.0 - tmp;
  }
  else if (t + u + v > 1.0)
  {
    const double tmp = v;
    v = t + u + v - 1.0;
    t = 1.0 - u - tmp;
  }

  // Convert to Barycentric and compute new point.
  const double a = 1.0 - t - u - v;
  p = (p0.vector()*a + p1.vector()*t + p2.vector()*u + p3.vector()*v).point();
}


double
tetrahedra_volume(const Point &p0, const Point &p1,
                  const Point &p2, const Point &p3)
{
  return fabs(Dot(Cross(p1-p0,p2-p0),p3-p0)) / 6.0;
}


void
TriTriIntersection(const Point &A0, const Point &A1, const Point &A2,
                   const Point &B0, const Point &B1, const Point &B2,
                   std::vector<Point> &results)
{
  double t, u, v;
  if (RayTriangleIntersection(t, u, v, false, A0, A1-A0, B0, B1, B2) &&
      t >= 0.0 && t <= 1.0)
  {
    results.push_back(A0 + (A1-A0) * t);
  }
  if (RayTriangleIntersection(t, u, v, false, A1, A2-A1, B0, B1, B2) &&
      t >= 0.0 && t <= 1.0)
  {
    results.push_back(A1 + (A2-A1) * t);
  }
  if (RayTriangleIntersection(t, u, v, false, A2, A0-A2, B0, B1, B2) &&
      t >= 0.0 && t <= 1.0)
  {
    results.push_back(A2 + (A0-A2) * t);
  }
  if (RayTriangleIntersection(t, u, v, false, B0, B1-B0, A0, A1, A2) &&
      t >= 0.0 && t <= 1.0)
  {
    results.push_back(B0 + (B1-B0) * t);
  }
  if (RayTriangleIntersection(t, u, v, false, B1, B2-B1, A0, A1, A2) &&
      t >= 0.0 && t <= 1.0)
  {
    results.push_back(B1 + (B2-B1) * t);
  }
  if (RayTriangleIntersection(t, u, v, false, B2, B0-B2, A0, A1, A2) &&
      t >= 0.0 && t <= 1.0)
  {
    results.push_back(B2 + (B0-B2) * t);
  }
}




}

