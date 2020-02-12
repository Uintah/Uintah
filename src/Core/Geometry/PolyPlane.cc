/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
 *  PolyPlane.cc: Uniform Plane containing triangular elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 */

#include <Core/Geometry/PolyPlane.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <Core/Math/Matrix3.h>
#include <iostream>
#include <tuple>

namespace Uintah {

const double PolyPlane::NEAR_ZERO = 1.0e-6;

PolyPlane::PolyPlane()
: n(Vector(0,0,1)), d(0)
{
}

PolyPlane::PolyPlane(double a, double b, double c, double d) : n(Vector(a,b,c)), d(d) {
    double l=n.length();
    d/=l;
    n.normalize();
}



PolyPlane::PolyPlane(const Point &p, const Vector &normal)
  : n(normal), d(-Dot(p, normal))
{
   n.normalize();
//  Vector centralDistance = p - origin;
//  double localD = Dot(centralDistance,n);
//  d = -localD - Dot(origin,n);
  d = -Dot(p,n);
  // double dummy = 0.0;
//  d = -Dot((p-origin).asVector(),n)+Dot(origin,n);
}

PolyPlane::PolyPlane(const PolyPlane &copy)
: n(copy.n), d(copy.d)
{
}

PolyPlane::PolyPlane(const Point &p1, const Point &p2, const Point &p3) {
    Vector v1(p2-p1);
    Vector v2(p2-p3);
    n=Cross(v2,v1);
    n.normalize();
    d=-Dot(p1, n);
}

PolyPlane::~PolyPlane() {
}

void PolyPlane::flip() {
   n.x(-n.x());
   n.y(-n.y());
   n.z(-n.z());
   d=-d;
}

double PolyPlane::eval_point(const Point &p) const {
    return Dot(p, n)+d;
}

bool PolyPlane::pointInterior(const Point &pointIn) const {
  double ProjectedDistance = -Dot(n, pointIn.asVector());
  return ((ProjectedDistance - d) < NEAR_ZERO);
}

Point PolyPlane::project(const Point& p) const
{
   return p-n*(d+Dot(p,n));
}

Vector PolyPlane::project(const Vector& v) const
{
    return v-n*Dot(v,n);
}

Vector PolyPlane::normal() const
{
   return n;
}

void
PolyPlane::ChangePlane(const Point &p1, const Point &p2, const Point &p3) {
    Vector v1(p2-p1);
    Vector v2(p2-p3);
    n=Cross(v2,v1);
    n.normalize();
    d=-Dot(p1, n);
}


void
PolyPlane::ChangePlane(const Point &P, const Vector &N) {
  //  std::cerr << N << std::endl;
  //  return;
  n = N;
  n.safe_normalize();
  d = -Dot(P,n);
}



int
PolyPlane::Intersect( Point s, Vector v, Point& hit )
{
  Point origin( 0., 0., 0. );
  Point ptOnPlane = origin - n * d;
  double tmp = Dot( n, v );

  if( tmp > -1.e-6 && tmp < 1.e-6 ) // Vector v is parallel to plane
  {
    // vector from origin of line to point on plane
      
    Vector temp = s - ptOnPlane;

    if ( temp.length() < 1.e-5 )
    {
      // the origin of plane and the origin of line are almost
      // the same
          
      hit = ptOnPlane;
      return 1;
    }
    else
    {
      // point s and d are not the same, but maybe s lies
      // in the plane anyways
          
      tmp = Dot( temp, n );

      if(tmp > -1.e-6 && tmp < 1.e-6)
      {
        hit = s;
        return 1;
      }
      else
        return 0;
    }
  }

  tmp = - ( ( d + Dot( s, n ) ) / Dot( v, n ) );

  hit = s + v * tmp;

  return 1;
}


#if 0
int
PolyPlane::Intersect( Point s, Vector v, double &t ) const
{
  Point origin( 0., 0., 0. );
  Point ptOnPlane = origin - n * d;
  double tmp = Dot( n, v );

  if( tmp > -1.e-6 && tmp < 1.e-6 ) // Vector v is parallel to plane
  {
    // vector from origin of line to point on plane
      
    Vector temp = s - ptOnPlane;

    if ( temp.length() < 1.e-5 )
    {
      // the origin of plane and the origin of line are almost
      // the same
          
      t = 0.0;
      return 1;
    }
    else
    {
      // point s and d are not the same, but maybe s lies
      // in the plane anyways
          
      tmp = Dot( temp, n );

      if(tmp > -1.e-6 && tmp < 1.e-6)
      {
        t = 0.0;
        return 1;
      }
      else
        return 0;
    }
  }

  t = - ( ( d + Dot( s, n ) ) / Dot( v, n ) );

  return 1;
}

#else
int
PolyPlane::Intersect(Point s, Vector v, double &t) const
{
  double tmp = Dot( n, v );
  if(tmp > -1.e-6 && tmp < 1.e-6) // Vector v is parallel to plane
  {
    // vector from origin of line to point on plane
    Vector temp = (s + n*d).asVector();
    if (temp.length() < 1.e-5)
    {
      // origin of plane and origin of line are almost the same
      t = 0.0;
      return 1;
    }
    else
    {
      // point s and d are not the same, but maybe s lies
      // in the plane anyways
      tmp = Dot(temp, n);
      if (tmp > -1.e-6 && tmp < 1.e-6)
      {
        t = 0.0;
        return 1;
      }
      else
        return 0;
    }
  }
  
  t = -((d + Dot(s, n)) / Dot(v, n));
  return 1;
}
#endif



void
PolyPlane::get(double (&abcd)[4]) const
{
  abcd[0] = n.x();
  abcd[1] = n.y();
  abcd[2] = n.z();
  abcd[3] = d;
}

std::tuple<bool,Point> PolyPlane::intersectWithTwoPlanes(const PolyPlane& plane2,
                                                     const PolyPlane& plane3  ) const
{
  bool    intersect2Valid;
  Vector  intersectionDirection;
  Point   intersectionOffset;

  std::tie(intersect2Valid, intersectionDirection,intersectionOffset) =
      this->intersectWithPlane(plane2);
  if (!intersect2Valid) {
    // Intersection of the first two planes failed
    return(std::make_tuple(intersect2Valid,intersectionDirection.asPoint()));
  }
  // The first two planes intersected and formed a line, now check the third.
  Point pointOnLine = -(d*n+plane2.d*plane2.n).asPoint();
  return(plane3.intersectWithLine(intersectionDirection,pointOnLine));
}

std::tuple<bool,Vector,Point> PolyPlane::intersectWithPlane(const PolyPlane& planeIn) const
{
  const Vector & n1 = n;
  const Vector & n2 = planeIn.n;
  const double & d1 = d;
  const double & d2 = planeIn.d;

  Vector n3 = Cross(n1,n2);
  double det123 = n3.length2();
  if (det123 < PolyPlane::NEAR_ZERO && det123 > -PolyPlane::NEAR_ZERO) {
    // Plane normals are almost parallel, therefore planes don't intersect
    return(std::make_tuple(false,Vector(0,0,0),Point(DBL_MAX,DBL_MAX,DBL_MAX)));
  }
  Vector pointOnLine = ((Cross(n3,n2)*d1+Cross(n1,n3)*d2))/det123;
  return(std::make_tuple(true,n3,pointOnLine.asPoint()));
}

std::tuple<bool,Point> PolyPlane::intersectWithLine(const Vector  & l,
                                                const Point   & l0) const
{
  // For all points, p:
  // Line:   p = l0 + d*l
  // Plane:  (p-p0).n  = 0

  // Intersection when both are equivalent:
  //  ((l0 + d*l) - p0).n = 0
  //   d*(l.n) + (l0 - p0).n = 0
  //   d*(l.n) = -(l0 - p0).n
  //   if (l.n) != 0 :
  //        d = -(l0 - p0).n
  //             -----------
  //                l.n

  //
  double ldotn = Dot(n,l);
  Vector p0minusl0 = -n*d - l0;
  double d = Dot(p0minusl0,n);

  if (ldotn < NEAR_ZERO && ldotn > -NEAR_ZERO) {
    // Vector is parallel to the plane; does not intersect in a -single- point.
    if (d < NEAR_ZERO && d > -NEAR_ZERO) { // Line in plane
      return(std::make_tuple(false,Point(DBL_MIN,DBL_MIN,DBL_MIN)));
    }
    // Line out of plane
    return(std::make_tuple(false,Point(DBL_MAX,DBL_MAX,DBL_MAX)));
  }
  d /= ldotn;
  Vector intersection = l0.asVector() + d*l;

  for (size_t index=0; index < 3; ++index) {
    if (intersection[index] < NEAR_ZERO && intersection[index] > -NEAR_ZERO) {
      intersection[index] = 0;
    }
  }
  return(std::make_tuple(true,intersection.asPoint()));
}

std::tuple<Vector,double> PolyPlane::getDirectionAndDistance() const {
  return(std::make_tuple(n,d));
}

std::tuple<Vector,Point> PolyPlane::getDirectionAndOffset() const {
  return(std::make_tuple(n,(-n*d).asPoint()));
}

bool PolyPlane::isInside(const Point & pointIn) const {
  double pointProjection = -Dot(pointIn.asVector(),n);
  double difference = pointProjection-d;
  if (!(difference < NEAR_ZERO)) {
    return false;
  }
  return true;
}

} // End namespace Uintah

