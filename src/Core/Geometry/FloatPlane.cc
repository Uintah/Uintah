/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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
 *  FloatPlane.cc: Uniform FloatPlane containing triangular elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 */

#include <Core/Geometry/FloatPlane.h>
#include <Core/Geometry/FloatPoint.h>
#include <Core/Geometry/FloatVector.h>
#include <iostream>

namespace Uintah {

FloatPlane::FloatPlane()
: n(FloatVector(0,0,1)), d(0)
{
}

FloatPlane::FloatPlane(float a, float b, float c, float d) : n(FloatVector(a,b,c)), d(d) {
    float l=n.length();
    d/=l;
    n.normalize();
}



FloatPlane::FloatPlane(const FloatPoint &p, const FloatVector &normal)
  : n(normal), d(-Dot(p, normal))
{
}

FloatPlane::FloatPlane(const FloatPlane &copy)
: n(copy.n), d(copy.d)
{
}

FloatPlane::FloatPlane(const FloatPoint &p1, const FloatPoint &p2, const FloatPoint &p3) {
    FloatVector v1(p2-p1);
    FloatVector v2(p2-p3);
    n=Cross(v2,v1);
    n.normalize();
    d=-Dot(p1, n);
}

FloatPlane::~FloatPlane() {
}

void FloatPlane::flip() {
   n.x(-n.x());
   n.y(-n.y());
   n.z(-n.z());
   d=-d;
}

float FloatPlane::eval_point(const FloatPoint &p) const {
    return Dot(p, n)+d;
}

FloatPoint FloatPlane::project(const FloatPoint& p) const
{
   return p-n*(d+Dot(p,n));
}

FloatVector FloatPlane::project(const FloatVector& v) const
{
    return v-n*Dot(v,n);
}

FloatVector FloatPlane::normal() const
{
   return n;
}

void
FloatPlane::ChangePlane(const FloatPoint &p1, const FloatPoint &p2, const FloatPoint &p3) {
    FloatVector v1(p2-p1);
    FloatVector v2(p2-p3);
    n=Cross(v2,v1);
    n.normalize();
    d=-Dot(p1, n);
}


void
FloatPlane::ChangePlane(const FloatPoint &P, const FloatVector &N) {
  //  std::cerr << N << std::endl;
  //  return;
  n = N;
  n.safe_normalize();
  d = -Dot(P,n);
}



int
FloatPlane::Intersect( FloatPoint s, FloatVector v, FloatPoint& hit )
{
  FloatPoint origin( 0., 0., 0. );
  FloatPoint ptOnPlane = origin - n * d;
  float tmp = Dot( n, v );

  if( tmp > -1.e-6 && tmp < 1.e-6 ) // Vector v is parallel to plane
  {
    // vector from origin of line to point on plane
      
    FloatVector temp = s - ptOnPlane;

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
FloatPlane::Intersect( FloatPoint s, FloatVector v, float &t ) const 
{
  FloatPoint origin( 0., 0., 0. );
  FloatPoint ptOnPlane = origin - n * d;
  float tmp = Dot( n, v );

  if( tmp > -1.e-6 && tmp < 1.e-6 ) // FloatVector v is parallel to plane
  {
    // vector from origin of line to point on plane
      
    FloatVector temp = s - ptOnPlane;

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
FloatPlane::Intersect(FloatPoint s, FloatVector v, float &t) const
{
  float tmp = Dot( n, v );
  if(tmp > -1.e-6 && tmp < 1.e-6) // FloatVector v is parallel to plane
  {
    // vector from origin of line to point on plane
    FloatVector temp = (s + n*d).asVector();
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
FloatPlane::get(float (&abcd)[4]) const
{
  abcd[0] = n.x();
  abcd[1] = n.y();
  abcd[2] = n.z();
  abcd[3] = d;
}

} // End namespace Uintah

