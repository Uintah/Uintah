#ifndef SCI_Wangxl_Datatypes_Mesh_Math_h
#define SCI_Wangxl_Datatypes_Mesh_Math_h

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/Exact.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Defs.h>

namespace Wangxl {

using namespace SCIRun;

Comparison_result compare_x(const Point &p, const Point &q)
{
  if ( p.x() > q.x() ) return LARGER;
  else if ( p.x() < q.x() ) return SMALLER;
  else {
    assert( p.x() == q.x() );
    return EQUAL;
  }
}

Comparison_result compare_y(const Point &p, const Point &q)
{
  if ( p.y() > q.y() ) return LARGER;
  else if ( p.y() < q.y() ) return SMALLER;
  else {
    assert( p.y() == q.y() );
    return EQUAL;
  }
}

Comparison_result compare_z(const Point &p, const Point &q)
{
  if ( p.z() > q.z() ) return LARGER;
  else if ( p.z() < q.z() ) return SMALLER;
  else {
    assert( p.z() == q.z() );
    return EQUAL;
  }
}

bool equal(const Point &p, const Point &q)
{
  return ( p.x() == q.x() && p.y() == q.y() && p.z() == q.z() );
}

// check s is at POSITIVE side, NEGATIVE side or COPLANAR to the Plane formed by p q r
Orientation
orientation(const Point &p, const Point &q, const Point &r, const Point &s)
{
  double a[3], b[3], c[3], d[3];
  a[0] = p.x(); a[1] = p.y(); a[2] = p.z();
  b[0] = q.x(); b[1] = q.y(); b[2] = q.z();
  c[0] = r.x(); c[1] = r.y(); c[2] = r.z();
  d[0] = s.x(); d[1] = s.y(); d[2] = s.z();

  double ret = orient3d(a,b,c,d);
  if ( ret > 0 ) return NEGATIVE;//POSITIVE; changed because of the direction
  else if ( ret < 0 ) return POSITIVE;//NEGATIVE;
  else {
    assert( ret == 0 ); 
    return COPLANAR;
  }
}

Orientation 
orientation(const Point &p, const Point &q, const Point &r)
{
  double a[2], b[2], c[2];
  double ret;

  a[0] = p.x(); a[1] = p.y(); 
  b[0] = q.x(); b[1] = q.y();
  c[0] = r.x(); c[1] = r.y();
  ret = orient2d(a,b,c);
  if ( ret != 0 ) { // not COLLINEAR in XY plane
    if ( ret < 0 ) return NEGATIVE;
    else { 
      assert( ret > 0 ); 
      return POSITIVE;
    }
  }
  a[0] = p.y(); a[1] = p.z();
  b[0] = q.y(); b[1] = q.z();
  c[0] = r.y(); c[1] = r.z();
  ret = orient2d(a,b,c);
  if ( ret != 0 ) { // not COLLINEAR in YZ plane
    if ( ret < 0 ) return NEGATIVE;
    else { 
      assert( ret > 0 ); 
      return POSITIVE;
    }
  }
  a[0] = p.x(); a[1] = p.z(); 
  b[0] = q.x(); b[1] = q.z();
  c[0] = r.x(); c[1] = r.z();
  ret = orient2d(a,b,c); // test on XZ plane
  if ( ret > 0 ) return  POSITIVE;
  else if ( ret < 0 ) return NEGATIVE;
  else {
    assert( ret == 0 ); 
    return COLLINEAR;
  }
}

bool collinear(const Point &p, const Point &q, const Point &r)
{
  return orientation(p, q, r) == COLLINEAR;
}

Oriented_side
side_of_oriented_sphere(const Point &p, const Point &q, const Point &r,
	                  const Point &s, const Point &t)
{
  double a[3], b[3], c[3], d[3], e[3];
  a[0] = p.x(); a[1] = p.y(); a[2] = p.z();
  b[0] = q.x(); b[1] = q.y(); b[2] = q.z();
  c[0] = r.x(); c[1] = r.y(); c[2] = r.z();
  d[0] = s.x(); d[1] = s.y(); d[2] = s.z();
  e[0] = t.x(); e[1] = t.y(); e[2] = t.z();
  double ret = insphere(a,b,c,d,e);
  if ( ret > 0 ) return ON_NEGATIVE_SIDE;//ON_POSITIVE_SIDE;
  else if ( ret < 0 ) return ON_POSITIVE_SIDE;//ON_NEGATIVE_SIDE;
  else { 
    assert( ret == 0 ); 
    return ON_ORIENTED_BOUNDARY;
  }
}

Bounded_side
side_of_bounded_circle(const Point &p, const Point &q, const Point &r, const Point &t)
{
  Point s;
  Vector pq, pr, v;
  pq = q-p;
  pr = r-p;
  v = Cross(pq,pr);
  s = t+v;
  return (Bounded_side)side_of_oriented_sphere(p,q,r,s,t);
}

}

#endif


