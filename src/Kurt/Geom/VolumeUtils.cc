#include "VolumeUtils.h"

namespace Kurt {
namespace GeomSpace {

using namespace SCICore::Geometry;

bool
isPowerOf2( int range)
{
  int i;
  unsigned int val;
  for( i = 31; i >= 0; i--)
    {
      val = range & ( 1 << i );
      if( val )
	break;
    }
  if( range & ~( ~0 << i ) ){
    return false;
  } else {
    return true;
  }
}

int nextPowerOf2( int range )
{
  int i;
  unsigned int val;
  for( i = 31; i >= 0; i--)
    {
      val = range & ( 1 << i );
      if( val ){
	val = (1 << i+1);
        break;
      }
    }
  return val;
}

int largestPowerOf2( int range )
{
  int i;
  unsigned int val;
  for( i = 31; i >= 0; i--)
    {
      val = range & ( 1 << i );
      if( val )
        break;
    }
  return val;
}


double intersectParam(const Vector& N, const Point& P, const Ray& R)
{
  // Computes the ray parameter t at which the ray R will
  // intersect the plane specified by the normal N and the 
  // point P

  /*  Dot(N, ((O + t0*V) - P)) = 0   solve for t0 */

  Point O = R.origin();
  Vector V = R.direction();
  double D = -(N.x()*P.x() + N.y()*P.y() + N.z()*P.z());
  double NO = (N.x()*O.x() + N.y()*O.y() + N.z()*O.z());

  double NV = Dot(N,V);
  if( NV == 0 ) {  /* No Intersection, plane is parallel */
    return -1.0;
  } else {
    return -(D + NO)/NV;
  }
}


void
sortParameters( double *t, int len_t )
{
  // sorts ray parameters from largest to smallest
  int i,j;
  double tmp;
  for(j = 0; j < len_t; j++){
    for(i = j+1; i < len_t; i++){
      if( t[j] < t[i] ){
	tmp = t[i];
	t[i] = t[j];
	t[j] = tmp;
      }
    }
  }
}


bool overlap(const BBox& b1, const BBox& b2){
  if( b1.inside( b2.min()) || b1.inside( b2.max()) )
    return true;
  else {
    Point p1(b2.min().x(), b2.min().y(), b2.max().z());
    Point p2(b2.min().x(), b2.max().y(), b2.min().z());
    Point p3(b2.max().x(), b2.min().y(), b2.min().z());
    Point p4(b2.min().x(), b2.max().y(), b2.max().z());
    Point p5(b2.max().x(), b2.min().y(), b2.max().z());
    Point p6(b2.max().x(), b2.max().y(), b2.min().z());
    
    if( b1.inside(p1) || b1.inside(p2) || b1.inside(p3) ||
	b1.inside(p4) || b1.inside(p5) || b1.inside(p6))
      return true;
    else
      return false;
  }
}

} // end namespace GeomSpace
} // end namespace Kurt

