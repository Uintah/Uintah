#include "VolumeUtils.h"

namespace Kurt {
using namespace SCIRun;

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
} // End namespace Kurt


