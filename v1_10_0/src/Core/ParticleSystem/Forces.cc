//===========================================================================
//
// Filename: Forces.cc
//
// 
//
//
//
//
//
// Description: A collection of different forces
//
// Includes: Directional (ie. Gravity or Wind)
//           Turbulence
//           Vortex
//           Torus
//
//===========================================================================

#ifndef __FORCES_H__
#include <Forces.h>
#endif

#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

VortexForce::VortexForce( const Vector& pos, const Vector& dir ) :
  _dir( dir ),
  _pos( pos )
{
  _radTime = 0.0f;
  _speed = 0.2f;  
}

VortexForce::~VortexForce( void )
{
  // EMPTY
}

Vector
VortexForce::GetForce( const Vector& pos ) const
{

  float radius = 175.0f;
  Vector temp = pos - _pos;
  temp.y() = 0.0;
  
  float t = temp.Length();
  temp.Normalize();
  
  float dist = radius / t / 10.0f;
  float angle = atan2( temp.z(), temp.x() );
  float factor = 0.0f;
  t *= 1.5f;
  if( t > radius ) {
    // temp.x() = temp.y() = temp.z() = 0.0f;
    factor = radius * 1.5f / t / 10.0f;
  } else {
    factor = MAX( MIN( dist, 0.0f ), 0.0f );
  }
  temp.x() = -sin( angle ) * factor; // .1f;
  temp.z() = cos( angle ) * factor;  //.1f;
  temp.y() = 0.0f;
  
  return temp;
  
}

#define rand3a(x,y,z)	frand(67*(x)+59*(y)+71*(z))
#define rand3b(x,y,z)	frand(73*(x)+79*(y)+83*(z))
#define rand3c(x,y,z)	frand(89*(x)+97*(y)+101*(z))
#define rand3d(x,y,z)	frand(103*(x)+107*(y)+109*(z))

static int	xlim[3][2];		// integer bound for point
float	xarg[3];		// fractional part

float frand( register int s )   // get random number from seed
{
	s = s << 13^s;
	return(1. - ((s*(s*s*15731+789221)+1376312589)&0x7fffffff)/1073741824.);
}

float hermite( float p0, float p1, float r0, float r1, float t )
{
	register float t2, t3, _3t2, _2t3 ;
	t2 = t * t;
	t3 = t2 * t;
	_3t2 = 3. * t2;
	_2t3 = 2. * t3 ;

	return(p0*(_2t3-_3t2+1) + p1*(-_2t3+_3t2) + r0*(t3-2.*t2+t) + r1*(t3-t2));
}

void interpolate( float f[4], register int i, register int n )
//
//	f[] returned tangent and value *
//	i   location ?
//	n   order
//
{
	float f0[4], f1[4] ;  //results for first and second halves

	if( n == 0 )	// at 0, return lattice value
	{
		f[0] = rand3a( xlim[0][i&1], xlim[1][i>>1&1], xlim[2][i>>2] );
		f[1] = rand3b( xlim[0][i&1], xlim[1][i>>1&1], xlim[2][i>>2] );
		f[2] = rand3c( xlim[0][i&1], xlim[1][i>>1&1], xlim[2][i>>2] );
		f[3] = rand3d( xlim[0][i&1], xlim[1][i>>1&1], xlim[2][i>>2] );
		return;
	}

	n--;
	interpolate( f0, i, n );		// compute first half
	interpolate( f1, i| 1<<n, n );	// compute second half

	// use linear interpolation for slopes
	//
	f[0] = (1. - xarg[n]) * f0[0] + xarg[n] * f1[0];
	f[1] = (1. - xarg[n]) * f0[1] + xarg[n] * f1[1];
	f[2] = (1. - xarg[n]) * f0[2] + xarg[n] * f1[2];

	// use hermite interpolation for values
	//
	f[3] = hermite( f0[3], f1[3], f0[n], f1[n], xarg[n] );
}

void
TorusForce::NoiseFunction( float *inNoise, float *out )
//
//	Descriptions:
//		A noise function.
//
{
	xlim[0][0] = (int)floor( inNoise[0] );
	xlim[0][1] = xlim[0][0] + 1;
	xlim[1][0] = (int)floor( inNoise[1] );
	xlim[1][1] = xlim[1][0] + 1;
	xlim[2][0] = (int)floor( inNoise[2] );
	xlim[2][1] = xlim[2][0] + 1;

	xarg[0] = inNoise[0] - xlim[0][0];
	xarg[1] = inNoise[1] - xlim[1][0];
	xarg[2] = inNoise[2] - xlim[2][0];

	interpolate( out, 0, 3 ) ;
}


TorusForce::TorusForce( const Vector& center ) :
  _center( center ),
  _applyMaxDistance( true )
{
  // EMPTY
}

TorusForce::~TorusForce( void )
{

}

Vector
TorusForce::GetForce( const Vector& pos ) const
{
  if( _applyMaxDistance )
    return ApplyMaxDistance( pos );
  else
    return ApplyNoMaxDistance( pos );
}
  

Vector
TorusForce::ApplyMaxDistance( const Vector& pos ) const
{

}

Vector
TorusForce::ApplyNoMaxDistance( const Vector& pos ) const
{

}

DragForce::~DragForce( void )
{
  // EMPTY
}

Vector
DragForce::GetForce( const Vector& pos ) const
{
  if( !_hasVel )
    return Vector( _drag, _drag, _drag );
  else
    return( _drag * _dir );
}

Vector
DragForce::GetForce( const Particle* par ) const
{
  if( !_hasVel )
    return( _drag * par->GetVelocity() );
  else
    return( ( par->GetVelocity() * _dir ) * _drag * par->GetVelocity() );
}
