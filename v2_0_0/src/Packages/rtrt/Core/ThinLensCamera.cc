//===================================================================================
//
// Filename: ThinLensCamera.cc
//
// 
//
//
//
// Description: Implementation of camera with depth-of-field
//
// HAs the same API as normal Camera except that it takes extra parameters
// such as focal length, f-stop, etc.
//
//==================================================================================

#include <Packages/rtrt/Core/ThinLensCamera.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Transform.h>
#include <iostream>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;

ThinLensCamera::ThinLensCamera( const Point& eye, const Point& lookat,
				const Vector& up, double fov,
				double aRatio,
				double dFocus,
				double fLength,
				double fn ) :
  Camera( eye, lookat, up, fov ),
  aspectRatio( aRatio ),
  focalLength( fLength ),
  fnumber( fn )
{
  distanceToFocus = dFocus;
  setup();
}
  
ThinLensCamera::~ThinLensCamera( void )
{
  // EMPTY
}

// This transforms points on [0,1]^2 to points on unit disk centered at
// origin.  Each "pie-slice" quadrant of square is handled as a seperate
// case.  The bad floating point cases are all handled appropriately.
// The regions for (a,b) are:
//
//        phi = pi/2
//       -----*-----
//       |\       /|
//       |  \ 2 /  |
//       |   \ /   |
// phi=pi* 3  *  1 *phi = 0
//       |   / \   |
//       |  / 4 \  |
//       |/       \|
//       -----*-----
//        phi = 3pi/2

const double PiOver4 = M_PI / 4.0;

void ConcentricDiskWarp( double seedx, double seedy, double& X, double& Y )
{
  double phi, r;

  double a = 2.0*seedx - 1;   // (a,b) is now on [-1,1]^2
  double b = 2.0*seedy - 1;

  if( a > -b ) {     // region 1 or 2
    if( a > b ) {  // region 1, also |a| > |b|
      r = a;
      phi = PiOver4 * (b/a);
    }
    else       {  // region 2, also |b| > |a|
      r = b;
      phi = PiOver4 * (2 - (a/b));
    }
  }
  else {        // region 3 or 4
    if (a < b) {  // region 3, also |a| >= |b|, a != 0
      r = -a;
      phi = PiOver4 * (4 + (b/a));
    }
    else       {  // region 4, |b| >= |a|, but a==0 and b==0 could occur.
      r = -b;
      if (b != 0)
	phi = PiOver4 * (6 - (a/b));
      else
	phi = 0;
    }
  }
  
  X = r * cos( phi );
  Y = r * sin( phi );
  
}

void
ThinLensCamera::makeRay( Ray& ray, double x, double y, double ixres, double iyres )
{
  double screenx=(x+0.5)*ixres-0.5;
  double screeny=(y+0.5)*iyres-0.5;

  double bu = uvMin.x() + screenx*uvSize.x();
  double bv = uvMin.y() + screeny*uvSize.y();
  double X, Y;
  //
  // Need to optimize this: store random numbers in an array
  //
  double seedx = drand48();
  double seedy = drand48();
  ConcentricDiskWarp( seedx, seedy, X, Y );
  X *= lensRadius;
  Y *= lensRadius;
  ray.set_origin( eye + X*uhat + Y*vhat );
  //
  // This can be optimized as well
  //
  Vector toWindow = ( bu - X ) * uhat + ( bv - Y ) * vhat + distanceToFocus * what;
  
  toWindow.normalize();
  ray.set_direction( toWindow );
}

void
ThinLensCamera::makeRayL(Ray& ray, double x, double y, double xres, double yres)
{
  // NOT IMPLEMENTED YET
  assert( 0 );
}

void
ThinLensCamera::makeRayR(Ray& ray, double x, double y, double xres, double yres)
{
  // NOT IMPLEMENTED YET
  assert( 0 );
}

void
ThinLensCamera::setup( void )
{
  Camera::setup();
  lensRadius = focalLength / ( 2.0 * fnumber );
  double scale = ( distanceToFocus - focalLength ) / focalLength;
  double halfWidth = distanceToFocus * tan( M_PI * fov / 360.0 );
  double halfHeight = halfWidth / aspectRatio;
  uvMin = scale * Vector( -halfWidth, -halfHeight, 0.0 );
  uvMax = scale * Vector( halfWidth, halfHeight, 0.0 );
  uvSize = uvMax - uvMin;
}

void
ThinLensCamera::print( void )
{
  Camera::print();
  // NOT IMPLEMENTED YET
}

void
ThinLensCamera::getParams( Point& origin, Vector& lookdir,
			   Vector& up, double& vfov,
			   double& fLength, double& fDistance,
			   double& fn )
{
  origin = eye;
  lookdir = direction;
  up = u;
  vfov = fov;
  fn = fnumber;
  fDistance = distanceToFocus;
  fLength = focalLength;
}

void
ThinLensCamera::set_fnumber( double fn )
{
  fnumber = fn;
}

void
ThinLensCamera::set_focusdistance( double d )
{
  distanceToFocus = d;
}

void
ThinLensCamera::set_focallength( double d )
{
  focalLength = d;
}
