//===================================================================================
//
// Filename: ThinLensCamera.h
//
// 
//
//
//
// Description: Camera with depth-of-field
//
// HAs the same API as normal Camera except that it takes extra parameters
// such as focal length, f-stop, etc.
//
//==================================================================================

#ifndef __THINLENSCAMERA_H__
#define __THINLENSCAMERA_H__ 1

#ifndef CAMERA_H
#include <Packages/rtrt/Core/Camera.h>
#endif

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;

class Ray;
class Stealth;
class Scene;
class PerProcessorContext;

class ThinLensCamera : public Camera
{

public:

  ThinLensCamera( const Point& eye, const Point& lookat,
		  const Vector& up, double fov,
		  double aspectRatio,
		  double distanceToFocus,
		  double focalLength,
		  double fnumber );
  //
  // aspectRatio -- aspect ratio of the camera
  // distanceToFocus -- distance to objects in perfect focus
  // focalLength -- focal length of the camera lens
  // f-number -- Ratio of focal length to the diameter of the lens
  //
  //======================================================================
  
  ThinLensCamera( void );
  virtual ~ThinLensCamera( void );
  void makeRay(Ray& ray, double x, double y, double xres, double yres);
  void makeRayL(Ray& ray, double x, double y, double xres, double yres);
  void makeRayR(Ray& ray, double x, double y, double xres, double yres);
  void setup( void );
  void print();

  void getParams( Point& origin, Vector& lookdir,
		  Vector& up, double& fov,
		  double& focalLength, double& focusDistance,
		  double& fnumber );
  
  double get_fnumber( void ) const { return fnumber; }
  double get_focusdistance( void ) const { return focusDistance; }
  double get_focallength( void ) const { return focalLength; }

  void set_fnumber( double fn );
  void set_focusdistance( double d );
  void set_focallength( double d );
  
private:

  double aspectRatio;
  double distanceToFocus;
  double lensRadius;
  double focalLength;    // Distance from image plane to Center of Projection
  double fnumber;        // Ratio focal length vs. aperture diameter
  double focusDistance;  // Distance to object in sharp focus
  Vector uvMin, uvMax, uvSize;

};

} // end namespace rtrt

#endif
