
/**********************************************************************\
 *                                                                    *
 * filename: VideoMap.h                                               *
 * author  : R. Keith Morley                                          *
 * last mod: 07/07/02                                                 *
 *                                                                    *
 * VideoMap inherits from class Material.  It reads in an array of    *
 * ppm files and indexes into them based on the program running       *
 * time in order to get a time varying video tex.                     *
 *                                                                    *
\**********************************************************************/

#ifndef _VIDEOMAP_H_
#define _VIDEOMAP_H_ 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/Color.h>
#include <stdio.h>

namespace rtrt 
{

class VideoMap : public Material
{
private:
  int nx;                             // pixel width of frames
  int ny;                             // pixel height of frames
  int numFrames;                      // number of image files to loop
  int curFrame;                       // frame index at last intersection
  double loopTime;                    // time for one complete video loop
  double framesPerSec;                // variable frame speed
  double curTime;                     // time at last intersect
  double specPower;                   // area of highlight
  double refl;                        // 
  Color specular;                     // highlight color
  Array1< Array2 <Color> * > frames;  // array of images to loop
   
public:   
  /*******************************************************************/
  // do not use!
  VideoMap ();
  /*******************************************************************/
  // fileName must be in format "name%d.ppm" 
  // files must be named in format --
  //      named name0.ppm , name1.ppm , ... ,  name[numFrames-1].ppm
  VideoMap (char* _fileName, int _numFrames, double _framesPerSec,
	    const Color& _specular, double _specPower, double _refl);
  ~VideoMap ();
  /*******************************************************************/
  // looks up diffuse color in current frame,  then calls default 
  // Material::PhongShade()
  virtual void shade (Color& result, const Ray& ray, const HitInfo& hit, 
		      int depth, double atten, const Color& accumcolor,
		      Context* cx); 	    
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
};   

} // namespace rtrt
  
#endif // _VIDEOMAP_H_
