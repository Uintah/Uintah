
/**********************************************************************\
 *                                                                    *
 * filename: VideoMap.cc                                              *
 * author  : R. Keith Morley                                          *
 * last mod: 07/07/02                                                 *
 *                                                                    *
\**********************************************************************/ 

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/VideoMap.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Thread/Time.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

/********************************************************************/
VideoMap::VideoMap () 
{ 
   fprintf(stderr, "\n WARNING -- Default VideoMap constructor used!\n\n"); 
}

/********************************************************************/
VideoMap::VideoMap (char* fileName, int _numFrames, double _framesPerSec, 
                    const Color& _specular, double _specPower, double _refl)
   : numFrames(_numFrames), framesPerSec(_framesPerSec), specular(_specular), 
     specPower(_specPower), refl(_refl)
{
   std::cerr << "/nInitializing video texture ... " << endl;
   
   frames.resize(numFrames);
   curFrame = 0;
   loopTime = numFrames / framesPerSec;

      
   /* read in all of the ppm files and store them in frames */
   char buffer[1024];
   for (int i = 0; i < numFrames; i++)
   {
      /* load the filename into a string */
      sprintf(buffer, fileName, i);
      unsigned long length = strlen(buffer);
      string file;
      for (unsigned long j = 0; j < length; j++)
      {
	 file += buffer[j];
      }

      /* now read in the file and store */
      std::cerr << "  Reading image " << file << " ... ";
      PPMImage ppm (file);
      if (ppm.valid()) 
      {
	 std::cerr << " succeeded\n";
      }
      else
      {
	 std::cerr << " FAILED!\n";
	 exit(-1);
      }
      /* get dimensions of new image */
      int _nx = ppm.get_width();
      int _ny = ppm.get_height();
      if (i == 0)
      {
	 nx = _nx;
	 ny = _ny;
      }
      else /* check to see if dimensions match previous */
      {
	 if (_nx != nx) 
	 {
	    std::cerr << " ERROR -- Image \"" 
	              << file 
		      << "\" has diff dimensions than rest ... aborting!\n\n";
	    exit(-1);
	 }
      }
      /* load image into frames array */		      
      Array2<Color> * tempArray = new Array2<Color>(nx, ny);
      ppm.get_dimensions_and_data(*tempArray, nx, ny);
      frames[i] = tempArray;
      
   }
   std::cerr << endl;
}
   
/********************************************************************/
VideoMap::~VideoMap ()
{}

/********************************************************************/
// function borrowed from ImageMaterial
Color interp_color(Array2<Color>& image, double u, double v)
{
   u *= (image.dim1()-1);
   v *= (image.dim2()-1);

   int iu = (int)u;
   int iv = (int)v;

   double tu = u-iu;
   double tv = v-iv;

   Color c = image(iu,iv)*(1-tu)*(1-tv)+
             image(iu+1,iv)*tu*(1-tv)+
             image(iu,iv+1)*(1-tu)*tv+
             image(iu+1,iv+1)*tu*tv;

   return c;
}

/********************************************************************/
void VideoMap::shade (Color& result, const Ray& ray, const HitInfo& hit,
                      int depth, double atten, const Color& accumcolor,
                      Context* cx) 
{
   /* find uv coords of hit position */
   UVMapping* map; 
   map = hit.hit_obj->get_uvmapping();
   UV uv;
   Point hitpos(ray.origin()+ray.direction()*hit.min_t);
   map->uv(uv, hitpos, hit);
   double u=uv.u();
   double v=uv.v();

   /* find which image to map */
   int index = 0;
   double time = SCIRun::Time::currentSeconds();
   time = fmodf(time, loopTime);  /* this is how far into video we are */
   index =  int(time * framesPerSec);
   if (index > numFrames) index = numFrames;

   /* get diffuse color from image and add phong lighting */
   Color diffuse = interp_color(*(frames[index]), u, v);
   phongshade(result, diffuse, specular, specPower, refl,
              ray, hit, depth,  atten,
              accumcolor, cx);
}
