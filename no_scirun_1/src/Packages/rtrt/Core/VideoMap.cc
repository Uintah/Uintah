/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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

#include <sgi_stl_warnings_off.h>
#include   <iostream>
#include   <string>
#include <sgi_stl_warnings_on.h>

#include <cstdlib>
#include <cstring>

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
                    const Color& _specular, int _specPower, double _refl)
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
