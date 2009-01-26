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



#ifndef TextureGridSpheres_H
#define TextureGridSpheres_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/GridSpheres.h>

namespace rtrt {

  class UV;
  
class TextureGridSpheres : public GridSpheres {
protected:
  int *tex_indices; // length of spheres
  unsigned char *tex_data;
  size_t ntextures;
  int tex_res;
  Color color;
  
  void get_uv(UV& uv, const Point& hitpos, const HitInfo& hit);
  float interp_luminance(unsigned char *image, double u, double v);

public:
  TextureGridSpheres(float* spheres, size_t nspheres, int ndata,
		     float radius,
		     int *tex_indices,
		     unsigned char* tex_data, size_t ntextures, int tex_res,
		     int nsides, int depth, RegularColorMap* cmap = 0,
		     const Color& color = Color(1.0, 1.0, 1.0));
  virtual ~TextureGridSpheres();
  virtual void io(SCIRun::Piostream &stream);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
