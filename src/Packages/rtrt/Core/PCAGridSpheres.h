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



#ifndef PCAGridSpheres_H
#define PCAGridSpheres_H 1

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/TextureGridSpheres.h>

namespace rtrt {

class PCAGridSpheres : public TextureGridSpheres {
  unsigned char *coeff; // size is nvecs*nbases
  unsigned char *mean; // mean texture of size tex_res*tex_res
  int nbases; // represents the number of textures in tex_data;
  int nvecs; // represents the number of vectors having valid coefficients
  float tex_min, tex_diff; // used to unquantize the basis texture
  float coeff_min, coeff_diff; // used to unquantize the coefficients 

  // From TextureGridSpheres
  
  // int *tex_indices; // length of nspheres*3, output [0..nchannels-1]
  // unsigned char *tex_data; // size = nbases * tex_res * tex_res;
  // size_t ntextures; // not used directly

  float get_pixel(int x, int y, int channel_index);
  float interp_luminance(double u, double v, int index);
public:
  PCAGridSpheres(float* spheres, size_t nspheres, int ndata,
		 float radius,
		 int *tex_indices,
		 unsigned char* tex_data, int nbases, int tex_res,
		 unsigned char *coeff, unsigned char *mean, int ndims,
		 float tex_min, float tex_max,
                 float coeff_min, float coeff_max,
		 int nsides, int depth, RegularColorMap* cmap = 0,
		 const Color& color = Color(1.0, 1.0, 1.0));

  virtual ~PCAGridSpheres();
  virtual void io(SCIRun::Piostream &stream);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
