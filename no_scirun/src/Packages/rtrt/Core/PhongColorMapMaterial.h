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


/*
  PhongColorMapMaterial.h

    Shades a surface based on the colors and opacities obtained from
    looking up the value (value_source->interior_value()) in diffuse_terms
    and opacity_terms.

 Author: James Bigler (bigler@cs.utah.edu)
 Date: July 11, 2002
 
 */
#ifndef PHONG_COLOR_MAP_MATERIAL_H
#define PHONG_COLOR_MAP_MATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

namespace rtrt {

class PhongColorMapMaterial : public Material {
  // This object must define a meaningful interior_value funtion, in order
  // for any results to work.
  Object *value_source;

  // diffuse_transform and opacity_transform should belong to someone
  // else, preferably to someone who can edit them at runtime.
  ScalarTransform1D<float,Color> *diffuse_transform;
  ScalarTransform1D<float,float> *opacity_transform;
  int spec_coeff;
  double reflectance; // Goes from 0 to 1

public:
  PhongColorMapMaterial(Object *value_source,
			ScalarTransform1D<float,Color> *diffuse_transform,
			ScalarTransform1D<float,float> *opacity_transform,
			int spec_coeff = 100, double reflectance = 0);
  virtual ~PhongColorMapMaterial();

  virtual void io(SCIRun::Piostream &/*str*/) { ASSERTFAIL("not implemented");}
  // This function is used by some shadow routines to determine intersections
  // for shadow feelers.  Because we need the HitInfo to determine the
  // opacity, we should always return 1.
  inline double get_opacity() { return 1; }

  // This is where all the magic happens.
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
