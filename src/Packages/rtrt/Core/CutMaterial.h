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


#ifndef CUTMATERIAL_H
#define CUTMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/ColorMap.h>

/*
New Material for Cutting Planes.
If the object returns true and a value from color_interior,
this will color the interior of an object according to a ColorMap.
Otherwise it will just call surfmat to color as normal.
*/

namespace rtrt {

class CutMaterial : public Material {
  Material *surfmat; //if not outside, use this to color instead
  CutPlaneDpy *dpy;
  ColorMap *cmap;
public:
  CutMaterial(Material *surfmat, ColorMap *cmap=0, CutPlaneDpy *dpy=0);
  CutMaterial(Material *surfmat, CutPlaneDpy *dpy=0, ColorMap *cmap=0);
  virtual ~CutMaterial() {};
  virtual void io(SCIRun::Piostream &/*str*/) { ASSERTFAIL("not implemented");}
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};
 
} // end namespace rtrt
#endif
