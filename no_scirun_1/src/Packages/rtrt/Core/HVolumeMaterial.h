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


#ifndef __HVOLUMEMATHERIAL_H__
#define __HVOLUMEMATHERIAL_H__

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace rtrt {

  struct Context;
  class HitInfo;
  class Scene;
  class Ray;
  class Stats;
  class Worker;
  class VolumeDpy;

class HVolumeMaterial: public Material {
  VolumeDpy *vdpy;
  ScalarTransform1D<float,float> *f1_to_f2;
  ScalarTransform1D<float,Material*> *f2_to_material;
  
public:
  HVolumeMaterial(VolumeDpy *dpy, ScalarTransform1D<float,float> *f1_to_f2,
		  ScalarTransform1D<float,Material*> *f2_to_material);
  virtual ~HVolumeMaterial() {}
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};
  
} // end namespace rtrt

#endif // __HVOLUMEMATHERIAL_H__
