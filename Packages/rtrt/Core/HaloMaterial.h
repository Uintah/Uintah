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



#ifndef HALOMATERIAL_H
#define HALOMATERIAL_H 1

#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <cmath>

namespace rtrt {
class HaloMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::HaloMaterial*&);
}

namespace rtrt {

class HaloMaterial : public Material 
{

 protected:

  InvisibleMaterial transparent_;
  Material          *fg_;
  double            pow_;

 public:

  HaloMaterial(Material *fg, double pow) 
    : transparent_(), fg_(fg), pow_(pow) {}
  virtual ~HaloMaterial() {}

  HaloMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, HaloMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
                     const HitInfo& hit, int depth,
                     double atten, const Color& accumcolor,
                     Context* cx)
  {
    Object* obj=hit.hit_obj;
    Point hitpos(ray.origin()+ray.direction()*hit.min_t);
    double percent=-Dot(obj->normal(hitpos,hit), ray.direction());
    //if (percent<0) percent=0;
    percent = pow(percent,pow_);
    fg_->shade(result,ray,hit,depth,atten,accumcolor,cx);
    Color fg = result;
    transparent_.shade(result,ray,hit,depth,atten,accumcolor,cx);
    Color bg = result;
    result = (fg*percent)+(bg*(1.-percent));
  }
};

} // end namespace

#endif
