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



#ifndef VOLUMEVISBASE_H
#define VOLUMEVISBASE_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>

namespace rtrt {
class VolumeVisBase;
class VolumeVisDpy;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::VolumeVisBase*&);
}

namespace rtrt {

class VolumeVisBase : public Object, public Material {
protected:
  Object* child;

  VolumeVisDpy* dpy;

  inline int clamp(const int min, const int val, const int max) {
    return (val>min?(val<max?val:max):min);
  }
public:
  VolumeVisBase(VolumeVisDpy* dpy);
  virtual ~VolumeVisBase();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, VolumeVisBase*&);

  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_hist(int nhist, int* hist,
			    float datamin, float datamax)=0;
  virtual void get_minmax(float& min, float& max)=0;
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx) = 0;

  void set_child(Object* kiddo) { child = kiddo; }
};

} // end namespace rtrt

#endif
