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


#ifndef DUMMYOBJECT_H
#define DUMMYOBJECT_H

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace rtrt { 

using SCIRun::Vector;
using SCIRun::Point;

class DummyObject : public Object
{
  Object* obj;
  Vector n;

public:
    
  DummyObject(Object* obj, Material* m) 
    : Object(m), obj(obj)
  {}
  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }
  void SetNormal(Vector& v)
  {
    n = v;
  }
  void SetObject(Object* o)
  {
    obj=o;
  }
  virtual void compute_bounds(BBox& bbox, double offset)
  {
    obj->compute_bounds(bbox, offset);
  }
    
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext* ppc)
  {
    obj->intersect(ray, hit, st, ppc);
  }
    
  virtual void light_intersect(Ray& ray,
			       HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc)
  {
    obj->light_intersect(ray, hit, atten, st, ppc);
  }
    
  virtual Vector normal(const Point&, const HitInfo& )
  {
    return n;
  }
    
};
} // end namespace rtrt

#endif

    
