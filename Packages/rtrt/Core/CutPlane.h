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



#ifndef CUTPLANE_H
#define CUTPLANE_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BBox.h>

namespace rtrt {

class PlaneDpy;
  
class CutPlane : public Object {
  Object* child;
  Point cen;
  Vector n;
  double d;
  PlaneDpy* dpy;
  BBox child_bbox;
  bool active;
  bool use_material;
public:
  CutPlane(Object* child, const Point& cen, const Vector& n);
  CutPlane(Object* child, PlaneDpy* dpy);
  CutPlane(Object* child, const Vector& n, const double d);
  virtual ~CutPlane();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
                         PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double radius, int& pp_offset, int& scratchsize);
  void update_displacement(double newd) { d = newd; }
  void update_normal(const Vector &newn) { n = newn; }
  void update_active_state(const bool newstate) { active = newstate; }
  void update_usemat_state(const bool newstate) { use_material = newstate; }
};

} // end namespace rtrt

#endif
