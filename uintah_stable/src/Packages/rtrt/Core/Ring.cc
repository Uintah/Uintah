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



#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* ring_maker() {
  return new Ring();
}

// initialize the static member type_id
PersistentTypeID Ring::type_id("Ring", "Object", ring_maker);

Ring::Ring(Material* matl, const Point& cen, const Vector& n,
	   double radius, double thickness)
  : Object(matl,this), cen(cen), n(n), radius(radius), thickness(thickness)
{
    this->n.normalize();
    d=Dot(this->n, cen);
}

Ring::~Ring()
{
}

void Ring::uv(UV& uv, const Point& hitpos, const HitInfo&)  
{
  uv.set(hitpos.x(),hitpos.y());
}

void Ring::intersect(Ray& ray, HitInfo& hit, DepthStats*,
		     PerProcessorContext*)
{
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-6 && dt > -1.e-6)
	return;
    double t=(d-Dot(n, orig))/dt;
    if(hit.was_hit && t>hit.min_t)
	return;
    Point p(orig+dir*t);
    double l=(p-cen).length2();
    double outer_radius=radius+thickness;
    if(l > radius*radius && l < outer_radius*outer_radius)
	hit.hit(this, t);
}

Vector Ring::normal(const Point&, const HitInfo&)
{
    return n;
}

void Ring::light_intersect(Ray& ray, HitInfo& hit, Color&,
			   DepthStats*, PerProcessorContext*)
{
  Vector dir(ray.direction());
  Point orig(ray.origin());
  double dt=Dot(dir, n);
  if(dt < 1.e-6 && dt > -1.e-6)
    return;
  double t=(d-Dot(n, orig))/dt;
  if(t>hit.min_t)
    return;
  Point p(orig+dir*t);
  double l=(p-cen).length2();
  double outer_radius=radius+thickness;
  if(l > radius*radius && l < outer_radius*outer_radius)
    hit.shadowHit(this, t);
}

void Ring::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(cen-Vector(1,1,1)*(offset+radius+thickness));
    bbox.extend(cen+Vector(1,1,1)*(offset+radius+thickness));
}

const int RING_VERSION = 1;

void 
Ring::io(SCIRun::Piostream &str)
{
  str.begin_class("Ring", RING_VERSION);
  Object::io(str);
  Pio(str, cen);
  Pio(str, n);
  Pio(str, d);
  Pio(str, radius);
  Pio(str, thickness);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Ring*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Ring::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Ring*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

