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



#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* disc_maker() {
  return new Disc();
}

// initialize the static member type_id
PersistentTypeID Disc::type_id("Disc", "Object", disc_maker);

Disc::Disc(Material* matl, const Point& cen, const Vector& n,
	   double radius)
  : Object(matl), cen(cen), n(n), radius(radius), tex_scale(Vector(1,1,1))
{
  this->n.normalize();
}

Disc::~Disc()
{
}

void Disc::preprocess(double, int&, int&)
{
  // Set up unit transformation
  xform.load_identity();
  xform.rotate(Vector(0,0,1), n);
  xform.pre_translate(cen.asVector());
}

void Disc::intersect(Ray& ray, HitInfo& hit, DepthStats*,
		     PerProcessorContext*)
{
  Vector xdir(xform.unproject(ray.direction()));
  Point xorig(xform.unproject(ray.origin()));
  double dt=xdir.z();
  if(dt < 1.e-6 && dt > -1.e-6)
    return;
  double t=-xorig.z()/dt;
  if(hit.was_hit && t>hit.min_t)
    return;
  Point xp(xorig+xdir*t);
  double l=xp.y()*xp.y()+xp.x()*xp.x();
  if(l < radius*radius)
    hit.hit(this, t);
}

Vector Disc::normal(const Point&, const HitInfo&)
{
    return n;
}

void Disc::light_intersect(Ray& ray, HitInfo& hit, Color&,
			   DepthStats*, PerProcessorContext*)
{
  Vector xdir(xform.unproject(ray.direction()));
  Point xorig(xform.unproject(ray.origin()));
  double dt=xdir.z();
  if(dt < 1.e-6 && dt > -1.e-6)
    return;
  double t=-xorig.z()/dt;
  if(hit.was_hit && t>hit.min_t)
    return;
  Point xp(xorig+xdir*t);
  double l=xp.y()*xp.y()+xp.x()*xp.x();
  if(l < radius*radius)
    hit.shadowHit(this, t);
}

void Disc::compute_bounds(BBox& bbox, double offset)
{
  Vector v(1,0,0);
  Vector v2, v3;
  v2=Cross(n,v);
  if (v2.length2()<1.e-8) {
    v=Vector(0,1,0);
    v2=Cross(n,v);
  }
  v2.normalize();
  v3=Cross(n,v2);
  v3.normalize();
  bbox.extend(cen-v2*(1.74*radius+offset));
  bbox.extend(cen+v2*(1.74*radius+offset));
  bbox.extend(cen+n*offset);
  bbox.extend(cen-n*offset);
  bbox.extend(cen-v3*(1.74*radius+offset));
  bbox.extend(cen+v3*(1.74*radius+offset));
}

void Disc::uv(UV &uv, const Point &p, const HitInfo &/*hit*/)
{
  Point xp = xform.project(p);
  uv.set(xp.x()/tex_scale.x(),xp.y()/tex_scale.y());
}

const int DISC_VERSION = 1;

void 
Disc::io(SCIRun::Piostream &str)
{
  str.begin_class("Disc", DISC_VERSION);
  Object::io(str);
  Pio(str, cen);
  Pio(str, n);
  Pio(str, d);
  Pio(str, radius);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Disc*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Disc::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Disc*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

