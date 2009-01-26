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



#include <Packages/rtrt/Core/DiscArc.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

DiscArc::DiscArc(Material* matl, const Point& cen, const Vector& n,
		 double radius)
  : Disc(matl, cen, n, radius), theta0(0), theta1(6.283185)
{
}

DiscArc::~DiscArc()
{
}

void DiscArc::intersect(Ray& ray, HitInfo& hit, DepthStats*,
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
  double theta=atan2(xp.x(), xp.y());
  if (theta<0) theta+=2*M_PI;
  if(l < radius*radius && (theta > theta0) && (theta < theta1))
    hit.hit(this, t);
}

void DiscArc::light_intersect(Ray& ray, HitInfo& hit, Color&,
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
  double theta=atan2(xp.x(), xp.y());
  if (theta<0) theta+=2*M_PI;
  if(l < radius*radius && (theta > theta0) && (theta < theta1))
    hit.shadowHit(this, t);
}
