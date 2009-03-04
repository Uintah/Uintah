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



#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/TrivialAllocator.h>

#include <Core/Math/MiscMath.h>
#include <sci_values.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* box_maker() {
  return new Box();
}

// initialize the static member type_id
PersistentTypeID Box::type_id("Box", "Object", box_maker);

Box::Box(Material* matl, const Point& min, const Point& max )
    : Object(matl), min(min), max(max)
{
}

Box::~Box()
{
}

void Box::intersect(Ray& r, HitInfo& hit, DepthStats* st,
		    PerProcessorContext*)
{
   st->box_isect++;
    
   double t1, t2, tx1, tx2, ty1, ty2, tz1, tz2;

/*
    if (r.direction().x() > FLT_EPSILON) {
        tx1 = (min.x() - r.origin().x()) / r.direction().x();
        tx2 = (max.x() - r.origin().x()) / r.direction().x();
    }
    else if (r.direction().x() < -FLT_EPSILON) {
        tx1 = (max.x() - r.origin().x()) / r.direction().x();
        tx2 = (min.x() - r.origin().x()) / r.direction().x();
    }
    else {
         tx1 = DBL_MIN;
         tx2 = DBL_MAX;
    }

    if (r.direction().y() > FLT_EPSILON) {
        ty1 = (min.y() - r.origin().y()) / r.direction().y();
        ty2 = (max.y() - r.origin().y()) / r.direction().y();
    }
    else if (r.direction().y() < -FLT_EPSILON) {
        ty1 = (max.y() - r.origin().y()) / r.direction().y();
        ty2 = (min.y() - r.origin().y()) / r.direction().y();
    }
    else {
         ty1 = DBL_MIN;
         ty2 = DBL_MAX;
    }

    if (r.direction().z() > FLT_EPSILON) {
        tz1 = (min.z() - r.origin().z()) / r.direction().z();
        tz2 = (max.z() - r.origin().z()) / r.direction().z();
    }
    else if (r.direction().z() < -FLT_EPSILON) {
        tz1 = (max.z() - r.origin().z()) / r.direction().z();
        tz2 = (min.z() - r.origin().z()) / r.direction().z();
    }
    else {
         tz1 = DBL_MIN;
         tz2 = DBL_MAX;
    }
   
    if (tx1 > ty1)
        t1 = tx1;
    else
        t1 = ty1;
    if (tz1 > t1) t1 = tz1;
   
    if (tx2 < ty2)
        t2 = tx2;
    else
        t2 = ty2;
    if (tz2 < t2) t2 = tz2;
*/
    if (r.direction().x() > 0) {
        tx1 = (min.x() - r.origin().x()) / r.direction().x();
        tx2 = (max.x() - r.origin().x()) / r.direction().x();
    }
    else {
        tx1 = (max.x() - r.origin().x()) / r.direction().x();
        tx2 = (min.x() - r.origin().x()) / r.direction().x();
    }

    if (r.direction().y() > 0) {
        ty1 = (min.y() - r.origin().y()) / r.direction().y();
        ty2 = (max.y() - r.origin().y()) / r.direction().y();
    }
    else {
        ty1 = (max.y() - r.origin().y()) / r.direction().y();
        ty2 = (min.y() - r.origin().y()) / r.direction().y();
    }

    if (r.direction().z() > 0) {
        tz1 = (min.z() - r.origin().z()) / r.direction().z();
        tz2 = (max.z() - r.origin().z()) / r.direction().z();
    }
    else {
        tz1 = (max.z() - r.origin().z()) / r.direction().z();
        tz2 = (min.z() - r.origin().z()) / r.direction().z();
    }
    t1 =  DBL_MIN; 
    t2 =  DBL_MAX;

    if (tx1 > t1) t1 = tx1;
    if (ty1 > t1) t1 = ty1;
    if (tz1 > t1) t1 = tz1;
   
    if (tx2 < t2) t2 = tx2;
    if (ty2 < t2) t2 = ty2;
    if (tz2 < t2) t2 = tz2;

    if (t2 > t1) {
       st->box_hit++;
       hit.hit(this, t1);
       hit.hit(this, t2);
    }

}

Vector Box::normal(const Point& hitpos, const HitInfo&)
{
    if (Abs(hitpos.x() - min.x()) < 0.0001)
         return Vector(-1, 0, 0 );
    else if (Abs(hitpos.x() - max.x()) < 0.0001)
         return Vector( 1, 0, 0 );
    else if (Abs(hitpos.y() - min.y()) < 0.0001)
         return Vector( 0,-1, 0 );
    else if (Abs(hitpos.y() - max.y()) < 0.0001)
         return Vector( 0, 1, 0 );
    else if (Abs(hitpos.z() - min.z()) < 0.0001)
         return Vector( 0, 0,-1 );
    else 
         return Vector( 0, 0, 1 );
}

void Box::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend( min - Vector(offset, offset, offset) );
    bbox.extend( max + Vector(offset, offset, offset) );
}

void Box::print(ostream& out)
{
    out << "Box: min=" << min << ", max=" << max << '\n';
}


const int BOX_VERSION = 1;

void 
Box::io(SCIRun::Piostream &str)
{
  str.begin_class("Box", BOX_VERSION);
  Object::io(str);
  SCIRun::Pio(str, min);
  SCIRun::Pio(str, max);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Box*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Box::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Box*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
