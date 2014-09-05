
#include "Box.h"
#include "HitInfo.h"
#include "Ray.h"
#include "Light.h"
#include "BBox.h"
#include "Stats.h"
#include "TrivialAllocator.h"
#include "MiscMath.h"
#include <iostream>
#include <float.h>

using namespace rtrt;

Box::Box(Material* matl, const Point& min, const Point& max )
    : Object(matl), min(min), max(max)
{
}

Box::~Box()
{
}

void Box::intersect(const Ray& r, HitInfo& hit, DepthStats* st,
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
       if (t1 > FLT_EPSILON)
          hit.hit(this, t1);
       else if (t2 > FLT_EPSILON)
          hit.hit(this, t2);
    }

}

void Box::light_intersect(Light*, const Ray& lightray,
                                  HitInfo& hit, double, Color&,
                                  DepthStats* ds, PerProcessorContext* ppc)
{
    intersect(lightray, hit, ds, ppc);
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


  
