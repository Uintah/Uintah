
#include <Packages/rtrt/Core/Hemisphere.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/TrivialAllocator.h>
#include <iostream>

using namespace rtrt;
using namespace SCIRun;

Persistent* hemisphere_maker() {
  return new Hemisphere();
}

// initialize the static member type_id
PersistentTypeID Hemisphere::type_id("Hemisphere", "Object", hemisphere_maker);


Hemisphere::Hemisphere(Material* matl, const Point& cen, const Vector& orientation, double radius)
    : Object(matl), cen(cen), orientation(orientation), radius(radius)
{
  cerr << "orientation="<<orientation<<"\n";
}

Hemisphere::~Hemisphere()
{
}

void Hemisphere::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		       PerProcessorContext*)
{
    Vector OC=cen-ray.origin();
    double tca=Dot(OC, ray.direction());
    double l2oc=OC.length2();
    double rad2=radius*radius;
    st->sphere_isect++;
    if(l2oc < rad2) {
	// Inside the sphere
	double t2hc=rad2-l2oc+tca*tca;
	double thc=sqrt(t2hc);
	double t=tca+thc-0.00001;
	Point pplus(ray.origin()+ray.direction()*t);
	if (Dot(pplus-cen, orientation)>0)
	  hit.hit(this, t);
	st->sphere_hit++;
	return;
    } else {
	if(tca < 0) {
	    // Behind ray, no intersections...
	    return;
	} else {
	    double t2hc=rad2-l2oc+tca*tca;
	    if(t2hc <= 0.0){
		// Ray misses, no intersections
		return;
	    } else {
		double thc=sqrt(t2hc);
		Point pminus(ray.origin()+ray.direction()*(tca-thc));
		if (Dot(pminus-cen, orientation)>0)
		  hit.hit(this, tca-thc-0.00001);
		Point pplus(ray.origin()+ray.direction()*(tca+thc));
		if (Dot(pplus-cen, orientation)>0)
		  hit.hit(this, tca+thc-0.00001);
		st->sphere_hit++;
		return;
	    }
	}
    }	
}

// Maybe this could be improved - steve
void Hemisphere::light_intersect(Ray& ray, HitInfo& hit, Color&,
			     DepthStats* st, PerProcessorContext*)
{
  Vector OC=cen-ray.origin();
  double tca=Dot(OC, ray.direction());
  double l2oc=OC.length2();
  double rad2=radius*radius;
  st->sphere_light_isect++;
  if(l2oc <= rad2){
    // Inside the sphere
    if (Dot(OC,orientation)<0) return;
    double t2hc=rad2-l2oc+tca*tca;
    double thc=sqrt(t2hc);
    double t=tca+thc;
    hit.shadowHit(this, t);
    st->sphere_light_hit++;
    return;
  } else {
    if(tca < 0.0){
      // Behind ray, no intersections...
      return;
    } else {
      double t2hc=rad2-l2oc+tca*tca;
      if(t2hc <= 0.0){
	// Ray misses, no intersections
	return;
      } else {
	double thc=sqrt(t2hc);
	Point pminus(ray.origin()+ray.direction()*(tca-thc));
	if (Dot(pminus-cen, orientation)>0)
	  hit.shadowHit(this, tca-thc);
	Point pplus(ray.origin()+ray.direction()*(tca+thc));
	if (Dot(pplus-cen, orientation)>0)
	  hit.shadowHit(this, tca+thc);
	st->sphere_light_hit++;
	return;
      }
    }
  }	
}

Vector Hemisphere::normal(const Point& hitpos, const HitInfo&)
{
    Vector n=hitpos-cen;
    n*=1./radius;
    return n;
}

void Hemisphere::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(cen, radius+offset);
}

void Hemisphere::print(ostream& out)
{
    out << "Hemisphere: cen=" << cen << ", orientation=" << orientation << ", radius=" << radius << '\n';
}

void
Hemisphere::updatePosition( const Point & pos )
{
  cen = pos;
}

void
Hemisphere::updateOrientation( const Vector & dir )
{
  orientation = dir;
}

const int HEMISPHERE_VERSION = 1;

void 
Hemisphere::io(SCIRun::Piostream &str)
{
  str.begin_class("Hemisphere", HEMISPHERE_VERSION);
  Object::io(str);
  Pio(str, cen);
  Pio(str, orientation);
  Pio(str, radius);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Hemisphere*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Hemisphere::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Hemisphere*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
