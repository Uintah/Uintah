
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/TrivialAllocator.h>
#include <iostream>

using namespace rtrt;

SCIRun::Persistent* cylinder_maker() {
  return new Cylinder();
}

// initialize the static member type_id
SCIRun::PersistentTypeID Cylinder::type_id("Cylinder", "Object", 
					   cylinder_maker);

Cylinder::Cylinder(Material* matl, const Point& bottom, const Point& top,
		   double radius)
    : Object(matl), bottom(bottom), top(top), radius(radius)
{
}

Cylinder::~Cylinder()
{
}

void Cylinder::preprocess(double, int&, int&)
{
    Vector axis=top-bottom;
    double height=axis.normalize();
    // Set up unit transformation
    xform.load_identity();
    xform.pre_translate(-bottom.asVector());
    xform.rotate(axis, Vector(0,0,1));
    xform.pre_scale(Vector(1./radius, 1./radius, 1./height));
//    print(cerr);
//    xform.print();
    ixform.load_identity();
    ixform.pre_scale(Vector(radius, radius, height));
    ixform.rotate(Vector(0,0,1), axis);
    ixform.pre_translate(bottom.asVector());
//    ixform.print();
    
}

void Cylinder::intersect(Ray& ray, HitInfo& hit, DepthStats*,
			 PerProcessorContext*)
{
  // Do a transformation to unit coordinates:
  //  double dist_scale;
  //  Ray xray(xform.xray(ray, dist_scale));

  // I had to get rid of Transform::xray when I started using
  // SCIRun::Transform as part of the code merger.
  // Since this is the only place where it is used, I'm going to just
  // transplant the code into here
  Vector v(xform.project(ray.direction()));
  double dist_scale=v.normalize();
  Ray xray(xform.project(ray.origin()), v);
    double dx=xray.direction().x();
    double dy=xray.direction().y();
    double a=dx*dx+dy*dy;
    if(a < 1.e-6)
	return;
    // Check sides...
    double ox=xray.origin().x();
    double oy=xray.origin().y();
    double oz=xray.origin().z();
    double dz=xray.direction().z();

    double b=2*(ox*dx+oy*dy);
    double c=ox*ox+oy*oy-1;
    double d=b*b-4*a*c;
    if(d<=0.0)
	return; // Off to the side

    double sd=sqrt(d);
    double t1=(-b+sd)/(2*a);
    double t2=(-b-sd)/(2*a);

    if(t1>t2){
	double tmp=t1;
	t1=t2;
	t2=tmp;
    }
    double z1=oz+t1*dz;
    double z2=oz+t2*dz;
    if(t1 > 1.e-6 && z1 > 0.0 && z1 < 1.0){
	hit.hit(this, t1/dist_scale);
    } else if(t2 > 1.e-6 && z2 > 0.0 && z2 < 1.0){
	hit.hit(this, t2/dist_scale);
    }
}

Vector Cylinder::normal(const Point& hitpos, const HitInfo&)
{
    Vector xn(xform.project(hitpos).asVector());
    xn.z(0.0);
    Vector v=ixform.project(xn);
    v.normalize();
    return v;
}

void Cylinder::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(bottom, radius+offset);
    bbox.extend(top, radius+offset);
}

void Cylinder::print(ostream& out)
{
  out << "Cylinder: top=" << top << ", bottom=" << bottom 
      << ", radius=" << radius << '\n';
}


const int CYLINDER_VERSION = 1;

void 
Cylinder::io(SCIRun::Piostream &str)
{
  str.begin_class("Cylinder", CYLINDER_VERSION);
  Object::io(str);
  SCIRun::Pio(str, top);
  SCIRun::Pio(str, bottom);
  SCIRun::Pio(str, radius);
  SCIRun::Pio(str, xform);
  SCIRun::Pio(str, ixform);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Cylinder*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Cylinder::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Cylinder*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
