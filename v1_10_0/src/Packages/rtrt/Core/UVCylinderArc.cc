
#include <Packages/rtrt/Core/UVCylinderArc.h>

#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/TrivialAllocator.h>

#include <iostream>
#include <math.h>

using namespace rtrt;
using namespace std;

SCIRun::Persistent* uvca_maker() {
  return new UVCylinderArc();
}

// initialize the static member type_id
SCIRun::PersistentTypeID UVCylinderArc::type_id("UVCylinderArc", "Object", 
						uvca_maker);

UVCylinderArc::UVCylinderArc(Material* matl, const Point& bottom, 
			     const Point& top, double radius) : 
  Object(matl,this), 
  top(top), 
  bottom(bottom), 
  radius(radius), 
  theta0(0), 
  theta1(3.141593), 
  tex_scale(Vector(1,1,1))
{
}

UVCylinderArc::~UVCylinderArc()
{
}

void UVCylinderArc::preprocess(double, int&, int&)
{
    Vector axis=top-bottom;
    double height=axis.normalize();
    // Set up unit transformation
    xform.load_identity();
    xform.pre_translate(-bottom.asVector());
    xform.rotate(axis, Vector(0,0,1));
    xform.pre_scale(Vector(1./radius, 1./radius, 1./height));
    print(cerr);
    xform.print();
    ixform.load_identity();
    ixform.pre_scale(Vector(radius, radius, height));
    ixform.rotate(Vector(0,0,1), axis);
    ixform.pre_translate(bottom.asVector());
    ixform.print();
    
}

void UVCylinderArc::intersect(Ray& ray, HitInfo& hit, DepthStats*,
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

    Point p1 = xray.origin()+xray.direction()*t1;
    Point p2 = xray.origin()+xray.direction()*t2;
    double thet1 = M_PI+atan2(p1.y(),p1.x());
    double thet2 = M_PI+atan2(p2.y(),p2.x());

    if(t1 > 1.e-6 && z1 > 0.0 && z1 < 1.0 &&
       thet1>theta0 && thet1<theta1 ){
        hit.hit(this, t1/dist_scale);
    } else if(t2 > 1.e-6 && z2 > 0.0 && z2 < 1.0 &&
        thet2>theta0 && thet2<theta1 ){
        hit.hit(this, t2/dist_scale);
    }
}

Vector UVCylinderArc::normal(const Point& hitpos, const HitInfo&)
{
    Vector xn(xform.project(hitpos).asVector());
    xn.z(0.0);
    Vector v=ixform.project(xn);
    v.normalize();
    return v;
}

void UVCylinderArc::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(bottom, radius+offset);
    bbox.extend(top, radius+offset);
}

void UVCylinderArc::print(ostream& out)
{
    out << "UVCylinderArc: top=" << top << ", bottom=" << bottom 
	<< ", radius=" << radius << '\n';
}

void UVCylinderArc::uv(UV &uv, const Point &p, const HitInfo &hit)
{
    Point xp = xform.project(p);
    double angle = xp.x();
    if(angle<-1)
      angle=-1;
    else if(angle>1)
      angle=1;
    double theta = acos(angle);
    uv.set((theta/6.283185)/tex_scale.x(),
           xp.z()/tex_scale.y());
}

const int UVCYLINDERARC_VERSION = 1;

void 
UVCylinderArc::io(SCIRun::Piostream &str)
{
  str.begin_class("UVCylinderArc", UVCYLINDERARC_VERSION);
  Object::io(str);
  UVMapping::io(str);
  SCIRun::Pio(str, top);
  SCIRun::Pio(str, bottom);
  SCIRun::Pio(str, radius);
  SCIRun::Pio(str, tex_scale);
  SCIRun::Pio(str, xform);
  SCIRun::Pio(str, ixform);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::UVCylinderArc*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::UVCylinderArc::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::UVCylinderArc*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
