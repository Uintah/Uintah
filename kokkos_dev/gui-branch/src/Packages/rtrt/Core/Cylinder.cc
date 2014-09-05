
#include "Cylinder.h"
#include "HitInfo.h"
#include "Ray.h"
#include "Light.h"
#include "BBox.h"
#include "Stats.h"
#include "TrivialAllocator.h"
#include <iostream>

using namespace rtrt;

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
    print(cerr);
    xform.print();
    ixform.load_identity();
    ixform.pre_scale(Vector(radius, radius, height));
    ixform.rotate(Vector(0,0,1), axis);
    ixform.pre_translate(bottom.asVector());
    ixform.print();
    
}

void Cylinder::intersect(const Ray& ray, HitInfo& hit, DepthStats*,
			 PerProcessorContext*)
{
    // Do a transformation to unit coordinates:
    double dist_scale;
    Ray xray(xform.xray(ray, dist_scale));
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

void Cylinder::light_intersect(Light*, const Ray& lightray, HitInfo& hit,
			       double, Color&, DepthStats* ds,
			       PerProcessorContext* ppc)
{
    intersect(lightray, hit, ds, ppc);
}

void Cylinder::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(bottom, radius+offset);
    bbox.extend(top, radius+offset);
}

void Cylinder::print(ostream& out)
{
    out << "Cylinder: top=" << top << ", bottom=" << bottom << ", radius=" << radius << '\n';
}
