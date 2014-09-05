
#include "Sphere.h"
#include "HitInfo.h"
#include "Ray.h"
#include "Light.h"
#include "BBox.h"
#include "Stats.h"
#include "TrivialAllocator.h"
#include <iostream>

using namespace rtrt;

#if 0
static TrivialAllocator Sphere_alloc(sizeof(Sphere));

void* Sphere::operator new(size_t)
{
    return Sphere_alloc.alloc();
}

void Sphere::operator delete(void* rp, size_t)
{
    Sphere_alloc.free(rp);
}
#endif

Sphere::Sphere(Material* matl, const Point& cen, double radius)
    : Object(matl), cen(cen), radius(radius)
{
}

Sphere::~Sphere()
{
}

void Sphere::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
		       PerProcessorContext*)
{
    Vector OC=cen-ray.origin();
    double tca=OC.dot(ray.direction());
    double l2oc=OC.length2();
    double rad2=radius*radius;
    st->sphere_isect++;
    if(l2oc <= rad2){
	// Inside the sphere
	double t2hc=rad2-l2oc+tca*tca;
	double thc=sqrt(t2hc);
	double t=tca+thc;
	hit.hit(this, t);
	st->sphere_hit++;
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
		hit.hit(this, tca-thc);
		hit.hit(this, tca+thc);
		st->sphere_hit++;
		return;
	    }
	}
    }	
}

Vector Sphere::normal(const Point& hitpos, const HitInfo&)
{
    Vector n=hitpos-cen;
    n*=1./radius;
    return n;
}

void Sphere::light_intersect(Light* light, const Ray& ray, HitInfo&,
			     double dist, Color& atten, DepthStats* st,
			     PerProcessorContext*)
{
    Vector OC=cen-ray.origin();
    double tca=OC.dot(ray.direction());
    st->sphere_light_isect++;
    if(tca<1.e-4)
	return;

    double delta=light->radius*tca/dist;
    double Ri=radius;
    double Ro=radius+delta;
    double Ri2=Ri*Ri;
    double Ro2=Ro*Ro;

    Vector normal=ray.direction()*tca-OC;
    double d2=normal.length2();

    if(d2>Ro2){
	return;
    }

    if(d2<Ri2){
	atten=Color(0,0,0);
	st->sphere_light_hit++;
	return;
    }



    st->sphere_light_penumbra++;
    double t=(sqrt(d2)-Ri)/(delta);
    double g=3*t*t-2*t*t*t;

    atten=g<atten.luminance()?Color(g,g,g):atten;
}

void Sphere::multi_light_intersect(Light*, const Point& orig,
				   const Array1<Vector>& dirs,
				   const Array1<Color>& attens,
				   double,
				   DepthStats*, PerProcessorContext*)
{
    Vector OC=orig-cen;
    double rad2=radius*radius;
    double C=OC.length2()-rad2;
    for(int i=0;i<dirs.size();i++){
	const Vector& dir=dirs[i];
	double B=dir.dot(OC);
	double B2=B*B;
	if(B2 > C){
	    double disc=sqrt(B2-C); // A=1
	    double t=(-B+disc);
	    if(t>1.e-6)
		attens[i]=Color(0,0,0);
	}
    }
}

void Sphere::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(cen, radius+offset);
}

void Sphere::print(ostream& out)
{
    out << "Sphere: cen=" << cen << ", radius=" << radius << '\n';
}

