
/*
 *  BSphere.cc: Bounding Sphere's
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geometry/BSphere.h>
#include <Classlib/Assert.h>
#include <Classlib/Persistent.h>
#include <Geometry/Ray.h>

BSphere::BSphere()
: have_some(0)
{
}

BSphere::~BSphere()
{
}

BSphere::BSphere(const BSphere& copy)
: have_some(copy.have_some), cen(copy.cen), rad(copy.rad), rad2(copy.rad2)
{
}

void BSphere::reset()
{
    have_some=0;
}

double BSphere::volume()
{
    ASSERT(have_some);
    return 4./3.*M_PI*rad*rad*rad;
}

void BSphere::extend(const BSphere& s)
{
    ASSERT(s.have_some);
    extend(s.cen, s.rad);
}

void BSphere::extend(const Point& ncen, double nrad)
{
    if(!have_some){
	cen=ncen;
	rad=nrad;
	rad2=rad*rad;
	have_some=1;
    } else {
	Vector dv=ncen-cen;
	double d=dv.length();
	if(rad >= d+nrad){
	    // This one is big enough...
	} else if(nrad >= d+rad){
	    // Use s's BV
	    cen=ncen;
	    rad=nrad;
	    rad2=rad*rad;
	} else {
	    // Union
	    double newrad=(d+rad+nrad)/2.;
	    cen=cen+dv*((newrad-rad)/d);
	    rad=newrad;
	    rad2=rad*rad;
	}
    }
}

int BSphere::intersect(const Ray& ray)
{
    Vector OC(cen-ray.origin());
    double tca=Dot(OC, ray.direction());
    double l2oc=OC.length2();
    double radius_sq=rad*rad;
    if(l2oc > radius_sq){
	if(tca >= 0.0){
	    double t2hc=radius_sq-l2oc+tca*tca;
	    if(t2hc > 0.0){
		return 1;
	    }
	}
    }
    return 0;
}

void BSphere::extend(const Point& p)
{
    if(have_some){
	Vector v(p-cen);
	double dist=v.normalize();
	if(dist > rad){
	    // extend it...
	    double new_rad=(dist+rad)/2;
	    double frac=new_rad-rad;
	    cen+=v*frac;
	}
    } else {
	cen=p;
	rad=1.e-4;
	rad2=rad*rad;
    }
}

void Pio(Piostream& stream, BSphere& s)
{
    stream.begin_cheap_delim();
    Pio(stream, s.have_some);
    Pio(stream, s.cen);
    Pio(stream, s.rad);
    if(stream.reading())
	s.rad2=s.rad*s.rad;
    stream.end_cheap_delim();
}
