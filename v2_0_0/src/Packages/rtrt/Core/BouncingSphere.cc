
#include <Packages/rtrt/Core/BouncingSphere.h>
#include <Packages/rtrt/Core/BBox.h>
#include <iostream>

using namespace rtrt;

BouncingSphere::BouncingSphere(Material* matl,
			       const Point& cen, double radius,
			       const Vector& motion)
    : Sphere(matl, cen, radius), ocen(cen), motion(motion)
{
}

BouncingSphere::~BouncingSphere()
{
}

void BouncingSphere::animate(double t, bool& changed)
{
    changed=true;
    int i=(int)t;
    double f=t-i;
    double tt=3*f*f-2*f*f*f;
    if(i%2){
	cen=ocen+motion*tt;
    } else {
	cen=ocen+motion*(1-tt);
    }
}

void BouncingSphere::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(ocen, radius+offset);
    bbox.extend(ocen+motion, radius+offset);
}
