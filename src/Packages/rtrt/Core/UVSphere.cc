/*
Name:		Shaun Ramsey
Location:	University of Utah
Email:		ramsey@cs.utah.edu

This file contains the information necessary to do uv mapping for a sphere.
It will be used in ImageMaterial for texture mapping as well as in
RamseyImage for BumpMapping information. Remember that whenever you create a
sphere which may or may not use these options its always good code to include
the change in mappying to a UVSphere map.
*/

#include <Packages/rtrt/Core/UVSphere.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/vec.h>

using namespace rtrt;
using namespace SCIRun;

UVSphere::UVSphere(Point c, double r)
{
radius = r;
cen = c;
}

UVSphere::~UVSphere()
{
}

//vv = acos(m.z()/radius) / M_PI;
//uu = acos(m.x() / (radius * sin (M_PI*vv)))* over2pi;
void UVSphere::uv(UV& uv, const Point& hitpos, const HitInfo&)  
{
Vector m(hitpos-cen);  
double uu,vv,theta,phi;  
theta = acos(m.z()/radius);
phi = atan2(m.y(), m.x());
if(phi < 0) phi += 6.28318530718;
 uu = phi * .159154943092; // 1_pi
 vv = (M_PI - theta) * .318309886184; // 1 / ( 2 * pi )
uv.set( uu,vv);
}
