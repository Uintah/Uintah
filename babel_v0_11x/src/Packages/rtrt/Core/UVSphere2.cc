/*
Name:		James Bigler, Christiaan Gribble
Location:	University of Utah
Email:		bigler@cs.utah.edu; cgribble@cs.utah.edu

This file contains the information necessary to do uv mapping for a sphere.
*/

#include <Packages/rtrt/Core/UVSphere2.h>
#include <Packages/rtrt/Core/UV.h>

#if HAVE_IEEEFP_H
#include <ieeefp.h>
#endif

using namespace rtrt;
using namespace SCIRun;

Persistent* uvsphere2_maker() {
  return new UVSphere2();
}

// initialize the static member type_id
PersistentTypeID UVSphere2::type_id("UVSphere2", "Object", uvsphere2_maker);


UVSphere2::UVSphere2(Material *matl, const Point &center, double radius):
  Sphere(matl,center,radius)
{
  set_uvmapping(this);
}

UVSphere2::~UVSphere2()
{
}

void UVSphere2::uv(UV& uv, const Point& hitpos, const HitInfo&)  
{
  // Get point on unit sphere
  Point point_on_sphere((hitpos - cen) * 1/radius);
  double uu,vv,theta,phi;  
  theta = acos(-point_on_sphere.y());
  phi = atan2(point_on_sphere.z(), point_on_sphere.x());
  if (phi < 0)
    phi += 2*M_PI;
  uu = phi * 0.5 * M_1_PI;
  vv = (M_PI - theta) * M_1_PI;
  //#if HAVE_IEEEFP_H
#if 0
  if (isnand(uu) || isnand(vv)) {
    cerr << "uu = "<<uu<<", vv = "<<vv<<"\n";
    cerr << "theta = "<<theta<<", phi = "<<phi<<"\n";
    cerr << "hitpos = "<<hitpos<<", point_on_sphere = "<<point_on_sphere<<"\n";
    cerr << "radius = "<<radius<<"\n";
  }
#endif
  uv.set( uu,vv);
}

void UVSphere2::get_frame(const Point &hitpos, const HitInfo&hit,
                         const Vector &norm,  Vector &pu, Vector &pv)
{
  UV uv_m;
  float u,v;
  double phi,theta;
  uv(uv_m,hitpos,hit);
  u = uv_m.u();
  v = uv_m.v();
  phi = 6.28318530718 * u;
  theta = -(M_PI*v) + M_PI;
  pu = Vector(-6.28318530718* radius * sin(phi) * sin(theta),
	      6.28318530718 * radius * sin(phi) * cos(theta), 	0);
  pv = Vector(M_PI * radius * cos(phi) * cos(theta),
	      M_PI * radius * cos(phi) * sin(theta),
	      -1 * M_PI * radius * sin(phi));
  VXV3(pu,norm,pv);
  VXV3(pv,norm,pu);
}

const int UVSPHERE2_VERSION = 1;

void 
UVSphere2::io(SCIRun::Piostream &str)
{
  str.begin_class("UVSphere2", UVSPHERE2_VERSION);
  Sphere::io(str);
  UVMapping::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::UVSphere2*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::UVSphere2::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::UVSphere2*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
