/*
Name:		Shaun Ramsey
Location:	University of Utah
Email:		ramsey@cs.utah.edu
*/

/* by Shaun Ramsey on 5-24-00 for tex/bump mapping a sphere */
/* Mods done on 6/10/02 for SCI Run and sgi demo at siggraph*/

#ifndef UVSPHERE_H
#define UVSPHERE_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/vec.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>


namespace rtrt {

using namespace rtrt;
using namespace SCIRun;

class UVSphere : public Object, UVMapping {
  Point  cen;
  Vector up;
  double radius;
  Transform xform;
  Transform ixform;
 public:
  UVSphere(Material *m, Point c, double r, const Vector &up=Vector(0,0,1));
  virtual ~UVSphere();
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void uv(UV& uv, const Point&, const HitInfo& hit);
  virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void get_frame(const Point &hitpos, const HitInfo&hit,
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
};
 

} // end namespacertrt
#endif


