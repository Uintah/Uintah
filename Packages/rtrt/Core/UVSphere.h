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
//#include <Packages/rtrt/Core/vec.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
class UVSphere;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::UVSphere*&);
}

namespace rtrt {

class UVSphere : public Object, public UVMapping {

 protected:

  Point  cen;
  Vector up;
  Vector right;
  double radius;
  Transform xform;
  Transform ixform;


  inline double _DET2(const Vector &v0, const Vector &v1, int i0, int i1) {
    return (v0[i0] * v1[i1] + v0[i1] * -v1[i0]);
  }

  inline void VXV3(Vector &to, const Vector &v1, const Vector &v2) {
    to[0] =  _DET2(v1,v2, 1,2);
    to[1] = -_DET2(v1,v2, 0,2);
    to[2] =  _DET2(v1,v2, 0,1);
  }
  
 public:
  UVSphere(Material *m, Point c, double r, const Vector &up=Vector(0,0,1),
           const Vector &right=Vector(1,0,0));
  UVSphere() : Object(0), UVMapping() {} // for Pio.
  virtual ~UVSphere();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, UVSphere*&);

  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void uv(UV& uv, const Point&, const HitInfo& hit);
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
#if 1
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
#endif
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

  Vector get_up() { return up; }
  void set_up(const Vector &v) { up = v; }

  Vector get_right() { return right; }
  void set_right(const Vector &v) { right = v; } 
};
 

} // end namespacertrt
#endif


