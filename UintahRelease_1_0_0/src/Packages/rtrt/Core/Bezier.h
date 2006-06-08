#ifndef Bezier_H
#define Bezier_H

#include <Packages/rtrt/Core/Mesh.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Group.h>

#define OVERLAP 0.1
#define PATCH_OVERLAP 0.05

namespace rtrt {
  class Bezier;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::Bezier*&);
}

namespace rtrt {

class Bezier: public Object {

public:
    
  Bezier(Material *, Mesh *, double u0=0, double u1=1, double v0=0, 
	 double v1=1);
  Bezier(Material *, Mesh *, Mesh *, double u0, double u1, double v0, 
	 double v1);

  Bezier() : Object(0) {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Bezier*&);

  inline void setParams(double u0, double u1, double v0, double v1)
  {
    ustart = u0;
    ufinish = u1;
    vstart = v0;
    vfinish = v1;
    uguess = .5*(ustart+ufinish);
    vguess = .5*(vstart+vfinish);
  }

  inline Group *Groupify() 
  {
    Group *g = new Group();

    if (isleaf) 
      g->add(this);
    else {
      g->add(nw->Groupify());
      g->add(ne->Groupify());
      g->add(se->Groupify());
      g->add(sw->Groupify());
    }
    return g;
  }

  inline BV1 *MakeBVH() {
    return new BV1(Groupify());
  }

  inline void SubDivide(int n, double w) {
    if (n > 0) {
      
      Mesh *nwm, *nem, *sem, *swm;
      Material *mat = get_matl();
      
      local->SubDivide(&nwm,&nem,&sem,&swm,w);
      
      nw = new Bezier (mat,control,nwm,
		       ustart, ustart+(ufinish-ustart)*w,
		       vstart, vstart+(vfinish-vstart)*w);
      ne = new Bezier (mat,control,nem,
		       ustart+(ufinish-ustart)*w, ufinish,
		       vstart, vstart+(vfinish-vstart)*w);
      sw = new Bezier (mat,control,swm,
		       ustart, ustart+(ufinish-ustart)*w,
		       vstart+(vfinish-vstart)*w, vfinish);
      se = new Bezier (mat,control,sem,
		       ustart+(ufinish-ustart)*w, ufinish,
		       vstart+(vfinish-vstart)*w, vfinish);

      isleaf = 0;
      local = 0;
      
      nw->SubDivide(n-1,w);
      ne->SubDivide(n-1,w);
      se->SubDivide(n-1,w);
      sw->SubDivide(n-1,w);
      
    }
    else {
      // Make the patches overlap be a little bit
      double udiff, vdiff;

      udiff = ufinish-ustart;
      vdiff = vfinish-vstart;
      ustart  -= udiff*PATCH_OVERLAP;
      ufinish += udiff*PATCH_OVERLAP;
      vstart  -= vdiff*PATCH_OVERLAP;
      vfinish += vdiff*PATCH_OVERLAP;

      bbox = new BBox();
      local->compute_bounds(bbox);

      // we don't need to keep the local Mesh around!
      delete local;
      local = 0;
    }
  }

  inline void compute_bounds(BBox &b, double) {
    b.extend(*bbox);
  }
  
  inline void intersect(Ray &r, HitInfo &hit, DepthStats*,
			PerProcessorContext *ppc) 
  {
    
    double u,v;
    double t;
    
    // reset initial guess
    u = uguess;
    v = vguess;
    //if(control->Hit(r,u,v,t,ustart,ufinish,vstart,vfinish,ppc)) {
    if(control->Hit_Broyden(r,u,v,t,ustart,ufinish,vstart,vfinish,ppc)) {
      if (t > 2e-2 && hit.hit(this,t)) {
	Vector *n = (Vector *)hit.scratchpad;
	*n = control->getNormal(u,v,ppc);
      }
    }
  }

  /*inline void intersect(Ray &r, HitInfo &hit, DepthStats* st,
    PerProcessorContext *ppc) {
      
    Point orig(r.origin());    
    Vector dir(r.direction());
    Vector idir(1./dir.x(), 1./dir.y(), 1./dir.z());
    double u,v;
      
    if (bvintersect(orig,idir,hit)) {
    //if (1) {
    if (isleaf) {
    double t;
    Point P;
	  
    // reset initial guess
    u = .5*(ustart+ufinish);
    v = .5*(vstart+vfinish);
    if(control->Hit(r,u,v,t,P,ustart,ufinish,vstart,vfinish,ppc)) {
    if (hit.hit(this,t)) {
    Vector *n = (Vector *)hit.scratchpad;
    *n = control->getNormal(u,v,ppc);
    }
    } else {
    // reset initial guess
    u = .5*(ustart+ufinish);
    v = .5*(vstart+vfinish);
    }
    } else {
    nw->intersect(r,hit,st,ppc);
    ne->intersect(r,hit,st,ppc);
    se->intersect(r,hit,st,ppc);
    sw->intersect(r,hit,st,ppc);
    }
    }
    }*/

  inline Vector normal(const Point&, const HitInfo& hit)
  {
    // We computed the normal at intersect time and tucked it
    // away in the scratchpad...
    Vector* n=(Vector*)hit.scratchpad;
    return *n;
  }

  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  void Out(int,int);
  void Print();
    
private:

  Mesh *control, *local;
  Bezier *ne, *nw, *se, *sw;
  BBox *bbox;
  //double bv[6];
  int isleaf;
  // the boundary of the current patch
  double ustart, ufinish, vstart, vfinish;
  double uguess,vguess;
  // the u,v of the current column
  //double u,v;
};

} // end namespace rtrt

#endif

