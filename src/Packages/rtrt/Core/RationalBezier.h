#ifndef RationalBezier_H
#define RationalBezier_H

#include <iostream>

#include <Packages/rtrt/Core/RationalMesh.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Group.h>

#define PATCH_OVERLAP 0.05
#define FNAME "bez.without.overlap"

namespace rtrt {
  
class RationalBezier: public Object {

public:
    
  RationalBezier(Material *, RationalMesh *, double u0=0., double u1=1., double v0=0., double v1=1.);
  RationalBezier(Material *, RationalMesh *, RationalMesh *, double u0, double u1, double v0, double v1);

  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }
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
      
      RationalMesh *nwm, *nem, *sem, *swm;
      Material *mat = get_matl();
      
      local->SubDivide(&nwm,&nem,&sem,&swm,w);
      
      /*nw = new RationalBezier (mat,control,nwm,
		       ustart, ustart+(ufinish-ustart)*w,
		       vstart, vstart+(vfinish-vstart)*w);
      ne = new RationalBezier (mat,control,nem,
		       ustart+(ufinish-ustart)*w, ufinish,
		       vstart, vstart+(vfinish-vstart)*w);
      sw = new RationalBezier (mat,control,swm,
		       ustart, ustart+(ufinish-ustart)*w,
		       vstart+(vfinish-vstart)*w, vfinish);
      se = new RationalBezier (mat,control,sem,
		       ustart+(ufinish-ustart)*w, ufinish,
		       vstart+(vfinish-vstart)*w, vfinish);*/
      
      nw = new RationalBezier(mat,control,nwm,
		      ustart, ustart+(ufinish-ustart)*w,
		      vstart, vstart+(vfinish-vstart)*w);
      ne = new RationalBezier (mat,control,nem,
		       ustart+(ufinish-ustart)*w, ufinish,
		       vstart, vstart+(vfinish-vstart)*w);
      sw = new RationalBezier (mat,control,swm,
		       ustart, ustart+(ufinish-ustart)*w,
		       vstart+(vfinish-vstart)*w, vfinish);
      se = new RationalBezier (mat,control,sem,
		       ustart+(ufinish-ustart)*w, ufinish,
		       vstart+(vfinish-vstart)*w, vfinish);
      
      isleaf = 0;
      local = 0;
      
      nw->SubDivide(n-1,w);
      ne->SubDivide(n-1,w);
      sw->SubDivide(n-1,w);
      se->SubDivide(n-1,w);

      //if (nw->isleaf) 
      //printf("patchdone\n");
      
    }
    else {
      //puv();
      // Make the patches overlap be a little bit
      double udiff, vdiff;

      udiff = ufinish-ustart;
      vdiff = vfinish-vstart;
      ustart  -= udiff*PATCH_OVERLAP;
      ufinish += udiff*PATCH_OVERLAP;
      vstart  -= vdiff*PATCH_OVERLAP;
      vfinish += vdiff*PATCH_OVERLAP;

      bbox = new BBox();
      local->compute_bounds(*bbox);

      // we don't need to keep the local RationalMesh around!
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
      if (t > 1.e-2 && hit.hit(this,t)) {
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
	  Point4D P;
	  
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
  void puv() {
    FILE *fp;
    fp = fopen(FNAME,"a");
    fprintf(fp,"u0:%lf u1:%lf v0:%lf v1:%lf uguess:%lf vguess:%lf\n",ustart,ufinish,vstart,vfinish,uguess,vguess);
    fclose(fp);
  }
    
private:

  RationalMesh *control, *local;
  RationalBezier *ne, *nw, *se, *sw;
  BBox *bbox;
  int isleaf;
  // the boundary of the current patch
  double ustart, ufinish, vstart, vfinish;
  double uguess,vguess;
};

} // end namespace rtrt

#endif

