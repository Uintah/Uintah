#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/CutMaterial.h>
#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Core/Math/MinMax.h>

using namespace rtrt;

CutMaterial::CutMaterial(Material *surfmat, ColorMap *_cmap, CutPlaneDpy *dpy) 
  : surfmat(surfmat), dpy(dpy) 
{
  if (!_cmap) cmap = new ColorMap(); else cmap = _cmap;
}

CutMaterial::CutMaterial(Material *surfmat, CutPlaneDpy *dpy, ColorMap *_cmap)
  : surfmat(surfmat), dpy(dpy) 
{
  if (!_cmap) cmap = new ColorMap(); else cmap = _cmap;
}

void CutMaterial::shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
				 Context* cx) {
  
  if (dpy && !dpy->on) {
    surfmat->shade(result, ray, hit, depth, atten, accumcolor, cx);
  } else {
    
    //find t at which the plane (pt) and the object (ot) were hit
    double pt = *((double *)(hit.scratchpad+CUTGROUPDIST));
    double ot = hit.min_t;
    double it = Min(pt,ot);
    
    Point hitpos = ray.origin()+(ray.direction()*it);
    if (pt <= ot) {
      //plane hit before object, may be inside it

      //use the following heuristic to determine if we are:
      //  fire a ray backward
      //  if we hit the same object, or a relative then we are inside
      Ray bray;
      HitInfo bhit;      
      bray.set_origin(hitpos);
      bray.set_direction(-1*ray.direction());
      cx->stats->ds[depth].nrays++;

      //use the cut group to test just the related objects, and to treat them
      //as one object (ie visible female sections are all equal)
      Object *cutobj = *((Object **)(hit.scratchpad+CUTGROUPPTR));
      CutGroup* cutgrp = (CutGroup *)hit.hit_obj;
      
      //use sub_intersect instead of intersect to ignore the cutting plane
      cutgrp->sub_intersect(bray, bhit, &cx->stats->ds[depth], cx->worker->get_ppc());
      
      if (bhit.was_hit) {
	//hit a relative, we are inside, try to ColorMap the interior value

	double internal_val = 0;
	//can this object be colored internally?
	if (cutgrp->interior_value(internal_val, ray, it)) {
	  //it can be, map it to a color map, and make some attempy to shade that

	  Color difflight = Color(0,0,0);
	  Color cmapcol = cmap->indexColorMap((float)internal_val);
	  
	  //don't bother shadowing, it's inside an object and will thus be shaded
	  //but react to light color
	  int ngloblights=cx->scene->nlights();
	  int nloclights=my_lights.size();
	  int nlights=ngloblights+nloclights;
	  cx->stats->ds[depth].nshadow+=nlights;
	  for(int i=0;i<nlights;i++){
	    Light* light;
	    if (i<ngloblights)
	      light=cx->scene->light(i);
	    else 
	      light=my_lights[i-ngloblights];
	    
	    if( !light->isOn() )
	      continue;

	    difflight+=light->get_color()*cmapcol*0.6;
	  }
	  result = 
	    //cmapcol*ambient(cx->scene, cutgrp->n) +
	    difflight;

	} else {
	  //nope, color as normal
	  surfmat->shade(result, ray, hit, depth, atten, accumcolor, cx);	
	}
      } else {
	//didn't hit any relatives, color as normal
	surfmat->shade(result, ray, hit, depth, atten, accumcolor, cx);
      }
    } else {
      //obj hit before plane, must not be inside it, color as normal
      surfmat->shade(result, ray, hit, depth, atten, accumcolor, cx);
    }
  }
}





