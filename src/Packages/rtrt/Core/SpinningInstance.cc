#include "SpinningInstance.h"

using namespace rtrt;
using namespace SCIRun;

SpinningInstance::SpinningInstance(InstanceWrapperObject* o, Transform* trans, Point cen, Vector axis, double _rate) 
  : Instance(o, trans), cen(cen), axis(axis)
{
  rate = 2*_rate; //2rad = 1 revolution
  axis.normalize();

  location_trans = new Transform();
  *location_trans = *t;
  location_trans->pre_translate(Point(0,0,0)-cen); 

  o->compute_bounds(bbox_orig,0);
  /*
    The Instance constructor takes care of any initial 
    scale, translate, and rotate. Now we have to sweep out the bbox 
    around cen+axis to account for runtime rotation.
  */
  //find min, and max l(ength), as well as the maximum r(adius) from the bbox corners

  double lmin = MAXDOUBLE;
  double lmax = -MAXDOUBLE;
  double rmax = -MAXDOUBLE;
  double l, r;

  Point b000 = bbox.min();
  Point b111 = bbox.max();

  cerr << "BBMIN " << b000 << endl;
  cerr << "BBMAX " << b111 << endl;

  double x0 = b000.x();
  double y0 = b000.y();
  double z0 = b000.z();
  double z1 = b111.z();
  double y1 = b111.y();
  double x1 = b111.x();
  Point b001(x0,y0,z1);
  Point b010(x0,y1,z0);
  Point b011(x0,y1,z1);
  Point b100(x1,y0,z0);
  Point b101(x1,y0,z1);
  Point b110(x1,y1,z0);

  
  l = Dot(b000-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b000-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  l = Dot(b001-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b001-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  l = Dot(b010-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b010-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  l = Dot(b011-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b011-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  l = Dot(b100-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b100-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  l = Dot(b101-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b101-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  l = Dot(b110-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b110-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  l = Dot(b111-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b111-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;

  //bounding cylinder for rotating object goes from cen+lmin*axis to cen+mlax*axis and is rmax units in radius
  //fit an axis aligned bbox around the cylinder
  Vector a = Cross(axis, Vector(1,0,0));
  Vector b = Cross(axis, Vector(0,1,0));
  Vector c = Cross(axis, Vector(0,0,1));

  Point pmin = cen+lmin*axis;
  Point pmax = cen+lmax*axis;

  Point p;
  bbox.reset();
  p = pmin+rmax*a;
  bbox.extend(p);
  p = pmin-rmax*a;
  bbox.extend(p);
  p = pmin+rmax*b;
  bbox.extend(p);
  p = pmin-rmax*b;
  bbox.extend(p);
  p = pmin+rmax*c;
  bbox.extend(p);
  p = pmin-rmax*c;
  bbox.extend(p);

  p = pmax+rmax*a;
  bbox.extend(p);
  p = pmax-rmax*a;
  bbox.extend(p);
  p = pmax+rmax*b;
  bbox.extend(p);
  p = pmax-rmax*b;
  bbox.extend(p);
  p = pmax+rmax*c;
  bbox.extend(p);
  p = pmax-rmax*c;
  bbox.extend(p);


  b000 = bbox.min();
  b111 = bbox.max();

  cerr << "BBMIN " << b000 << endl;
  cerr << "BBMAX " << b111 << endl;

}

void SpinningInstance::compute_bounds(BBox& b, double /*offset*/)
{
  b.extend(bbox);
}

void SpinningInstance::animate(double time, bool& changed) {
  o->animate(time, changed);

  //There should be a more efficient way to do this, the copies are bad.
  //But seem necessary to prevent degeneration on off angles
  *t=*location_trans;
  //the pretranslate is done in the constructor
  t->pre_rotate(time*rate, axis);
  t->pre_translate(cen-Point(0,0,0));
  changed = true;
}

void SpinningInstance::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
				 PerProcessorContext* ppc)
{
  
  double min_t = hit.min_t;
  if (!bbox.intersect(ray, min_t)) return;	  
  
  Ray tray;
  
  ray.transform(t,tray);
  
  //this give a slight speed improvement 4.2 fps to 4.4 in one test
  //it accounts for the fact that the spinning bbox is larger than original
  min_t = hit.min_t;
  if (!bbox_orig.intersect(tray, min_t)) return;
  
  o->intersect(tray,hit,st,ppc);
  
  // if the ray hit one of our objects....
  if (min_t > hit.min_t)
    {
      InstanceHit* i = (InstanceHit*)(hit.scratchpad);
      Point p = ray.origin() + hit.min_t*ray.direction();
      i->normal = hit.hit_obj->normal(p,hit);
      i->obj = hit.hit_obj;
      hit.hit_obj = this;
    }
}

