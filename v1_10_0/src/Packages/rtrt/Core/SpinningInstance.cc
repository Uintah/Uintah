#include "SpinningInstance.h"

using namespace rtrt;
using namespace SCIRun;

SpinningInstance::SpinningInstance(InstanceWrapperObject* o, Transform* trans, Point cen, Vector _axis, double _rate) 
  : Instance(o, trans), cen(cen), dorotate(true), ctime(0)
{
  cpdpy = 0;

  rate = 2*_rate; //2rad = 1 revolution
  axis = _axis.normal();

  //the location trans is the original position, before any spin is applied
  location_trans = new Transform();
  *location_trans = *currentTransform;
  location_trans->pre_translate(Point(0,0,0)-cen); //optimization do this just once

  
  o->compute_bounds(bbox_orig,0);
  //cerr << "OBJ " << bbox_orig.min() << " to " << bbox_orig.max() << endl;
  //cerr << "INS " << bbox.min() << " to " << bbox.max() << endl;

  /*
    The Instance constructor takes care of any initial 
    scale, translate, and rotate. Now we have to sweep out the bbox 
    around cen+axis to account for runtime rotation.
  */
  //find min, and max l(ength), as well as the maximum r(adius) from the bbox corners

  double lmin;
  double lmax;
  double rmax;
  double l, r;

  //bbox has already been transformed to world coords by the Instance constructor
  Point b000 = bbox.min();
  Point b111 = bbox.max();

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

  //cerr << "cen " << cen << endl;
  //cerr << "axis " << axis << endl;

  l = Dot(b000-cen, axis);
  lmin = l;
  lmax = l;
  r = (b000-(cen+(l*axis))).length();
  rmax = r;
  //cerr << "b000" << b000 << " " << l << ", " << r << endl;

  l = Dot(b001-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b001-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;
  //cerr << "b001" << b001 << " " << l << ", " << r << endl;

  l = Dot(b010-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b010-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;
  //cerr << "b010" << b010 << " " << l << ", " << r << endl;

  l = Dot(b011-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b011-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;
  //cerr << "b011" << b011 << " " << l << ", " << r << endl;

  l = Dot(b100-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b100-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;
  //cerr << "b100" << b100 << " " << l << ", " << r << endl;

  l = Dot(b101-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b101-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;
  //cerr << "b101" << b101 << " " << l << ", " << r << endl;

  l = Dot(b110-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b110-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;
  //cerr << "b110" << b110 << " " << l << ", " << r << endl;

  l = Dot(b111-cen, axis);
  if (l < lmin) lmin = l;
  if (l > lmax) lmax = l;
  r = (b111-(cen+(l*axis))).length();
  if (r > rmax) rmax = r;
  //cerr << "b111" << b111 << " " << l << ", " << r << endl;

  //bounding cylinder for rotating object goes from cen+lmin*axis to cen+mlax*axis and is rmax units in radius
  //fit an axis aligned bbox around that cylinder
  Vector a = Cross(axis, Vector(1,0,0));
  Vector b = Cross(axis, Vector(0,1,0));
  Vector c = Cross(axis, Vector(0,0,1));
  a.safe_normalize();
  b.safe_normalize();
  c.safe_normalize();
  
  //cerr << "A,B,C " << a << ", " << b << ", " << c << endl;
  Point pmin = cen+lmin*axis;
  Point pmax = cen+lmax*axis;
  //cerr << "PMIN,PMAX " << pmin << ", " << pmax << endl;

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

  //cerr << "S INS " << bbox.min() << " to " << bbox.max() << endl;

}

void SpinningInstance::compute_bounds(BBox& b, double /*offset*/)
{
  b.extend(bbox);
}

void SpinningInstance::animate(double time, bool& changed) {
  if (dorotate) ctime = time;
  
  o->animate(ctime, changed);

  *currentTransform=*location_trans;
  //the pretranslate to the origin is done in the constructor
  currentTransform->pre_rotate(ctime*rate, axis);
  currentTransform->pre_translate(cen-Point(0,0,0));
  changed = true;  
}

void SpinningInstance::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
				 PerProcessorContext* ppc)
{

  double min_t = hit.min_t;
  if (!bbox.intersect(ray, min_t)) return;	  

  Ray tray;  
  ray.transform(currentTransform,tray);
  Vector td = tray.direction();
  double scale = td.normalize();
  tray.set_direction(td);

  HitInfo thit;
  if (hit.was_hit) thit.min_t = hit.min_t * scale;
  min_t = thit.min_t;

  if (!bbox_orig.intersect(tray, min_t)) return;
  
  o->intersect(tray, thit, st, ppc);
  
  // if the ray hit one of our objects....
  if (thit.was_hit)
    {
      min_t = thit.min_t / scale;
      if(hit.hit(this, min_t)){
	InstanceHit* i = (InstanceHit*)(hit.scratchpad);
	Point p = tray.origin() + thit.min_t*tray.direction();
	i->normal = thit.hit_obj->normal(p,thit);
	i->obj = thit.hit_obj;
	UVMapping * theUV = thit.hit_obj->get_uvmapping();
	theUV->uv(i->uv, p, thit );
      }
    }	      
}

void SpinningInstance::incMagnification()
{
  location_trans->pre_scale(Vector(1.2,1.2,1.2));
}
void SpinningInstance::decMagnification()
{
  location_trans->pre_scale(Vector(.83333,.83333,.83333));
}
void SpinningInstance::upPole()
{
  location_trans->pre_translate(0.1*axis);
}
void SpinningInstance::downPole()
{
  location_trans->pre_translate(-0.1*axis);
}


