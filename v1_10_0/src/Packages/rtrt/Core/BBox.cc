
#include <Packages/rtrt/Core/BBox.h>
#include <values.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* bBox_maker() {
  return new BBox();
}

// initialize the static member type_id
PersistentTypeID BBox::type_id("BBox", "Object", bBox_maker);

BBox::BBox(const BBox& copy)
    : cmin(copy.cmin), cmax(copy.cmax), have_some(copy.have_some)
{
}

void BBox::reset() {
    have_some=false;
}

Point BBox::center() const {
    return cmin+(cmax-cmin)*0.5;
}

bool BBox::intersect_nontrivial(const Ray& ray, double &min_t) {
  double Tnear = -MAXDOUBLE;
  double Tfar = MAXDOUBLE;
  double ray_dir[3];
  double bbox_min[3];
  double bbox_max[3];
  double ray_origin[3];
  ray_dir[0]=ray.direction().x();
  ray_dir[1]=ray.direction().y();
  ray_dir[2]=ray.direction().z();
  ray_origin[0]=ray.origin().x();
  ray_origin[1]=ray.origin().y();
  ray_origin[2]=ray.origin().z();
  bbox_min[0]=cmin.x(); bbox_min[1]=cmin.y(); bbox_min[2]=cmin.z();
  bbox_max[0]=cmax.x(); bbox_max[1]=cmax.y(); bbox_max[2]=cmax.z();
  
  for (int axis=0; axis<3; axis++) {
    if (ray_dir[axis] == 0) {
      if (ray_origin[axis] > bbox_max[axis] || 
	  ray_origin[axis] < bbox_min[axis]) return false;
    } else {
      double T1 = (bbox_min[axis] - ray_origin[axis])/ray_dir[axis];
      double T2 = (bbox_max[axis] - ray_origin[axis])/ray_dir[axis];
      if (T1 > T2) {
	double tmp=T2;
	T2=T1;
	T1=tmp;
      }
      if (T1 > Tnear) Tnear = T1;
      if (T2 < Tfar) Tfar = T2;
      if (Tnear > Tfar) return false;
      if (Tfar < 0) return false;
    }
  }
  if (Tnear < min_t) {min_t = Tnear; return true;}
  return false;
}


const int BBOX_VERSION = 1;

void 
BBox::io(SCIRun::Piostream &str)
{
  str.begin_class("BBox", BBOX_VERSION);
  Pio(str, cmin);
  Pio(str, cmax);
  Pio(str, have_some);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::BBox*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::BBox::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::BBox*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

