
#include <Packages/rtrt/Core/Checker.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* checker_maker() {
  return new Checker();
}

// initialize the static member type_id
PersistentTypeID Checker::type_id("Checker", "Material", checker_maker);


Checker::Checker(Material* matl0, Material* matl1,
		 const Vector& u, const Vector& v)
    : matl0(matl0), matl1(matl1), u(u), v(v)
{
}

Checker::~Checker()
{
}

void Checker::shade(Color& result, const Ray& ray,
		    const HitInfo& hit, int depth, 
		    double atten, const Color& accumcolor,
		    Context* cx)
{
    Point p(ray.origin()+ray.direction()*hit.min_t);
    double uu=Dot(u, p);
    double vv=Dot(v, p);
    if(uu<0)
	uu=-uu+1;
    if(vv<0)
	vv=-vv+1;
    int i=(int)uu;
    int j=(int)vv;
    int m=(i+j)%2;
    (m==0?matl0:matl1)->shade(result, ray, hit, depth, atten,
			      accumcolor, cx);
}

const int CHECKER_VERSION = 1;

void 
Checker::io(SCIRun::Piostream &str)
{
  str.begin_class("Checker", CHECKER_VERSION);
  Material::io(str);
  SCIRun::Pio(str, matl0);
  SCIRun::Pio(str, matl1);
  SCIRun::Pio(str, u);
  SCIRun::Pio(str, v);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Checker*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Checker::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Checker*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
