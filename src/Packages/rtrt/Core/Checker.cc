
#include <Packages/rtrt/Core/Checker.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>

using namespace rtrt;

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
