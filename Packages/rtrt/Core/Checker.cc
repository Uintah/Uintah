
#include "Checker.h"
#include "Point.h"
#include "Vector.h"
#include "Ray.h"
#include "HitInfo.h"

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
    double uu=u.dot(p);
    double vv=v.dot(p);
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
