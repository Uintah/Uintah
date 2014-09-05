
#include "UVPlane.h"
#include "UV.h"

using namespace rtrt;

UVPlane::UVPlane(const Point& cen, const Vector& v1, const Vector& v2)
    : cen(cen), v1(v1), v2(v2)
{
}

UVPlane::~UVPlane()
{
}

void UVPlane::uv(UV& uv, const Point& hitpos, const HitInfo&)
{
    Vector p(hitpos-cen);
    double uu=v1.dot(p);
    double vv=v2.dot(p);
    uv.set(uu,vv);
}
