
#include <Packages/rtrt/Core/UVPlane.h>
#include <Packages/rtrt/Core/UV.h>

using namespace rtrt;
using namespace SCIRun;

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
    double uu=Dot(v1, p);
    double vv=Dot(v2, p);
    uv.set(uu,vv);
}
