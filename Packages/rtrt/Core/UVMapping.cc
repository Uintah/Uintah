
#include <Packages/rtrt/Core/UVMapping.h>
#include <Core/Geometry/Vector.h>


using namespace rtrt;
using namespace SCIRun;

UVMapping::UVMapping()
{
}

UVMapping::~UVMapping()
{
}


void UVMapping::get_frame(const Point &, const HitInfo&,const Vector &norm, Vector &v2, Vector &v3) {
  Vector v(1,0,0);
  v2=Cross(norm,v);
  if (v2.length2()<1.e-8) {
    v=Vector(0,1,0);
    v2=Cross(norm,v);
  }
  v2.normalize();
  v3=Cross(norm,v2);
  v3.normalize();
}
