
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


void UVMapping::get_frame(const Point &, const HitInfo&,const Vector &norm, Vector &v2, Vector &v3)
    {
      Vector v(1,0,0);
      VXV3(v2,norm,v);
      VXV3(v3,norm,v2);
      }
