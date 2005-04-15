#include "GeomEllipsoid.h"
#include <SCICore/Geometry/BBox.h>
//#include <SCICore/Geometry/BSphere.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace GeomSpace {


Persistent* make_GeomEllipsoid()
{
    return scinew GeomEllipsoid;
}

PersistentTypeID GeomEllipsoid::type_id("GeomEllipsoid", "GeomSphere", make_GeomEllipsoid);

void GeomEllipsoid::get_bounds(BBox& bb)
{
    bb.extend(cen, mev);
}

// void GeomEllipsoid::get_bounds(BSphere& bs)
// {
//     bs.extend(cen, mev*1.000001);
// }
} // End namespace GeomSpace
} // End namespace SCICore
