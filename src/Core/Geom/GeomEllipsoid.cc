#include "GeomEllipsoid.h"
#include <Core/Geometry/BBox.h>
//#include <Core/Geometry/BSphere.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


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
} // End namespace SCIRun
