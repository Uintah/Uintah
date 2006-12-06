#ifndef TRISREADER_H
#define TRISREADER_H

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Group.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/SmallTri.h>
#include <Packages/rtrt/Core/Tri.h>

namespace rtrt {

Group* readtris(char *fname, Material* matl);
}
#endif
