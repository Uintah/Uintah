#ifndef ASEReader_h
#define ASEReader_h 1

#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Material.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>

namespace SCIRun {
  class Point;
  class Transform;
}

namespace rtrt {

using SCIRun::Point;
using SCIRun::Transform;

bool
readASEFile(const string fname, const Transform t, Group *objgroup, 
	    Array1<Material*> &ase_matls, string &env_map, bool stadium=false);
}

#endif
