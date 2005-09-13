#ifndef ObjReader_h
#define ObjReader_h 1

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>

namespace SCIRun {
  class Point;
  class Transform;
}

namespace rtrt {

using SCIRun::Point;
using SCIRun::Transform;

bool
readObjFile(const string geom_fname, const string matl_fname, 
	    Transform &t, Array1<Material *> &matl, Group *g, 
	    int gridsize=0, Material *m=0);

bool
readObjFile(const string geom_fname, const string matl_fname, 
	    Transform &t, Group *g, int gridsize=0, Material *m=0);
}

#endif
