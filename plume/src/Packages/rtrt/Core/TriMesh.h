#ifndef TRIMESH_H
#define TRIMESH_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
class TriMesh;
}

namespace rtrt {

class TriMesh 
  {
  public:
    Array1<Point> verts;
    Array1<Vector> norms;
    Array1<Color> colors;

    inline void transform(Transform& T)
      {
	for (int i=0; i<verts.size(); i++)
	  T.project_inplace(verts[i]);
	for (int i=0; i<norms.size(); i++) {
	  T.project_normal_inplace(norms[i]);
	  norms[i].normalize();
	}
      }
  };
}
#endif
