//  TriSurfaceGeom.h - A base class for regular geometries with alligned axes
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_TriSurfaceGeom_h
#define SCI_project_TriSurfaceGeom_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomPolyline.h>
#include <Core/Datatypes/SurfaceGeom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
#include <sstream>
#include <vector>
#include <string>
#include <set>


namespace SCIRun {

using std::vector;
using std::string;

class TriSurfaceGeom:public SurfaceGeom
{
public:

  TriSurfaceGeom(const vector<NodeSimp>&, const vector<FaceSimp>&);
  ~TriSurfaceGeom();

   // Interpolate
  template <class A>
  int slinterpolate(A* att, elem_t, const Point& p, double& outval,
		    double eps=1.0e-6);

  void set_faces(const vector<FaceSimp>&);

 ///////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
 
  vector<NodeSimp> nodes;
  vector<FaceSimp> faces;

protected:
  bool has_neighbors;

private:
  static DebugStream dbg;
};

template <class A>
int TriSurfaceGeom::slinterpolate(A* att, elem_t elem_type, const Point& p, double& outval,
				    double eps){
}

} // End namespace SCIRun


#endif
