//  TetMeshGeom.h - A base class for regular geometries with alligned axes
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_TetMeshGeom_h
#define SCI_project_TetMeshGeom_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomPolyline.h>
#include <Core/Datatypes/MeshGeom.h>
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

class TetMeshGeom:public MeshGeom
{
public:

  TetMeshGeom(const vector<NodeSimp>&, const vector<TetSimp>&);
  ~TetMeshGeom();

  //////////
  // Interpolate
  template <class A>
  int slinterpolate(A* att, elem_t, const Point& p, double& outval,
		    double eps=1.0e-6);

  void set_tets(const vector<TetSimp>&);

 ///////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
 
  vector<TetSimp> tets;

protected:
  bool has_neighbors;

private:
  static DebugStream dbg;
};

template <class A>
int TetMeshGeom::slinterpolate(A* att, elem_t elem_type, const Point& p, double& outval,
				    double eps){
}

} // End namespace SCIRun


#endif
