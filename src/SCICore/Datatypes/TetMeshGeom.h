//  TetMeshGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_TetMeshGeom_h
#define SCI_project_TetMeshGeom_h 1

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/GeomPolyline.h>
#include <SCICore/Datatypes/MeshGeom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Util/DebugStream.h>
#include <sstream>
#include <vector>
#include <string>
#include <set>


namespace SCICore{
namespace Datatypes{

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;
using SCICore::GeomSpace::GeomTrianglesP;
using SCICore::GeomSpace::GeomPolyline;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Math::Interpolate;
using SCICore::Util::DebugStream;

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

} // end Datatypes
} // end SCICore


#endif
