//  MeshGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_MeshGeom_h
#define SCI_project_MeshGeom_h 1

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Datatypes/UnstructuredGeom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Util/DebugStream.h>

#include <sstream>
#include <vector>
#include <string>

namespace SCICore{
namespace Datatypes{

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Min;
using SCICore::Geometry::Max;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Math::Interpolate;
using SCICore::Util::DebugStream;

class MeshGeom:public UnstructuredGeom
{
public:

  MeshGeom();
  ~MeshGeom();

  virtual string get_info();
  
  //////////
  // Compute the bounding box and diagnal, set has_bbox to true
  virtual bool compute_bbox();

  //////////
  // Interpolate
  template <class A>
  int slinterpolate(A* att, elem_t, const Point& p, double& outval,
		    double eps=1.0e-6);

  void add_points(const vector<Point>&);
  void add_tets(const vector<Tetrahedral>&);
  
  void clear();
  
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  
protected:
  bool has_bbox;
  vector<Node> nodes;
  vector<Tetrahedral> tets;

private:
  static DebugStream dbg;
};

template <class A>
int MeshGeom::slinterpolate(A* att, elem_t elem_type, const Point& p, double& outval,
				    double eps){
}

} // end Datatypes
} // end SCICore


#endif
