//  TriSurfGeom.h - A base class for regular geometries with alligned axes
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   November 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_TriSurfGeom_h
#define SCI_project_TriSurfGeom_h 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/GeomPolyline.h>
#include <Core/Datatypes/ContourGeom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>
//#include <sstream>
#include <list>
#include <vector>
#include <string>
//#include <set>


namespace SCIRun {

class TriSurfGeom;
typedef LockingHandle<TriSurfGeom> TriSurfGeomHandle;

struct TriSurfVertex
{
  int pointIndex() { return point_index_; }
  int pointIndex(int i) { return point_index_ = i; }
  int point_index_;
  int neighbor_;
  //ni_;
  //ti_;
};

class TriSurfGeom : public UnstructuredGeom
{
public:

  TriSurfGeom() : UnstructuredGeom() {}
  virtual ~TriSurfGeom() {}
  
  virtual string getInfo();
  virtual string getTypeName(int=0);

  ///////////
  // Persistent representation...
  virtual void io(Piostream&) {}
  static PersistentTypeID type_id;
  static string typeName(int);

  int pointSize() { return points_.size(); }
  int edgeSize() { return mesh_.size(); }
  int triangleSize() { return mesh_.size() / 3; }

  TriSurfVertex &edge(int index) { return mesh_[index]; }
  Point &point(int index) { return points_[index]; }

  int nextEdgeIndex(int e) { return (e%3==2)?(e - 2):(e+1);}
  int prevEdgeIndex(int e) { return (e%3==0)?(e + 2):(e-1);}

  void pushPoint(double a, double b, double c);

  void trivialConnect();
  void remove_duplicates(vector<Point> old_points,
			 vector<Point> new_points,
			 vector<int> mapping);
  void collapse_points();

  // TODO: remove this crap from GenField
  void interp(Attrib *, const Point &, double &) {}

protected:
  bool computeBoundingBox();

  vector<Point> points_;
  vector<TriSurfVertex> mesh_;  // 3 * number of triangles.
};


} // End namespace SCIRun


#endif


