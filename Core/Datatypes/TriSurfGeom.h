//  TriSurfGeom.h - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_TriSurfGeom_h
#define SCI_project_TriSurfGeom_h 1

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/GeomPolyline.h>
#include <SCICore/Datatypes/ContourGeom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Util/DebugStream.h>
//#include <sstream>
#include <list>
#include <vector>
#include <string>
//#include <set>


namespace SCICore {
namespace Datatypes {


struct TriSurfVertex
{
  int pointIndex() { return d_point_index; }

  int d_point_index;
  int d_neighbor;
  //d_ni;
  //d_ti;
};

class TriSurfGeom : public UnstructuredGeom
{
public:

  TriSurfGeom() : UnstructuredGeom() {}
  virtual ~TriSurfGeom() {}
  
  virtual string getInfo();
  
  ///////////
  // Persistent representation...
  virtual void io(Piostream&) {}
  static PersistentTypeID type_id;

  int pointSize() { return d_points.size(); }
  int edgeSize() { return d_mesh.size(); }
  int triangleSize() { return d_mesh.size() / 3; }

  TriSurfVertex &edge(int index) { return d_mesh[index]; }
  Point &point(int index) { return d_points[index]; }

  int nextEdgeIndex(int e) { return (e + 1) % 3; }
  int prevEdgeIndex(int e) { return (e + 2) % 3; }

  void pushPoint(double a, double b, double c);

  void trivialConnect();

  // TODO: remove this crap from GenField
  bool resize(int x, int y, int z) { return false;}
  void interp(Attrib *a, const Point &p, double &out) {}

protected:
  bool computeBoundingBox();

  vector<Point> d_points;
  vector<TriSurfVertex> d_mesh;  // 3 * number of triangles.
};


PersistentTypeID TriSurfGeom::type_id("TriSurfGeom", "Datatype", 0);


void
TriSurfGeom::pushPoint(double a, double b, double c)
{
  Point p(a, b, c);
  d_points.push_back(p);
}


void
TriSurfGeom::trivialConnect()
{
  for (int i = 0; i < d_points.size(); i++)
  {
    TriSurfVertex v;
    v.d_neighbor = -1;
    v.d_point_index = i;
    d_mesh.push_back(v);
  }
}

bool
TriSurfGeom::computeBoundingBox()
{
  d_bbox.reset();
  if (d_points.empty())
  {
    return false;
  }
  else
  {
    int i;
    for (i=0; i< d_points.size(); i++)
    {
      d_bbox.extend(d_points[i]);
    }
    return true;
  }
}


string
TriSurfGeom::getInfo()
{
  ostringstream retval;

  retval << 
    "Type = " << "TriSurfGeom" << endl <<
    "Pointsize " << pointSize() << endl <<
    "Edgesize " << edgeSize() << endl;

  return retval.str();
}

} // end Datatypes
} // end SCICore


#endif


