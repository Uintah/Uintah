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
  int pointIndex() { return d_point_index; }
  int pointIndex(int i) { return d_point_index = i; }
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
  static string typeName();

  int pointSize() { return d_points.size(); }
  int edgeSize() { return d_mesh.size(); }
  int triangleSize() { return d_mesh.size() / 3; }

  TriSurfVertex &edge(int index) { return d_mesh[index]; }
  Point &point(int index) { return d_points[index]; }

  int nextEdgeIndex(int e) { return (e%3==2)?(e - 2):(e+1);}
  int prevEdgeIndex(int e) { return (e%3==0)?(e + 2):(e-1);}

  void pushPoint(double a, double b, double c);

  void trivialConnect();
  void remove_duplicates(vector<Point> old_points,
			 vector<Point> new_points,
			 vector<int> mapping);
  void collapse_points();

  // TODO: remove this crap from GenField
  bool resize(int, int, int) { return false;}
  void interp(Attrib *, const Point &, double &) {}

protected:
  bool computeBoundingBox();

  vector<Point> d_points;
  vector<TriSurfVertex> d_mesh;  // 3 * number of triangles.
};


string TriSurfGeom::typeName(){
  static string typeName = "TriSurfGeom";
  return typeName;
}

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


// TODO: make faster, not O(n^2).  Use better equality test for points.
void
TriSurfGeom::remove_duplicates(vector<Point> old_points,
			       vector<Point> new_points,
			       vector<int> mapping)
{
  int i, j;

  for (i=0; i < old_points.size(); i++)
  {
    for (j = 0; j < new_points.size(); j++)
    {
      if (old_points[i] == new_points[j])
      {
	break;
      }
    }
    if (j == new_points.size())
    {
      new_points.push_back(old_points[i]);
    }
    mapping[i] = j;
  }
}


void
TriSurfGeom::collapse_points()
{
  int i;
  vector<Point> new_points;
  vector<int> index;

  // Find the subset, dump it int d_new_points and the mapping in index
  remove_duplicates(d_points, new_points, index);

  // Set the old points to be the new points.
  d_points = new_points;

  // Fix all the references to the old points.
  for (i = 0; i < d_mesh.size(); i++)
  {
    d_mesh[i].pointIndex(index[d_mesh[i].pointIndex()]);
  }
}


#if 0
void
clockwise_faces(cell)
{
  const int a = cell * 4 + 0;
  const int b = cell * 4 + 1;
  const int c = cell * 4 + 2;
  const int d = cell * 4 + 3;

  (a, b, c); //	  a b c;
  (d, c, b); //	  b c d;
  (c, d, a); //	  c d a;
  (b, a, d); //	  d a b;
}


void
anticlockwise_faces(cell)
{
  const int a = cell * 4 + 0;
  const int b = cell * 4 + 1;
  const int c = cell * 4 + 2;
  const int d = cell * 4 + 3;

  (c, b, a); //   a b c;
  (b, c, d); //   b c d;
  (a, d, c); //   c d a;
  (d, a, b); //   d a b;
}
#endif


} // End namespace SCIRun


#endif


