//  TriSurfGeom.h - A base class for regular geometries with alligned axes
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   November 2000
//  Copyright (C) 2000 SCI Institute

#include <Core/Datatypes/TriSurfGeom.h>

namespace SCIRun {


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

} // end namespace SCIRun
