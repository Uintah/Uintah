#include <Packages/Uintah/Core/Grid/TriGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Ray.h>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;

TriGeometryPiece::TriGeometryPiece(ProblemSpecP &ps)
{
  std::string file;

  ps->require("name",file);
  
  readPoints(file);
  readTri(file);
  makePlanes();
  makeTriBoxes();

}

TriGeometryPiece::~TriGeometryPiece()
{
}

bool TriGeometryPiece::inside(const Point &p) const
{
  // Count the number of times a ray from the point p
  // intersects the triangular surface.  If the number
  // of crossings is odd, the point is inside, else it 
  // is outside.

  Vector infinity(-1e10,0,0);
  
  int crossings = 0;
  for (int i = 0; i < (int) d_planes.size(); i++) {
    Point hit;
    Plane plane = d_planes[i];
    int hit_me = plane.Intersect(p,infinity,hit);
    if (hit_me) {
      // Check if hit point is inside of the triangle
      // Look to see if total angle is 0 or 2PI.  If
      // the point is interior, then angle will be 2PI,
      // else it will be zero.
      bool in = insideTriangle(hit,i);
      if (in)
	crossings++;
    } else
      continue;
  }

  if (crossings%2 == 0)
    return false;
  else 
    return true;
}

Box TriGeometryPiece::getBoundingBox() const
{
  return d_box;
}

void TriGeometryPiece::readPoints(const string& file)
{
  string f = file + ".pts";
  ifstream source(f.c_str());
  if (!source) {
    throw ProblemSetupException("ERROR: opening MPM Tri points file: \n The file must be in the same directory as sus");
  }

  double x,y,z;
  while (source >> x >> y >> z) {
    d_points.push_back(Point(x,y,z));
  }

  source.close();

  // Find the min and max points so that the bounding box can be determined.
  Point min(1e30,1e30,1e30),max(-1e30,-1e30,-1e30);
  vector<Point>::const_iterator itr;
  for (itr = d_points.begin(); itr != d_points.end(); ++itr) {
    min = Min(*itr,min);
    max = Max(*itr,max);
  }
  Vector fudge(1.e-5,1.e-5,1.e-5);
  min = min - fudge;
  max = max + fudge;
  d_box = Box(min,max);
}


void TriGeometryPiece::readTri(const string& file)
{
  string f = file + ".tri";
  ifstream source(f.c_str());
  if (!source) {
    throw ProblemSetupException("ERROR: opening MPM Tri file: \n The file must be in the same directory as sus");
  }

  int x,y,z;
  while (source >> x >> y >> z) {
    d_tri.push_back(IntVector(x,y,z));
  }

  source.close();


}

void TriGeometryPiece::makePlanes()
{

  for (int i = 0; i < (int) d_tri.size(); i++) {
    Point pt[3];
    IntVector tri = d_tri[i];
    pt[0] = d_points[tri.x()];
    pt[1] = d_points[tri.y()];
    pt[2] = d_points[tri.z()];
    Plane plane(pt[0],pt[1],pt[2]);
    d_planes.push_back(plane);
  }

}

void TriGeometryPiece::makeTriBoxes()
{

  for (int i = 0; i < (int) d_tri.size(); i++) {
    Point pt[3];
    IntVector tri = d_tri[i];
    pt[0] = d_points[tri.x()];
    pt[1] = d_points[tri.y()];
    pt[2] = d_points[tri.z()];
    Point min=Min(Min(pt[0],pt[1]),Min(pt[1],pt[2]));
    Point max=Max(Max(pt[0],pt[1]),Max(pt[1],pt[2]));
    Box box(min,max);
    d_boxes.push_back(box);
  }

}



bool TriGeometryPiece::insideTriangle(const Point& q,int num) const
{
  if (!(q == Max(q,d_boxes[num].lower()) && q == Min(q,d_boxes[num].upper())))
    return false;
       
  Vector u[3],v[3],qvector = q.asVector();
  Point p[3];
  
  p[0] = d_points[d_tri[num].x()];
  p[1] = d_points[d_tri[num].y()];
  p[2] = d_points[d_tri[num].z()];
  
  double anglesum = 0.;
  double TWOPI = 6.283185307179586476925287;
  for (int i = 0; i < 3; i++) {
    u[i] = Vector(p[i]) - qvector;
    v[i] = Vector(p[(i+1)%3]) - qvector;
    anglesum += acos(Dot(u[i],v[i])/(u[i].length() * v[i].length()));
  }
  if (anglesum == TWOPI)
    return true;
  else
    return false;
}

 
