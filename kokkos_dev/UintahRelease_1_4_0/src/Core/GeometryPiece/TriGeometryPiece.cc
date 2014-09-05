/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/GeometryPiece/TriGeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Box.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Ray.h>
#include <Core/Malloc/Allocator.h>

#include   <iostream>
#include   <fstream>

using namespace Uintah;


#define INSIDE_NEW
//#undef INSIDE_NEW

const string TriGeometryPiece::TYPE_NAME = "tri";

TriGeometryPiece::TriGeometryPiece(ProblemSpecP &ps)
{
  name_ = "Unnamed Tri";
  
  ps->require("name",d_file);
  
  readPoints(d_file);
  readTri(d_file);
  makePlanes();
  makeTriBoxes();
  
  // cout << "Triangulated surfaces read: \t" <<d_tri.size() <<endl;

  list<Tri> tri_list;
  Tri tri;

  tri_list = tri.makeTriList(d_tri,d_points);
  d_grid = scinew UniformGrid(d_box);
  d_grid->buildUniformGrid(tri_list);
			      

}

TriGeometryPiece::TriGeometryPiece(const TriGeometryPiece& copy)
{
  d_box = copy.d_box;
  d_points = copy.d_points;
  d_tri = copy.d_tri;
  d_planes = copy.d_planes;
  d_boxes = copy.d_boxes;

  d_grid = scinew UniformGrid(*copy.d_grid);

}

TriGeometryPiece& TriGeometryPiece::operator=(const TriGeometryPiece& rhs)
{
  if (this == &rhs)
    return *this;

  // Clean out lhs

  d_points.clear();
  d_tri.clear();
  d_planes.clear();
  d_boxes.clear();

  delete d_grid;

  // Copy the rhs stuff
  d_box = rhs.d_box;
  d_points = rhs.d_points;
  d_tri = rhs.d_tri;
  d_planes = rhs.d_planes;
  d_boxes = rhs.d_boxes;

  d_grid = scinew UniformGrid(*rhs.d_grid);

  return *this;
}

TriGeometryPiece::~TriGeometryPiece()
{
  delete d_grid;
}

void
TriGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("name",d_file);
}

GeometryPieceP
TriGeometryPiece::clone() const
{
  return scinew TriGeometryPiece(*this);
}

bool
TriGeometryPiece::insideNew(const Point &p,int& cross) const
{
  // Count the number of times a ray from the point p
  // intersects the triangular surface.  If the number
  // of crossings is odd, the point is inside, else it 
  // is outside.

  // Check if Point p is outside the bounding box
  if (!(p == Max(p,d_box.lower()) && p == Min(p,d_box.upper())))
    return false;

  d_grid->countIntersections(p,cross);
  //  cout << "Point " << p << " has " << cross << " crossings " << endl;
  if (cross % 2)
    return true;
  else
    return false;
}


bool
TriGeometryPiece::inside(const Point &p) const
{
  // Count the number of times a ray from the point p
  // intersects the triangular surface.  If the number
  // of crossings is odd, the point is inside, else it 
  // is outside.
#if 0
  Point test_point = Point(.0025,0.0475, 0.0025);
  if (!test_point.InInterval(p,1e-10))
    return false;
#endif

  // Check if Point p is outside the bounding box
  if (!(p == Max(p,d_box.lower()) && p == Min(p,d_box.upper())))
    return false;
#if 1
  int cross_new = 0;
  bool inside_new = insideNew(p,cross_new);

  return inside_new;
#else 
      
  Vector infinity = Vector(1e10,0.,0.) - p.asVector();
  //cerr << "Testing point " << p << endl;
  int crossings = 0, NES = 0;
  for (int i = 0; i < (int) d_planes.size(); i++) {
    int NCS = 0;
    Point hit(0.,0.,0.);
    Plane plane = d_planes[i];
    //cerr << "i = " << i << endl;
    int hit_me = plane.Intersect(p,infinity,hit);
    if (hit_me) {
      // Check if hit point is inside of the triangle
      // Look to see if total angle is 0 or 2PI.  If
      // the point is interior, then angle will be 2PI,
      // else it will be zero.
      // Need to check that the dot product of the intersection pt - p
      // and infinity - p is greater than 0.  This means that the 
      // intersection point is NOT behind the p.
      Vector int_ray = hit.asVector() - p.asVector();
      double cos_angle = Dot(infinity,int_ray)/
	(infinity.length()*int_ray.length());
      if (cos_angle < 0.)
	continue;

      insideTriangle(hit,i,NCS,NES);
      // cerr << "in = " << endl;
      if (NCS % 2 != 0) {
	crossings++;
#  if 0
	cout << "Inside_old hit = " << hit << " vertices: " << 
	  d_points[d_tri[i].x()] << " " << d_points[d_tri[i].y()] <<
	  " " << d_points[d_tri[i].z()] << endl;
#  endif
      }
      if (NES != 0)
	crossings -= NES/2;
    } else
      continue;
  }

  bool crossing_test;
  if (crossings%2 == 0)
    crossing_test = false;
  else
    crossing_test = true;
#  if 0
  if (inside_new != crossing_test) {
    cout << "Point " << p << " have different inside test results" << endl;
    cout << "inside_new = " << inside_new << " crossing_test = " 
	 << crossing_test << endl;
    cout << "cross_new = " << cross_new << " crossings = " << crossings 
	 << endl;
  }
#  endif
  return crossing_test;
#endif
}


Box
TriGeometryPiece::getBoundingBox() const
{
  return d_box;
}

void
TriGeometryPiece::readPoints(const string& file)
{
  string f = file + ".pts";
  std::ifstream source(f.c_str());
  if (!source) {
    std::ostringstream warn;
    warn << "\n ERROR: opening geometry pts points file ("<< f 
         << ").\n  The file must be in the same directory as sus \n"
         << "  Do not enclose the filename in quotation marks\n";
    throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
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


void
TriGeometryPiece::readTri(const string& file)
{
  string f = file + ".tri";
  std::ifstream source(f.c_str());
  if (!source) {
    std::ostringstream warn;
    warn << "\n ERROR: opening geometry tri points file ("<< f 
         << ").\n   The file must be in the same directory as sus"
         << "   Do not enclose the filename in quotation marks\n";
    throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
  }

  int x,y,z;
  while (source >> x >> y >> z) {
    d_tri.push_back(IntVector(x,y,z));
  }
  source.close();
}

void
TriGeometryPiece::makePlanes()
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

void
TriGeometryPiece::makeTriBoxes()
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

void
TriGeometryPiece::insideTriangle( Point& q,int num,int& NCS,
                                  int& NES ) const
{
#if 0
  // Check if the point is inside the bounding box of the triangle.
  if (!(q == Max(q,d_boxes[num].lower()) && q == Min(q,d_boxes[num].upper()))){
    NCS = NES = 0;
    return;
  }
#endif

  // Pulled from makemesh.77.c
  //  Now we have to do the pt_in_pgon test to determine if ri is
  //  inside or outside the triangle.  Use Eric Haines idea in
  //  Essential Ray Tracing Algorithms, pg 53.  Don't have to worry
  //  about whether the intersection point is on the edge or vertex,
  //  cause the edge and/or vertex will be defined away.
  //
  
  // Now we project the ri and the vertices of the triangle onto
  // the dominant coordinate, i.e., the plane's normal largest 
  // magnitude.  
  //
   

  Vector plane_normal = d_planes[num].normal();
  Vector plane_normal_abs = Abs(plane_normal);
  double largest = plane_normal_abs.maxComponent();
  // WARNING: if dominant_coord is not 1-3, then this code breaks...
  int dominant_coord = -1;
  if (largest == plane_normal_abs.x()){
   dominant_coord = 1;
  }
  else if (largest == plane_normal_abs.y()){
    dominant_coord = 2;
  }
  else if (largest == plane_normal_abs.z()){
    dominant_coord = 3;
  }

  if (dominant_coord == -1){
    std::cout << " dominant coordinate not found " << endl;
    throw InternalError("Dominant coordinate not found", __FILE__, __LINE__);
  }
  Point p[3];
  p[0] = d_points[d_tri[num].x()];
  p[1] = d_points[d_tri[num].y()];
  p[2] = d_points[d_tri[num].z()];

  Tri tri(p[0],p[1],p[2]);
  //bool inside = tri.inside(q);
  //  cout << "inside = " << inside << endl;

  // Now translate the points that make up the vertices of the triangle.
  Point trans_pt(0.,0.,0.), trans_vt[3];
  trans_vt[0] = Point(0.,0.,0.);
  trans_vt[1] = Point(0.,0.,0.);
  trans_vt[2] = Point(0.,0.,0.);

  if (dominant_coord == 1) {
    trans_pt.x(q.y());
    trans_pt.y(q.z());
    for (int i = 0; i < 3; i++) {
      trans_vt[i].x(p[i].y());
      trans_vt[i].y(p[i].z());
    }
  } else if (dominant_coord == 2) {
    trans_pt.x(q.x());
    trans_pt.y(q.z());
    for (int i = 0; i < 3; i++) {
      trans_vt[i].x(p[i].x());
      trans_vt[i].y(p[i].z());
    }
  } else if (dominant_coord == 3 ) {
    trans_pt.x(q.x());
    trans_pt.y(q.y());
    for (int i = 0; i < 3; i++) {
      trans_vt[i].x(p[i].x());
      trans_vt[i].y(p[i].y());
    }
  }

  // Now translate the intersecting point to the origin and the vertices
  // as well.

  for (int i = 0; i < 3; i++) 
    trans_vt[i] -= trans_pt.asVector();

  int SH = 0, NSH = 0;
  double out_edge = 0.;

  if (trans_vt[0].y() < 0.0) 
    SH = -1;
  else
    SH = 1;

  if (trans_vt[1].y() < 0.0)
    NSH = -1;
  else
    NSH = 1;

  if (SH != NSH) {
    if ( (trans_vt[0].x() > 0.0) && (trans_vt[1].x() > 0.0) )
      NCS += 1;
    else if ( (trans_vt[0].x() > 0.0) || (trans_vt[1].x() > 0.0) ) {
      out_edge = (trans_vt[0].x() - trans_vt[0].y() * 
		  (trans_vt[1].x() - trans_vt[0].x())/
		  (trans_vt[1].y() - trans_vt[0].y()) );
      if (out_edge == 0.0) {
	NES += 1;
	NCS += 1;
      }
      if (out_edge > 0.0)
	NCS += 1;
    }
    SH = NSH;
  }

  if (trans_vt[2].y() < 0.0)
    NSH = -1;
  else
    NSH = 1;

  if (SH != NSH) {
    if ( (trans_vt[1].x() > 0.0) && (trans_vt[2].x() > 0.0) )
      NCS += 1;
    else if ( (trans_vt[1].x() > 0.0) || (trans_vt[2].x() >0.0) ) {
      out_edge = (trans_vt[1].x() - trans_vt[1].y() * 
		  (trans_vt[2].x() -  trans_vt[1].x())/
		  (trans_vt[2].y() - trans_vt[1].y()) );
      if (out_edge == 0.0){
	NES += 1;
	NCS += 1;
      }
      if (out_edge > 0.0)
	NCS +=1;
    }
    SH = NSH;
  }

  if (trans_vt[0].y() < 0.0)
    NSH = -1;
  else
    NSH = 1;
  
  
  if ( SH != NSH) {
    if ( (trans_vt[2].x() > 0.0) && (trans_vt[0].x() > 0.0) )
      NCS += 1;
    
    else if ( (trans_vt[2].x() > 0.0) || (trans_vt[0].x() >0.0) ) {
      out_edge =  (trans_vt[2].x() - trans_vt[2].y() * 
		   (trans_vt[0].x() - trans_vt[2].x())/
		   (trans_vt[0].y() - trans_vt[2].y()) );
      if (out_edge == 0.0) {
	NES +=1;
	NCS +=1;
      }
      if (out_edge > 0.0)
	NCS += 1;
    }
    SH = NSH;
  }
  
}


void TriGeometryPiece::scale(const double factor)
{
  Vector origin(0.,0.,0.);

  for (vector<Point>::iterator itr = d_points.begin(); itr != d_points.end(); 
       itr++) {
    origin = origin +  itr->asVector();
  }
  origin = origin/(static_cast<double>(d_points.size()));

  for (vector<Point>::iterator itr = d_points.begin(); itr != d_points.end(); 
       itr++) {
    *itr = factor*(*itr - origin) + origin;
  }
}

double TriGeometryPiece::surfaceArea() const
{

  double surfaceArea = 0.;
  for (vector<IntVector>::const_iterator itr = d_tri.begin(); 
       itr != d_tri.end(); itr++) {
    Point pt[3];
    pt[0] = d_points[itr->x()];
    pt[1] = d_points[itr->y()];
    pt[2] = d_points[itr->z()];
    Vector v[2];
    v[0] = pt[0].asVector() - pt[1].asVector();
    v[1] = pt[2].asVector() - pt[1].asVector();

    Vector area = Cross(v[0],v[1]);
    surfaceArea += .5 * area.length();
  }
  return surfaceArea;
}
