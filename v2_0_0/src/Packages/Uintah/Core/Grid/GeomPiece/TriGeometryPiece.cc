#include <Packages/Uintah/Core/Grid/GeomPiece/TriGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Ray.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace std;

TriGeometryPiece::TriGeometryPiece(ProblemSpecP &ps)
{
  setName("tri");
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

  Vector infinity = Vector(-1e10,0.,0.) - p.asVector();
  // cerr << "Testing point " << p << endl;
  int crossings = 0, NES = 0;
  for (int i = 0; i < (int) d_planes.size(); i++) {
    int NCS = 0;
    Point hit;
    Plane plane = d_planes[i];
    int hit_me = plane.Intersect(p,infinity,hit);
    if (hit_me) {
      // Check if hit point is inside of the triangle
      // Look to see if total angle is 0 or 2PI.  If
      // the point is interior, then angle will be 2PI,
      // else it will be zero.
      //  cerr << "Hit me" << endl;
      // Need to check that the dot product of the intersection pt - p
      // and infinity - p is greater than 0.  This means that the 
      // intersection point is NOT behind the p.
      Vector int_ray = hit.asVector() - p.asVector();
      double cos_angle = Dot(infinity,int_ray)/
	(infinity.length()*int_ray.length());
      //cerr << "cos_angle = " << cos_angle << endl;
      if (cos_angle < 0.)
	continue;

      insideTriangle(hit,i,NCS,NES);
      // cerr << "in = " << in << endl;
      if (NCS % 2 != 0)
	crossings++;
      if (NES != 0)
	crossings -= NES/2;
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

void TriGeometryPiece::insideTriangle(const Point& q,int num,int& NCS,
				      int& NES) const
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
  int dominant_coord;
  if (largest == plane_normal_abs.x()) dominant_coord = 1;
  else if (largest == plane_normal_abs.y()) dominant_coord = 2;
  else if (largest == plane_normal_abs.z()) dominant_coord = 3;

  Point p[3];
  p[0] = d_points[d_tri[num].x()];
  p[1] = d_points[d_tri[num].y()];
  p[2] = d_points[d_tri[num].z()];

  // Now translate the points that make up the vertices of the triangle.
  Point trans_pt, trans_vt[3];

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
