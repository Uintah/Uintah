#ifndef __UNIFORM_GRID_H__
#define __UNIFORM_GRID_H__

#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Plane.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <list>
#include <sgi_stl_warnings_on.h>

using std::vector;
using std::list;

namespace Uintah {

using namespace SCIRun;

/**************************************
	
CLASS
   UniformGrid
	
   ...not sure what it does....
	
GENERAL INFORMATION
	
   UniformGrid.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
 
	
KEYWORDS
   UniformGrid
	
DESCRIPTION
	
WARNING
	
****************************************/


 class Tri {

 public:
   enum coord {X=0,Y=1,Z=2};

   Tri(Point& p1, Point& p2, Point& p3);
   Tri();
   ~Tri();
   Point centroid();
   Point vertex(int i);
   list<Tri> makeTriList(vector<IntVector>& tris, vector<Point>& pts);
   bool inside(Point& p);
   Plane plane();
 private:
   Point d_points[3];
   Plane d_plane;
 };
 
 class UniformGrid {
   
 public:
   UniformGrid(Box& bound_box);
   ~UniformGrid();
   IntVector cellID(Point point);
   void buildUniformGrid(list<Tri>& polygons);
   void countIntersections(const Point& ray, int& crossings);
      
 private:
   Array3<list<Tri> > d_grid;
   Box d_bound_box;
   Vector d_max_min;
 };


} // End namespace Uintah

#endif // __UNIFORM_GRID_H__
