/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __UNIFORM_GRID_H__
#define __UNIFORM_GRID_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Plane.h>
#include <vector>
#include <list>

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
   UniformGrid& operator=(const UniformGrid&);
   UniformGrid(const UniformGrid&);
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
