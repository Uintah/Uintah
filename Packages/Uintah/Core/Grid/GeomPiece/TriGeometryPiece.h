#ifndef __TRI_GEOMETRY_OBJECT_H__
#define __TRI_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Plane.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

using std::vector;

namespace Uintah {

using namespace SCIRun;

/**************************************
	
CLASS
   TriGeometryPiece
	
   Creates a triangulated surface piece from the xml input file description.
	
GENERAL INFORMATION
	
   TriGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   TriGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a triangulated surface piece from the xml input file description.
   Requires one input: file name (convetion use suffix .dat).  
   There are methods for checking if a point is inside the surface
   and also for determining the bounding box for the surface.
   The input form looks like this:
       <tri>
         <file>surface.dat</file>
       </tri>
	
	
WARNING
	
****************************************/

      class TriGeometryPiece : public GeometryPiece {
      public:
	 //////////
	 //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds the triangulated surface piece.
	 TriGeometryPiece(ProblemSpecP &);
	 //////////
	 
	 // Destructor
	 virtual ~TriGeometryPiece();
	 
	 //////////
	 // Determins whether a point is inside the triangulated surface.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the triangulated surface.
	 virtual Box getBoundingBox() const;
	 
      private:
	 void readPoints(const string& file);
	 void readTri(const string& file);
	 void makePlanes();
	 void makeTriBoxes();
	 void insideTriangle(const Point& p, int i, int& NCS, int& NES) const;
	 
	 Box d_box;
	 vector<Point> d_points;
	 vector<IntVector> d_tri;
	 vector<Plane> d_planes;
	 vector<Box> d_boxes;
	 
      };
} // End namespace Uintah

#endif // __TRI_GEOMETRY_PIECE_H__
