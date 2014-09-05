#ifndef __NULL_GEOMETRY_OBJECT_H__
#define __NULL_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

/**************************************
	
CLASS
   NullGeometryPiece
	
   Reads in a set of points and optionally a volume for each point from an
   input text file.
	
GENERAL INFORMATION
	
   NullGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   NullGeometryPiece BoundingBox inside
	
DESCRIPTION
   Reads in a set of points from an input file.  Optionally, if the
   <var> tag is present, the volume will be read in for each point set.
   Requires one input: file name <name>points.txt</name>
   Optional input : <var>p.volume </var>
   There are methods for checking if a point is inside the box
   and also for determining the bounding box for the box (which
   just returns the box itself).
   The input form looks like this:
       <name>file_name.txt</name>
         <var>p.volume</var>
	
	
WARNING
	
****************************************/


      class NullGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a generalized box.
	 NullGeometryPiece(ProblemSpecP&);

	 //////////
	 // Construct a box from a min/max point
	 NullGeometryPiece(const string& file_name);
	 
	 //////////
	 // Destructor
	 virtual ~NullGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the box.
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 //  Returns the bounding box surrounding the cylinder.
	 virtual Box getBoundingBox() const;

      private:


	 Box d_box;
	 
      };
} // End namespace Uintah

#endif // __NULL_GEOMTRY_Piece_H__
