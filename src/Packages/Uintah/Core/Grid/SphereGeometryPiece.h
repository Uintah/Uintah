#ifndef __SPHERE_GEOMETRY_OBJECT_H__
#define __SPHERE_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Core/Geometry/Point.h>

#include <math.h>

namespace Uintah {

/**************************************
	
CLASS
   SphereGeometryPiece
	
   Creates a sphere from the xml input file description.
	
GENERAL INFORMATION
	
   SphereGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   SphereGeometryPiece  BoundingBox inside
	
DESCRIPTION
   Creates a sphere from the xml input file description.
   Requires two inputs: origin and a radius.  
   There are methods for checking if a point is inside the sphere
   and also for determining the bounding box for the sphere.
   The input form looks like this:
       <sphere>
         <origin>[0.,0.,0.]</origin>
	 <radius>2.0</radius>
       </sphere>
	
	
WARNING
	
****************************************/


      class SphereGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a sphere.
	 SphereGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 //  Constructor that takes a origin and radius
	 SphereGeometryPiece(const Point& origin, double radius);
	 
	 //////////
	 // Destructor
	 virtual ~SphereGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the sphere. 
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the box.
	 virtual Box getBoundingBox() const;
	 
	 //////////
	 // Get the center and radius
	 //
	 inline Point origin() const {return d_origin;}
	 inline double radius() const {return d_radius;}

      private:
	 
	 Point d_origin;
	 double d_radius;
      };
} // End namespace Uintah

#endif // __SPHERE_GEOMETRY_PIECE_H__
