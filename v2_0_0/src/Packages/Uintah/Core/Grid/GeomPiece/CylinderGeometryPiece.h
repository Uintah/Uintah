#ifndef __CYLINDER_GEOMETRY_OBJECT_H__
#define __CYLINDER_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/**************************************
	
CLASS
   CylinderGeometryPiece
	
   Creates a generalized cylinder from the xml input file description.
	
GENERAL INFORMATION
	
   CylinderGeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   CylinderGeometryPiece BoundingBox inside
	
DESCRIPTION
   Creates a generalized cylinder from the xml input file description.
   Requires three inputs: bottom point, top point and a radius.  
   There are methods for checking if a point is inside the cylinder
   and also for determining the bounding box for the cylinder.
   The input form looks like this:
       <cylinder>
         <bottom>[0.,0.,0.]</bottom>
	 <top>[0.,0.,0.]</top>
	 <radius>2.0</radius>
       </cylinder>
	
WARNING
	
****************************************/

      class CylinderGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 // Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a generalized cylinder.
         //
	 CylinderGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 // Constructor that takes top, bottom and radius
	 //
	 CylinderGeometryPiece(const Point& top, 
			       const Point& bottom,
			       double radius);

	 //////////
	 // Destructor
         //
	 virtual ~CylinderGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the cylinder.
         //
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the cylinder.
	 virtual Box getBoundingBox() const;
	 
	 //////////
	 // Calculate the surface area
         //
	 virtual inline double surfaceArea() const
	 {
	   return ((2.0*M_PI*d_radius)*height());
	 }

	 //////////
	 // Calculate the volume
         //
	 virtual inline double volume() const
	 {
	   return ((M_PI*d_radius*d_radius)*height());
	 }

	 //////////
	 // Calculate the unit normal vector to axis from point
         //
	 Vector radialDirection(const Point& pt) const;

	 //////////
	 // Get the top, bottom, radius, height
	 //
	 inline Point top() const {return d_top;}
	 inline Point bottom() const {return d_bottom;}
	 inline double radius() const {return d_radius;}
	 inline double height() const { return (d_top-d_bottom).length();}

      protected:
         
         //////////
         // Constructor needed for subclasses
         //
	 CylinderGeometryPiece();
	 Point d_bottom;
	 Point d_top;
	 double d_radius;
	 
	 
	 
      };
} // End namespace Uintah
      
#endif // __CYLINDER_GEOMTRY_Piece_H__
