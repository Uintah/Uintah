#ifndef __SPHERE_MEMBRANE_GEOMETRY_OBJECT_H__
#define __SPHERE_MEMBRANE_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

#include <math.h>

namespace Uintah {

/**************************************
	
CLASS
   SphereMembraneGeometryPiece
	
   Creates a sphere from the xml input file description.
	
GENERAL INFORMATION
	
   SphereMembraneGeometryPiece.h
	
   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   SphereMembraneGeometryPiece  BoundingBox inside
	
DESCRIPTION
   Creates a sphere from the xml input file description.
   Requires five inputs: origin, radius and thickness as well as
   num_lat and num_long.  These last two indicate how many lines of
   latitude and longitude there are that are made up by particles.
   There are methods for checking if a point is inside the sphere
   and also for determining the bounding box for the sphere.
   The input form looks like this:
       <sphere_membrane>
         <origin>[0.,0.,0.]</origin>
	 <radius>2.0</radius>
	 <thickness>0.1</thickness>
	 <num_lat>20</num_lat>
	 <num_long>40</num_long>
       </sphere_membrane>
	
	
WARNING
	
****************************************/


      class SphereMembraneGeometryPiece : public GeometryPiece {
	 
      public:
	 //////////
	 //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
	 // input specification and builds a sphere.
	 SphereMembraneGeometryPiece(ProblemSpecP &);
	 
	 //////////
	 // Destructor
	 virtual ~SphereMembraneGeometryPiece();
	 
	 //////////
	 // Determines whether a point is inside the sphere. 
	 virtual bool inside(const Point &p) const;
	 
	 //////////
	 // Returns the bounding box surrounding the box.
	 virtual Box getBoundingBox() const;

         int returnParticleCount(const Patch* patch);

         int createParticles(const Patch* patch,
                             ParticleVariable<Point>&  pos,
                             ParticleVariable<double>& vol,
                             ParticleVariable<Vector>& pt1,
                             ParticleVariable<Vector>& pt2,
                             ParticleVariable<Vector>& pn,
                             ParticleVariable<Vector>& psize,
                             particleIndex start);


      private:
	 
	 Point  d_origin;
	 double d_radius;
	 double d_h;
	 double d_numLat;
	 double d_numLong;
      };
} // End namespace Uintah

#endif // __SPHERE_MEMBRANE_GEOMETRY_PIECE_H__
