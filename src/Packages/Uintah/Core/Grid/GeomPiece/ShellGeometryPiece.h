#ifndef __SHELL_GEOMETRY_OBJECT_H__
#define __SHELL_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

#include <math.h>

namespace Uintah {

/**************************************
	
CLASS

   ShellGeometryPiece
	
   Creates a shell from the xml input file description.
	
GENERAL INFORMATION
	
   ShellGeometryPiece.h
	
   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
KEYWORDS

   ShellGeometryPiece  BoundingBox inside
	
DESCRIPTION

   Creates a shell from the xml input file description.
   The input form looks like this:
   
   a) Sphere
   <shell>
     <sphere>
       <origin>[0.,0.,0.]</origin>
       <radius>2.0</radius>
       <thickness>0.1</thickness>
       <num_lat>20</num_lat>
       <num_long>40</num_long>
     </sphere>
   </shell>

   b) Cylinder
   <shell>
     <cylinder>
     </cylinder>
   </shell>
	
   c) Plane
   <shell>
     <plane>
     </plane>
   </shell>

   d) Union/Intersection operations
   <shell>
     <union>
     </union>
   </shell>

   <shell>
     <intersect>
     </intersect>
   </shell>
	
WARNING
	
****************************************/


  class ShellGeometryPiece : public GeometryPiece {
	 
  public:
    //////////
    // Destructor
    virtual ~ShellGeometryPiece();
	 
    //////////
    // Returns the bounding box surrounding the box.
    virtual Box getBoundingBox() const = 0;

    //////////
    // Determines whether a point is inside the shell
    virtual bool inside(const Point &p) const = 0;
	 
    //////////
    // Returns the number of particles associated with the shell
    virtual int returnParticleCount(const Patch* patch) = 0;

    //////////
    // Create the particles in the shell
    virtual int createParticles(const Patch* patch,
				ParticleVariable<Point>&  pos,
				ParticleVariable<double>& vol,
				ParticleVariable<double>& pThickTop,
				ParticleVariable<double>& pThickBot,
				ParticleVariable<Vector>& pNormal,
				ParticleVariable<Vector>& psize,
				particleIndex start) = 0;

  protected:
    ShellGeometryPiece();
    ShellGeometryPiece(ShellGeometryPiece&);

  };
} // End namespace Uintah

#endif // __SHELL_GEOMETRY_PIECE_H__
