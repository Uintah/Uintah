#ifndef __SPHERE_SHELL_PIECE_H__
#define __SPHERE_SHELL_PIECE_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/ShellGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

#include <math.h>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/**************************************
	
CLASS

   SphereShellPiece
	
   Creates a spherical shell from the xml input file description.
	
GENERAL INFORMATION
	
   SphereShellPiece.h
	
   Jim Guilkey
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
KEYWORDS

   SphereShellPiece  BoundingBox 
	
DESCRIPTION

   Creates a sphere from the xml input file description.
   Requires five inputs: origin, radius and thickness as well as
   num_lat and num_long.  These last two indicate how many lines of
   latitude and longitude there are that are made up by particles.
   There are methods for checking if a point is inside the sphere
   and also for determining the bounding box for the sphere.
   The input form looks like this:
       <sphere>
         <origin>[0.,0.,0.]</origin>
	 <radius>2.0</radius>
	 <thickness>0.1</thickness>
	 <num_lat>20</num_lat>
	 <num_long>40</num_long>
       </sphere>
	
WARNING
	
****************************************/

  class SphereShellPiece : public ShellGeometryPiece {
	 
  public:
    //////////
    //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
    // input specification and builds a sphere.
    SphereShellPiece(ProblemSpecP &);
	 
    //////////
    // Destructor
    virtual ~SphereShellPiece();
	 
    //////////
    // Determines whether a point is inside the sphere. 
    virtual bool inside(const Point &p) const;
	 
    //////////
    // Returns the bounding box surrounding the box.
    virtual Box getBoundingBox() const;

    //////////
    // Returns the number of particles
    int returnParticleCount(const Patch* patch);

    //////////
    // Creates the particles
    int createParticles(const Patch* patch,
			ParticleVariable<Point>&  pos,
			ParticleVariable<double>& vol,
			ParticleVariable<double>& pThickTop,
			ParticleVariable<double>& pThickBot,
			ParticleVariable<Vector>& pNormal,
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

#endif // __SPHERE_SHELL_PIECE_H__
