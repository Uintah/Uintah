#ifndef __CYLINDER_SHELL_PIECE_H__
#define __CYLINDER_SHELL_PIECE_H__

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

   CylinderShellPiece
	
   Creates a cylindrical shell from the xml input file description.
	
GENERAL INFORMATION
	
   CylinderShellPiece.h
	
   Biswajit Banerjee
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
KEYWORDS

   Cylindrical Shell
	
DESCRIPTION

   Creates a cylinder from the xml input file description.
   The input form looks like this:
       <cylinder>
         <bottom>[0.,0.,0.]</bottom>
         <top>[0.,0.,0.]</top>
	 <radius>2.0</radius>
	 <thickness>0.1</thickness>
	 <num_axis>20</num_axis>
	 <num_circum>40</num_circum>
       </cylinder>
	
WARNING
	
****************************************/

  class CylinderShellPiece : public ShellGeometryPiece {
	 
  public:
    //////////
    //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
    // input specification and builds a sphere.
    CylinderShellPiece(ProblemSpecP &);
	 
    //////////
    // Destructor
    virtual ~CylinderShellPiece();
	 
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
	 
    Point  d_bottom;
    Point  d_top;
    double d_radius;
    double d_thickness;
    double d_numAxis;
    double d_numCircum;
  };
} // End namespace Uintah

#endif // __CYLINDER_SHELL_PIECE_H__
