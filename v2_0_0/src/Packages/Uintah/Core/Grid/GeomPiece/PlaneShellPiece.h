#ifndef __PLANE_SHELL_PIECE_H__
#define __PLANE_SHELL_PIECE_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/ShellGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

#include <math.h>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/////////////////////////////////////////////////////////////////////////////
/*! 
  \class PlaneShellPiece
	
  \brief Creates a plane shell from the xml input file description.
	
  \author  Biswajit Banerjee \n
   C-SAFE and Department of Mechanical Engineering \n
   University of Utah \n
	
   Creates a plane circle from the xml input file description.
   The input form looks like this:
   \verbatim
     <plane>
       <center>[0.,0.,0.]</center>
       <normal>[0.,0.,1.]</normal>
       <radius>2.0</radius>
       <thickness>0.1</thickness>
       <num_radial>20</num_radial>
     </plane>
   \endverbatim
	
   \warning Needs to be converted into the base class for classes such as
   TriShellPiece, QuadShellPiece, HexagonShellPiece etc.  Currently
   provides implementation for Rectangular Shell Piece.

*/
/////////////////////////////////////////////////////////////////////////////
	
  class PlaneShellPiece : public ShellGeometryPiece {
	 
  public:
    //////////
    //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
    // input specification and builds a plane.
    PlaneShellPiece(ProblemSpecP &);
	 
    //////////
    // Destructor
    virtual ~PlaneShellPiece();
	 
    //////////
    // Determines whether a point is inside the plane. 
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
	 
    Point  d_center;
    Vector d_normal;
    double d_radius;
    double d_thickness;
    int d_numRadius;
  };
} // End namespace Uintah

#endif // __PLANE_SHELL_PIECE_H__
