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

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class CylinderShellPiece
	
    \brief Creates a cylindrical shell from the xml input file description.
	
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
	
    \warning Symmetry boundary conditions should be applied with care

    Creates a cylinder from the xml input file description.
    The input form looks like this:
    \verbatim
    <cylinder>
      <bottom>[0.,0.,0.]</bottom>
      <top>[0.,0.,0.]</top>
      <radius>2.0</radius>
      <thickness>0.1</thickness>
      <num_axis>20</num_axis>
      <num_circum>40</num_circum>
    </cylinder>
    \endverbatim
	
  */
  /////////////////////////////////////////////////////////////////////////////

  class CylinderShellPiece : public ShellGeometryPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*! \brief  Constructor that takes a ProblemSpecP argument.   
      It reads the xml input specification and builds a cylinder. */
    //////////////////////////////////////////////////////////////////////
    CylinderShellPiece(ProblemSpecP &);
	 
    //////////////////////////////////////////////////////////////////////
    /*! \brief Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~CylinderShellPiece();
	 
    //////////////////////////////////////////////////////////////////////
    /*! \brief Determines whether a point is inside the cylinder.  */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const;
	 
    //////////////////////////////////////////////////////////////////////
    /*! \brief Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const;

    //////////////////////////////////////////////////////////////////////
    /*! \brief Returns the number of particles */
    //////////////////////////////////////////////////////////////////////
    int returnParticleCount(const Patch* patch);

    //////////////////////////////////////////////////////////////////////
    /*! \brief Creates the particles */
    //////////////////////////////////////////////////////////////////////
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
