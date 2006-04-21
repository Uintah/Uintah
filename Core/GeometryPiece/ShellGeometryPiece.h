#ifndef __SHELL_GEOMETRY_OBJECT_H__
#define __SHELL_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>

#include <math.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ShellGeometryPiece
	
    \brief Abstract class for shell geometries
	
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
	
    Creates a shell from the xml input file description.
    The input form looks like this:
   
    \verbatim
    a) Sphere
      <sphere_shell>
        <origin>[0.,0.,0.]</origin>
        <radius>2.0</radius>
        <thickness>0.1</thickness>
        <num_lat>20</num_lat>
        <num_long>40</num_long>
      </sphere_shell>

    b) Cylinder
      <cylinder_shell>
      </cylinder_shell>
	
    c) Plane
      <plane_shell>
      </plane_shell>

    \endverbatim
	
  */
  /////////////////////////////////////////////////////////////////////////////

  class ShellGeometryPiece : public GeometryPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*! \brief Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~ShellGeometryPiece();

    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const = 0;
	 
    //////////////////////////////////////////////////////////////////////
    /*! \brief Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const = 0;

    //////////////////////////////////////////////////////////////////////
    /*! \brief Determines whether a point is inside the shell */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const = 0;
	 
    //////////////////////////////////////////////////////////////////////
    /*! \brief Returns the number of particles associated with the shell */
    //////////////////////////////////////////////////////////////////////
    virtual int returnParticleCount(const Patch* patch) = 0;

    //////////////////////////////////////////////////////////////////////
    /*! \brief Create the particles in the shell */
    //////////////////////////////////////////////////////////////////////
    virtual int createParticles(const Patch* patch,
				ParticleVariable<Point>&  pos,
				ParticleVariable<double>& vol,
				ParticleVariable<double>& pThickTop,
				ParticleVariable<double>& pThickBot,
				ParticleVariable<Vector>& pNormal,
				ParticleVariable<Vector>& psize,
				particleIndex start) = 0;

  protected:
    virtual void outputHelper( ProblemSpecP & ps ) const = 0;

    ShellGeometryPiece();

  };
} // End namespace Uintah

#endif // __SHELL_GEOMETRY_PIECE_H__
