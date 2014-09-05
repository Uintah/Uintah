#ifndef __SHELL_GEOMETRY_OBJECT_H__
#define __SHELL_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

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
    \endverbatim
	
  */
  /////////////////////////////////////////////////////////////////////////////

  class ShellGeometryPiece : public GeometryPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*! \brief Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~ShellGeometryPiece();
	 
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
    ShellGeometryPiece();
    ShellGeometryPiece(ShellGeometryPiece&);

  };
} // End namespace Uintah

#endif // __SHELL_GEOMETRY_PIECE_H__
