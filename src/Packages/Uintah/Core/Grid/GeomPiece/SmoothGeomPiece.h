#ifndef __SMOOTH_PIECE_H__
#define __SMOOTH_PIECE_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
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
	
  \class SmoothGeomPiece
	
  \brief Abstract base class for smooth geometry pieces
	
  \warning Does not allow for correct application of symmetry 
  boundary conditions.  Use symmetry at your own risk.
  The end caps are exactly the same diameter as the outer
  diameter of the cylinder and are welded perfectly to the 
  cylinder.

  \author  Biswajit Banerjee \n
  C-SAFE and Department of Mechanical Engineering \n
  University of Utah \n
  */
  /////////////////////////////////////////////////////////////////////////////

  class SmoothGeomPiece : public GeometryPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*! Constructor */
    //////////////////////////////////////////////////////////////////////
    SmoothGeomPiece();
	 
    //////////////////////////////////////////////////////////////////////
    /*! Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~SmoothGeomPiece();
	 
    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside the cylinder. */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const = 0;
	 
    //////////////////////////////////////////////////////////////////////
    /*! Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const = 0;

    //////////////////////////////////////////////////////////////////////
    /*! Returns the number of particles */
    //////////////////////////////////////////////////////////////////////
    virtual int returnParticleCount(const Patch* patch) = 0;

    //////////////////////////////////////////////////////////////////////
    /*! Creates the particles */
    //////////////////////////////////////////////////////////////////////
    virtual int createParticles(const Patch* patch,
				ParticleVariable<Point>&  pos,
				ParticleVariable<double>& vol,
				ParticleVariable<Vector>& psize,
				particleIndex start) = 0;

  };
} // End namespace Uintah

#endif // __SMOOTH_PIECE_H__
