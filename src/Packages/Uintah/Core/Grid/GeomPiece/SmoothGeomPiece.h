#ifndef __SMOOTH_PIECE_H__
#define __SMOOTH_PIECE_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <math.h>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

  using std::vector;
  using std::string;

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
    /*! Returns the vector containing the set of particle locations */
    //////////////////////////////////////////////////////////////////////
    vector<Point>* getPoints();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle volumes */
    //////////////////////////////////////////////////////////////////////
    vector<double>* getVolume();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the number of particles */
    //////////////////////////////////////////////////////////////////////
    int returnPointCount() const;

  protected:

    //////////////////////////////////////////////////////////////////////
    /*! Writes the particle locations to a file that can be read by
        the FileGeometryPiece */
    //////////////////////////////////////////////////////////////////////
    void writePoints(const string& f_name, const string& var);

    vector<Point> d_points;
    vector<double> d_volume;
  };
} // End namespace Uintah

#endif // __SMOOTH_PIECE_H__
