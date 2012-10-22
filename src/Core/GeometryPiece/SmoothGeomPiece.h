/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __SMOOTH_PIECE_H__
#define __SMOOTH_PIECE_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <vector>
#include <string>

#include <cmath>
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

    /// Make a clone
    virtual GeometryPieceP clone() const = 0;
	 
    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside the cylinder. */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const = 0;
	 
    //////////////////////////////////////////////////////////////////////
    /*! Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const = 0;

    //////////////////////////////////////////////////////////////////////
    /*! Creates points and returns count of points */
    //////////////////////////////////////////////////////////////////////
    virtual unsigned int createPoints() = 0;

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle locations */
    //////////////////////////////////////////////////////////////////////
    vector<Point>* getPoints();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle volumes */
    //////////////////////////////////////////////////////////////////////
    vector<double>* getVolume();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle temperatures */
    //////////////////////////////////////////////////////////////////////
    vector<double>* getTemperature();
    
    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle colors */
    //////////////////////////////////////////////////////////////////////
    vector<double>* getColors();

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle forces */
    //////////////////////////////////////////////////////////////////////
    vector<Vector>* getForces();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle fiber directions */
    //////////////////////////////////////////////////////////////////////
    vector<Vector>* getFiberDirs();

    /////////////////////////////////////////////////////////////  // gcd adds
    /*! Returns the vector containing the set of particle velocity */
    //////////////////////////////////////////////////////////////////////
    vector<Vector>* getVelocity();                          // gcd add end

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle locations */
    //////////////////////////////////////////////////////////////////////
    void deletePoints();

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle volumes */
    //////////////////////////////////////////////////////////////////////
    void deleteVolume();


    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle temperatures */
    //////////////////////////////////////////////////////////////////////
    void deleteTemperature();
    //////////////////////////////////////////////////////////////////////
    /*! Returns the number of particles */
    //////////////////////////////////////////////////////////////////////
    int returnPointCount() const;

    //////////////////////////////////////////////////////////////////////
    /*! Set the particle spacing */
    //////////////////////////////////////////////////////////////////////
    void setParticleSpacing(double dx);

  protected:

    //////////////////////////////////////////////////////////////////////
    /*! Writes the particle locations to a file that can be read by
        the FileGeometryPiece */
    //////////////////////////////////////////////////////////////////////
    void writePoints(const string& f_name, const string& var);

    vector<Point> d_points;
    vector<double> d_volume;
    vector<double> d_temperature;
    vector<double> d_color;
    vector<Vector> d_forces;
    vector<Vector> d_fiberdirs;
    vector<Vector> d_velocity;    // gcd adds
    double d_dx;
  };
} // End namespace Uintah

#endif // __SMOOTH_PIECE_H__
