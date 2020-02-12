/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Box.h>
#include <vector>
#include <string>

#include <cmath>
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

    /// Make a clone
    virtual GeometryPieceP clone() const = 0;
         
    static const std::string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside the geometry.              */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p, const bool defVal) const = 0;
         
    //////////////////////////////////////////////////////////////////////
    /*! Returns the bounding box surrounding the particle domain.       */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const = 0;

    //////////////////////////////////////////////////////////////////////
    /*! Creates points and returns count of points                      */
    //////////////////////////////////////////////////////////////////////
    virtual unsigned int createPoints() = 0;

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle locations     */
    //////////////////////////////////////////////////////////////////////
    std::vector<Point>* getPoints();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle volumes       */
    //////////////////////////////////////////////////////////////////////
    std::vector<double>* getVolume();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle temperatures  */
    //////////////////////////////////////////////////////////////////////
    std::vector<double>* getTemperature();
    
    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle concentrations */
    //////////////////////////////////////////////////////////////////////
    std::vector<double>* getConcentration();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle + charges     */
    //////////////////////////////////////////////////////////////////////
    std::vector<double>* getPosCharge();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle - charges     */
    //////////////////////////////////////////////////////////////////////
    std::vector<double>* getNegCharge();
    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle permittivities*/
    //////////////////////////////////////////////////////////////////////
    std::vector<double>* getPermittivity();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle colors        */
    //////////////////////////////////////////////////////////////////////
    std::vector<double>* getColors();

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle forces        */
    //////////////////////////////////////////////////////////////////////
    std::vector<Vector>* getForces();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set particle fiber directions */
    //////////////////////////////////////////////////////////////////////
    std::vector<Vector>* getFiberDirs();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vectors containing the CPDI or CPTI R-vectors       */
    //////////////////////////////////////////////////////////////////////
    std::vector<Vector>* getRvec1();
    std::vector<Vector>* getRvec2();
    std::vector<Vector>* getRvec3();
                                                               // gcd adds
    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle velocity      */
    //////////////////////////////////////////////////////////////////////
    std::vector<Vector>* getVelocity();                     // gcd add end

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle size tensor   */
    //////////////////////////////////////////////////////////////////////
    std::vector<Matrix3>* getSize();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle area vector   */
    //////////////////////////////////////////////////////////////////////
    std::vector<Vector>* getArea();

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle locations     */
    //////////////////////////////////////////////////////////////////////
    void deletePoints();

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle volumes       */
    //////////////////////////////////////////////////////////////////////
    void deleteVolume();

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle sizes         */
    //////////////////////////////////////////////////////////////////////
    void deleteSizes();

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle areas         */
    //////////////////////////////////////////////////////////////////////
    void deleteAreas();

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle temperatures  */
    //////////////////////////////////////////////////////////////////////
    void deleteTemperature();

    //////////////////////////////////////////////////////////////////////
    /*! Deletes the vector containing the set of particle concentrations*/
    //////////////////////////////////////////////////////////////////////
    void deleteConcentration();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the number of particles                                 */
    //////////////////////////////////////////////////////////////////////
    int returnPointCount() const;

    //////////////////////////////////////////////////////////////////////
    /*! Set the particle spacing                                        */
    //////////////////////////////////////////////////////////////////////
    void setParticleSpacing(double dx);

    //////////////////////////////////////////////////////////////////////
    /*! Set the grid cell size                                          */
    //////////////////////////////////////////////////////////////////////
    void setCellSize(Vector DX);

  protected:

    //////////////////////////////////////////////////////////////////////
    /*! Writes the particle locations to a file that can be read by
        the FileGeometryPiece */
    //////////////////////////////////////////////////////////////////////
    void writePoints(const std::string& f_name, const std::string& var);

    std::vector<Point> d_points;
    std::vector<double> d_volume;// CPDI or CPTI
    std::vector<double> d_temperature;
    std::vector<double> d_concentration;
    std::vector<double> d_negcharge;
    std::vector<double> d_poscharge;
    std::vector<double> d_permittivity;
    std::vector<double> d_color;
    std::vector<Vector> d_forces;
    std::vector<Vector> d_fiberdirs;
    std::vector<Vector> d_velocity; // gcd adds
    std::vector<Vector> d_rvec1; // CPDI or CPTI
    std::vector<Vector> d_rvec2; // CPDI or CPTI
    std::vector<Vector> d_rvec3; // CPDI or CPTI
    std::vector<Matrix3> d_size; // CPDI or CPTI
    std::vector<Vector>  d_area; // CPDI or CPTI
    double d_dx;
    Vector d_DX;
  };
} // End namespace Uintah

#endif // __SMOOTH_PIECE_H__
