/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef __CYLINDER_SHELL_PIECE_H__
#define __CYLINDER_SHELL_PIECE_H__

#include <Core/GeometryPiece/ShellGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <cmath>
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

    static const string TYPE_NAME;    
    std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const;
	 
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
			ParticleVariable<Matrix3>& psize,
			particleIndex start);

  private:
	 
    virtual void outputHelper( ProblemSpecP & ps ) const;

    Point  d_bottom;
    Point  d_top;
    double d_radius;
    double d_thickness;
    double d_numAxis;
    double d_numCircum;
  };
} // End namespace Uintah

#endif // __CYLINDER_SHELL_PIECE_H__
