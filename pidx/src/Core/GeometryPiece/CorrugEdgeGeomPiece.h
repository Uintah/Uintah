/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef __CORRUG_EDGE_PIECE_H__
#define __CORRUG_EDGE_PIECE_H__

#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Geometry/Point.h>

#include <cmath>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/////////////////////////////////////////////////////////////////////////////
/*!
	
 \class CorrugEdgeGeomPiece
	
 \brief Creates a plate with one edge corrugated. 
	
 \warning Does not allow for correct application of symmetry 
          boundary conditions.  Use symmetry at your own risk.
          Normal is not used .. needs to be implemented.
          Volume is not calculated accurately.

 \author  Biswajit Banerjee \n
          C-SAFE and Department of Mechanical Engineering \n
          University of Utah \n

   Creates a plate with one corrugated edge.  The particle spacing
   is determined from the grid size and the number of particles
   per grid cell.\n
   The input form for a solid cylinder looks like this: \n
   \verbatim
   <corrugated> 
     <xymin>      [0.0,0.0,0.0]    </xymin> 
     <xymax>      [20.0,20.0,0.0]  </xymax> 
     <thickness>  1.0              </thickness>
     <normal>     [0.0,0.0,1.0]    </normal>
     <corr_edge>  x+               </corr_edge>
     <curve>      sin              </curve>
     <wavelength> 2.0              </wavelength>
     <amplitude>  2.0              </amplitude>
   </corrugated> 
   \endverbatim
*/
/////////////////////////////////////////////////////////////////////////////

  class CorrugEdgeGeomPiece : public SmoothGeomPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*!  
      \brief Constructor that takes a ProblemSpecP argument.   
      It reads the xml input specification and builds a plate with
      one corrugated edge.
    */
    //////////////////////////////////////////////////////////////////////
    CorrugEdgeGeomPiece(ProblemSpecP &);
	 
    //////////////////////////////////////////////////////////////////////
    /*! Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~CorrugEdgeGeomPiece();

    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const;
	 
    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside the cylinder. */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const;
	 
    //////////////////////////////////////////////////////////////////////
    /*! Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const;

    //////////////////////////////////////////////////////////////////////
    /*! Creates the particles */
    //////////////////////////////////////////////////////////////////////
    unsigned int createPoints();

  private:
    virtual void outputHelper( ProblemSpecP & ps) const;
	 
    Point  d_xymin;
    Point  d_xymax;
    double d_thickness;
    Vector d_normal;
    string d_edge;
    string d_curve;
    double d_wavelength;
    double d_amplitude;
  };
} // End namespace Uintah

#endif // __CORRUG_EDGE_PIECE_H__
