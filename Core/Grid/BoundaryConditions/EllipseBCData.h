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

#ifndef UINTAH_GRID_EllipseBCData_H
#define UINTAH_GRID_EllipseBCData_H

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

namespace Uintah {

  /*!
    
  \class EllipseBCData
  
   Defines an ellipse geometry object for a boundary condition.
   The convention for creating ellipses from the input is as follows.
   The user provides four quantities: origin, minor radius, major radius, and angle.
   The origin defines the "center" of the ellipse or the intersection of the minor
   and major axes. The minor radius refers to the half length of the minor axis while
   the major radius denotes the half length of the major axis. Finally, the angle
   refers to the "tilt" of the ellipse measured counterclockwise with respect 
   to a reference axis, and looking from OUTSIDE the computational logical box. 
   We define the reference axis the major axis with angle zero. The convention 
   here is as follows. Consider an orthogonal coordinate system x_1, x_2, x_3 
   (e.g. x, y, z). An ellipse created on face x_i will have its reference axis 
   aligned with the x_{i+1} axis. The angle is then measured counterclockwise 
   from that reference axis, looking from OUTSIDE the computational box.
   NOTE that x_4 \equiv x_1. 
   
   So for example, an ellipse created on an "x-" face will have its reference axis
   aligned with the "y" axis. The angle will be measured counter-clockwise. An ellipse
   created on a "z+" face will have its reference axis aligned with the "x" axis.
  
  \author Tony Saad \n
  \date   February 9, 2012 \n
          Institute for Clean and Secure Energy \
  University of Utah \n
  */
  
  using namespace SCIRun;

  class EllipseBCData : public BCGeomBase  {

   public:
    /// Constructor
    EllipseBCData();

    /// Constructor used with a point defining the origin and the radius.
    EllipseBCData(Point& p, double minorRadius, double majorRadius, const std::string face, double d_angle=0.0);

    /// Destructor
    virtual ~EllipseBCData();

    virtual bool operator==(const BCGeomBase&) const;

    /// Clone the boundary condition geometry -- allocates memory.
    EllipseBCData* clone();

    /// Add the boundary condition data
    void addBCData(BCData& bc);

    /// Add the old boundary condition data -- no longer used.
    void addBC(BoundCondBase* bc);

    /// Get the boundary condition data
    void getBCData(BCData& bc) const;

    /// Determines if a point is inside the circle
    bool inside(const Point& p) const;

    /// Print out the boundary condition geometry type.
    virtual void print();

    /// Determine the cell and node centered iterators
    virtual void determineIteratorLimits(Patch::FaceType face,
                                         const Patch* patch, 
                                         vector<Point>& test_pts);
    
  private:
    BCData d_bc;
    Point  d_origin;
    double d_minorRadius;
    double d_majorRadius;
    double d_angleDegrees;
    const std::string d_face;
  };
  
} // End namespace Uintah

#endif




