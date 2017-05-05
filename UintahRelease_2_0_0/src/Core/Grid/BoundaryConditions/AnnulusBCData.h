/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef UINTAH_GRID_AnnulusBCData_H
#define UINTAH_GRID_AnnulusBCData_H

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

namespace Uintah {

  /*!
    
  \class CircleBCData
  
  \ brief Defines an annulus geometry for a boundary condition.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */
  

  class AnnulusBCData : public BCGeomBase  {

   public:
    /// Constructor
    AnnulusBCData();

    /// Constructor used with a point defining the origin and the radius.
    AnnulusBCData(Point& p, double inRadius, double outRadius);

    /// Destructor
    virtual ~AnnulusBCData();

    virtual bool operator==(const BCGeomBase&) const;

    /// Clone the boundary condition geometry -- allocates memory.
    AnnulusBCData* clone();

    /// Add the boundary condition data
    void addBCData(BCData& bc);

    /// Add the old boundary condition data -- no longer used.
    void addBC(BoundCondBase* bc);

    /// Add boundary condition within a scheduled task.
    void sudoAddBC(BoundCondBase* bc);

    /// Get the boundary condition data
    void getBCData(BCData& bc) const;

    /// Determines if a point is inside the circle
    bool inside(const Point& p) const;

    /// Print out the boundary condition geometry type.
    virtual void print();

    /// Determine the cell and node centered iterators
    virtual void determineIteratorLimits(Patch::FaceType face,
                                         const Patch* patch, 
                                         std::vector<Point>& test_pts);
    
  private:
    BCData d_bc;
    double d_innerRadius;
    double d_outerRadius;
  };
  
} // End namespace Uintah

#endif




