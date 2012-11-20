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

#ifndef UINTAH_GRID_SideBCData_H
#define UINTAH_GRID_SideBCData_H

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <vector>

namespace Uintah {

  /*!

  \class SideBCData
  
  \ brief Defines a boundary condition geometry for the entire side of the
  domain.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */

  class SideBCData : public BCGeomBase {

  public:
    /// Constructor
    SideBCData( const string & name, const Patch::FaceType & side );

    /// Assignment Operator
    SideBCData& operator=( const SideBCData & bc );

    /// Destructor
    virtual ~SideBCData();

    virtual bool operator==( const BCGeomBase & rhs ) const;

    /// Determines if a point is inside -- always returns true.
    bool inside(const Point& p) const;

    /// Print out the boundary condition geometry type. (Only if debug stream turned on!)
    virtual void print( int depth = 0 ) const;

    /// Determine the cell and node centered iterators
    virtual void determineIteratorLimits(       Patch::FaceType   face,
                                          const Patch           * patch, 
                                                vector<Point>   & test_pts );

  private:
    SideBCData() {} // Default constructor should not be used...
  };
  
} // End namespace Uintah

#endif
