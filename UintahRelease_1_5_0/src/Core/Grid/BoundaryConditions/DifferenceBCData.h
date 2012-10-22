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

#ifndef UINTAH_GRID_DifferenceBCData_H
#define UINTAH_GRID_DifferenceBCData_H

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

namespace Uintah {
  /*!
    
  \class DifferenceBCData
  
  \ brief Stores the difference of two different boundary condition geometries.
   
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */
  
  using namespace SCIRun;

  class DifferenceBCData : public BCGeomBase  {
   public:

    /// Default constructor
    DifferenceBCData();

    /// Copy constructor
     DifferenceBCData(const DifferenceBCData& rhs);

     /// Assignment operator
     DifferenceBCData& operator=(const DifferenceBCData& bc);


     /// Constructor with two boundary condition geometries.  The second 
     /// argument is subtracted from the first argument.
     DifferenceBCData(BCGeomBase* p1,BCGeomBase* p2);

     /// Destructor
     virtual ~DifferenceBCData();

     virtual bool operator==(const BCGeomBase&) const;

     /// Clone the boundary condition geometry -- allocates memory.
     DifferenceBCData* clone();

     /// Get the boundary condition data
     void getBCData(BCData& bc) const;

     /// Add the boundary condition data -- no longer used.
     void addBCData(BCData& bc);

     /// Add the old boundary condition data -- no longer used.
     void addBC(BoundCondBase* bc);

     /// Determine if a point is inside the object.
     bool inside(const Point& p) const;

     /// Print out the boundary condition geometry types.
     virtual void print();


     /// Determine the cell and node boundary iterators.
     virtual void determineIteratorLimits(Patch::FaceType face,
                                          const Patch* patch, 
                                          vector<Point>& test_pts);


   private:
     BCGeomBase* left;
     BCGeomBase* right;

     friend class BCReader;
   };

} // End namespace Uintah

#endif




