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

#ifndef UINTAH_GRID_UnionBCData_H
#define UINTAH_GRID_UnionBCData_H

#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>

namespace Uintah {

 /*!

  \class UnionBCData

  \ brief Stores the union of several different boundary condition geometries.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */

   class UnionBCData : public BCGeomBase {
   public:
     /// Constructor
     UnionBCData();

     /// Copy constructor
     UnionBCData(const UnionBCData& bc);

     /// Assignment operator
     UnionBCData& operator=(const UnionBCData& bc);

     /// Constructor taking the problem specification
     UnionBCData(ProblemSpecP& ps);

     /// Destructor
     virtual ~UnionBCData();

     virtual bool operator==(const BCGeomBase&) const;

     /// Clone the boundary condition -- allocates memory
     UnionBCData* clone();

     /// Get the boundary condition data
     void getBCData(BCData& bc) const;

     /// Add the boundary condition data -- no longer used.
     void addBCData(BCData& bc);

     /// Add the old boundary condition data -- no longer used.
     void addBC(BoundCondBase* bc);

     void sudoAddBC(BoundCondBase* bc);

     /// Add the boundary condition geometry
     void addBCData(BCGeomBase* bc);

     /// Determines if a point is inside the collection of boundary condition
     /// geometries.
     bool inside(const Point& p) const;
     
     /// Print out the boundary condition geometry types.
     virtual void print();
     
     /// Determine the cell and node boundary iterators.
     virtual void determineIteratorLimits(Patch::FaceType face,
                                          const Patch* patch, 
                                          std::vector<Point>& test_pts);
   private:
     std::vector<BCGeomBase*> child;
     friend class BoundCondReader;
   };

} // End namespace Uintah

#endif




