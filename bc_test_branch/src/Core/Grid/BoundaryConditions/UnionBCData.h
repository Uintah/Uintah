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

  /// Copy constructor - Union of p1 and p2.
  UnionBCData( std::vector<BCGeomBase*> & children, const string & name, const Patch::FaceType & side );

  /// Destructor
  virtual ~UnionBCData();

  virtual bool operator==(const BCGeomBase&) const;

  /// Determines if a point is inside the collection of boundary condition
  /// geometries.
  bool inside( const Point & p ) const;
     
  /// Print out the boundary condition geometry types.
  virtual void print( int depth = 0 ) const;

  /// Determine the cell and node boundary iterators.
  virtual void determineIteratorLimits( const Patch::FaceType   face,
                                        const Patch           * patch, 
                                        const vector<Point>   & test_pts );

  // Returns a list of all the materials that the BCGeom corresponds to
  virtual std::set<int> getMaterials() const;

private:

  std::vector<BCGeomBase*> d_children;

  /// Constructors -- Made private in order to disallow non-initialized Union objects...
  UnionBCData() {}
  UnionBCData( const UnionBCData & bc );


  /// Assignment operator - don't use this.
  UnionBCData& operator=( const UnionBCData & bc ) { return *this; }

};

} // End namespace Uintah

#endif

