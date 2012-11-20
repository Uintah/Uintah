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

#ifndef UINTAH_GRID_BCGeomBase_H
#define UINTAH_GRID_BCGeomBase_H

#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/BaseIterator.h>

#include <iostream>
#include <iterator>
#include <typeinfo>
#include <set>
#include <vector>

namespace Uintah {

  // Forward declarations:
  class UnionBCData; 
  class DifferenceBCData;

  /*!

  \class BCGeomBase

  \ brief Base class for the boundary condition geometry types.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */

  class BCGeomBase {
  public:

    /// Destructor
    virtual ~BCGeomBase();

    /// Equality test
    virtual bool operator==( const BCGeomBase& ) const = 0;

    void addBC( const BoundCondBase* bc );

    /// Get the boundary condition data for the given 'material'
    const BCData * getBCData( int material ) const;

    const Iterator & getCellFaceIterator( const Patch * patch ) const;

    const Iterator & getNodeFaceIterator( const Patch * patch ) const;

    /// Determine if a point is inside the geometry where the boundary
    /// condition is applied.
    virtual bool inside( const Point& p ) const = 0;

    /// Print out info about the boundary condition (Mostly for debugging.)
    virtual void print( int depth = 0 ) const = 0;

    /// Determine the cell centered boundary and node centered boundary
    /// iterators.
    virtual void determineIteratorLimits( const Patch::FaceType      face,
                                          const Patch              * patch,
                                          const std::vector<Point> & test_pts );
    
    /// Print out the iterators for the boundary.
    void printLimits() const;

    const string &          getName() const { return d_name; }

    const Patch::FaceType & getSide() const { return d_faceSide; }

    // Returns a list of all the materials that the BCGeom corresponds to
    virtual std::set<int>   getMaterials() const;

  protected:

    std::map<int,BCData*>      d_bcs;   // Indexed by material id: the int. (-1 is for all materials)
    std::map<int,Iterator*>    d_cells; // Indexed by patch id (the int).
    std::map<int,Iterator*>    d_nodes; // Indexed by patch id (the int).
    string                     d_name;
    Patch::FaceType            d_faceSide;

    std::map<int,const Patch*> d_iteratorLimitsDetermined; // Indexed by patch id (the int).  (If the patch* is non-NULL, limits have been determined.)

    /// These constructors are protected as they shouldn't be used directly...
    //
    BCGeomBase( const string & name, const Patch::FaceType & side );
    BCGeomBase();
    BCGeomBase( const BCGeomBase& rhs ); // Copy constructor

    /// Assignment operator - This should not be used. 
    BCGeomBase& operator=( const BCGeomBase& rhs );

  private:
    // Helper function for constructors.
    void init( const string & name = "NotSet", const Patch::FaceType & side = Patch::invalidFace );

    friend class UnionBCData;
    friend class DifferenceBCData;

  };

} // End namespace Uintah

#endif
