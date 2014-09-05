#ifndef UINTAH_GRID_BCDataArray_H
#define UINTAH_GRID_BCDataArray_H

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

#include <Core/Geometry/Vector.h>
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <vector>
#include <map>

namespace Uintah {

/*!

  \class BCDataArray
        
  \brief Supporting class for holding the basic boundary condition geometry
  classes such as side, circle, rectangle, union, and difference.   

  The boundary conditions may be applied to \b all materials.  In this case,
  we use the mat_id of -1.  Other materials are stored according to the mat_id
  specified such as 0, 1, 2, etc.  
        

  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

*/

class BCDataArray {
public:

  /// Constructor
  BCDataArray();

  /// Copy constructor
  BCDataArray( const BCDataArray & bc );

  /// Destructor
  ~BCDataArray();

  /// Get the boundary condition data for a given material and a given
  /// type for a given child.
  const BoundCondBase* getBoundCondData( int mat_id, const string & type, int ichild ) const;

  /// Determine the iterator limits.
  void determineIteratorLimits( Patch::FaceType face, const Patch* patch );

  // Remove the '-1' (all) material (when it is not needed).
  void removeGenericMaterial();

  /// Add boundary condition data
  void addBCGeomBase( BCGeomBase * bc );

  /// Combine the duplicate BCGeometryTypes into a single BCGeometryType
  //  void combineBCGeometryTypes( int mat_id );
  //  void combineBCGeometryTypes_NEW( int mat_id );

  /// Get the cell centered face iterator for the ith face child on mat_id.
  const Iterator & getCellFaceIterator( int mat_id, int ichild, const Patch * patch ) const;

  /// Get the node centered face iterator for the ith face child on mat_id.
  const Iterator & getNodeFaceIterator( int mat_id, int ichild, const Patch * patch ) const;

  /// Return the number of children in the vector<BCGeomBase*>.
  int getNumberChildren( int mat_id ) const;

  /// Get the ith child. (0 based)
  BCGeomBase* getChild( int mat_id, int ichild ) const;

  /// Print out the various boundary condition geometry types.
  void print() const;
     
private:

  typedef std::map< int, std::vector<BCGeomBase*> > bcDataArrayType;

  /// The d_BCDataArray maps from mat_id to BCGeomBase(s)*.  (-1 is for mat_id = "all", 0 for mat_id = "0", etc.)
  /// It should be noted that a BCGeomBase can actually hold (in its BCData) BoundCondBases that
  /// correspond to multiple materials... In this case, the same BCGeomBase may be pointed to multiple
  /// times in the d_BCDataArray (for each corresonding material)...
  ///
  bcDataArrayType d_BCDataArray;

  friend class Patch;
  friend class BoundCondReader;


  /// Assignment operator - DON'T USE THIS
  BCDataArray& operator=( const BCDataArray & bc );

};


// Used for inserting IntVectors into a set.  The standard < operator
// for IntVectors is too restrictive.

/// Sorts along the x axis
struct ltiv_x {
  bool operator()(const IntVector& i1,const IntVector& i2) {
    if (i2.x() < i1.x()) return false;
    if (i1.y() < i2.y()) return true;
    if (i1.z() < i2.z()) return true;

    return false;
  }
};

/// Sorts along the y axis
struct ltiv_y {
  bool operator()(const IntVector& i1,const IntVector& i2) {
    if (i2.y() < i1.y()) return false;
    if (i1.z() < i2.z()) return true;
    if (i1.x() < i2.x()) return true;

    return false;
  }
};

/// Sorts along the z axis
struct ltiv_z {
  bool operator()(const IntVector& i1,const IntVector& i2) {
    if (i2.z() < i1.z()) return false;
    if (i1.y() < i2.y()) return true;
    if (i1.x() < i2.x()) return true;

    return false;
  }
};
   
/// A less restrictive < operator rather than the built-in one for 
/// IntVectors.
struct ltiv_xyz {
  bool operator()(const IntVector& i1,const IntVector& i2) {
    if (i1.x() < i2.x() && i1.y() < i2.y() && i1.z() < i2.z())
      return true;
    else
      return false;
  }
};

} // End namespace Uintah

#endif

