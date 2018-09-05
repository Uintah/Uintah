/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef UINTAH_GRID_UnstructuredBCDataArray_H
#define UINTAH_GRID_UnstructuredBCDataArray_H

#include <Core/Geometry/Vector.h>
#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/BoundaryConditions/UnstructuredBCGeomBase.h>
#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Grid/Variables/Iterator.h>

#include <map>
#include <vector>

namespace Uintah {

/*!

  \class UnstructuredBCDataArray
        
  \brief Supporting class for holding the basic boundary condition geometry
  classes such as side, circle, rectangle, union, and difference.   

  The boundary conditions may be applied to \b all materials.  In this case,
  we use the mat_id of -1.  Other materials are stored according to the mat_id
  specified such as 0, 1, 2, etc.  
  
 The UnstructuredBCDataArray typically represents a domain's Face and stores the list of boundary conditions
 applied on that face.

  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

*/

   class UnstructuredBCDataArray {
   public:

     /// Constructor
     UnstructuredBCDataArray();

     /// Destructor
     ~UnstructuredBCDataArray();

     /// Copy constructor
     UnstructuredBCDataArray(const UnstructuredBCDataArray& bc);

     /// Assignment operator
     UnstructuredBCDataArray& operator=(const UnstructuredBCDataArray& bc);

     /// Make a clone of self.  Must use delete to free.
     UnstructuredBCDataArray* clone();

     /// Get the boundary condition data for a given material and a given
     /// type for a given child.
     const BoundCondBase* getBoundCondData(       int           mat_id,
                                            const std::string & type,
                                                  int           ichild ) const;

     bool checkForBoundCondData(       int         & mat_id,
                                 const std::string & type,
                                       int           ichild ) ;

     /// Determine the iterator limits for domain boundaries.
     void determineIteratorLimits(UnstructuredPatch::FaceType face, const UnstructuredPatch* patch);
     
     /*
      \author Tony Saad
      \date   September 2014
      \brief  Determine the iterator points associated with interior boundaries.
      */
     void determineInteriorBndIteratorLimits(UnstructuredPatch::FaceType face, const UnstructuredPatch* patch);

     /// Add boundary condition data
     void addBCData(int mat_id,UnstructuredBCGeomBase* bc);

     /// Combine the duplicate BCGeometryTypes into a single BCGeometryType
     void combineBCGeometryTypes(int mat_id);
     void combineBCGeometryTypes_NEW(int mat_id);

     /// Get the cell centered face iterator for the ith face child on mat_id.
     void getCellFaceIterator(int mat_id,Iterator& b_ptr, int ichild) const;

     /// Get the node centered face iterator for the ith face child on mat_id.
     void getNodeFaceIterator(int mat_id,Iterator& b_ptr, int ichild) const;

     /// Return the number of children in the vector<UnstructuredBCGeomBase*>.
     int getNumberChildren(int mat_id) const;

     /// Get the ith child.
     UnstructuredBCGeomBase* getChild(int mat_id,int ichild) const;

     /// Print out the various boundary condition geometry types.
     void print() const;
     
     std::vector <UnstructuredBCGeomBase*>   getBCGeom( int matl_index){
       return d_UnstructuredBCDataArray[matl_index];
     }

     /// The map is for the mat_id.  -1 is for mat_id = "all", 0, for 
     /// mat_id = "0", etc.
     typedef std::map<int,std::vector<UnstructuredBCGeomBase*> > bcDataArrayType;

   private:
     friend class UnstructuredPatch;
     friend class UnstructuredBoundCondReader;

     bcDataArrayType d_UnstructuredBCDataArray;
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




