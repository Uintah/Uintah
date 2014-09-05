#ifndef UINTAH_GRID_BCDataArray_H
#define UINTAH_GRID_BCDataArray_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

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

  using std::vector;
  using std::map;

   class BCDataArray {
   public:

     /// Constructor
     BCDataArray();

     /// Destructor
     ~BCDataArray();

     /// Copy constructor
     BCDataArray(const BCDataArray& bc);

     /// Assignment operator
     BCDataArray& operator=(const BCDataArray& bc);

     /// Make a clone of self.  Must use delete to free.
     BCDataArray* clone();

     /// Get the boundary condition data for a given material and a given
     /// type for a given child.
     const BoundCondBase* getBoundCondData(int mat_id,string type, int i) const;

     /// Determine the iterator limits.
     void determineIteratorLimits(Patch::FaceType face, const Patch* patch);

     /// Add boundary condition data
     void addBCData(int mat_id,BCGeomBase* bc);

     /// Combine the duplicate BCGeometryTypes into a single BCGeometryType
     void combineBCGeometryTypes(int mat_id);

     /// Set the cell centered boundary iterators. 
     void setBoundaryIterator(int mat_id,vector<IntVector>& b, int i);

     /// Set the node centered boundary iterators.
     void setNBoundaryIterator(int mat_id,vector<IntVector>& b, int i);

     /// Set the face centered (X) boundary iterators.
     void setSFCXIterator(int mat_id,vector<IntVector>& i, int ii);

     /// Set the face centered (Y) boundary iterators.
     void setSFCYIterator(int mat_id,vector<IntVector>& i, int ii);
     
     /// Set the face centered (Z) boundary iterators.
     void setSFCZIterator(int mat_id,vector<IntVector>& i, int ii);

     /// Get the cell centered boundary iterator.
     void getBoundaryIterator(int mat_id,vector<IntVector>& b, int i) const;

     /// Get the node centered boundary iterator.
     void getNBoundaryIterator(int mat_id,vector<IntVector>& b, int i) const;

     /// Get the face centered (X) boundary iterator.
     void getSFCXIterator(int mat_id,vector<IntVector>& i, int ii) const;

     /// Get the face centered (Y) boundary iterator.
     void getSFCYIterator(int mat_id,vector<IntVector>& i, int ii) const;

     /// Get the face centered (Z) boundary iterator.
     void getSFCZIterator(int mat_id,vector<IntVector>& i, int ii) const;

     /// Return the number of children in the vector<BCGeomBase*>.
     int getNumberChildren(int mat_id) const;

     /// Get the ith child.
     BCGeomBase* getChild(int mat_id,int i) const;

     /// Print out the various boundary condition geometry types.
     void print();
     
     /// The map is for the mat_id.  -1 is for mat_id = "all", 0, for 
     /// mat_id = "0", etc.
     typedef map<int,vector<BCGeomBase*> > bcDataArrayType;         
   private:
     bcDataArrayType d_BCDataArray;
     friend class Patch;
     friend class BoundCondReader;
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




