#ifndef UINTAH_GRID_UnionBCData_H
#define UINTAH_GRID_UnionBCData_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

 /*!

  \class UnionBCData

  \ brief Stores the union of several different boundary condition geometries.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */
  using namespace SCIRun;

   class UnionBCData : public BCGeomBase {
   public:
     /// Constructor
     UnionBCData();

     /// Constructor with the actual boundary condition
     UnionBCData(BCData& bc);

     /// Copy constructor
     UnionBCData(const UnionBCData& bc);

     /// Assignment operator
     UnionBCData& operator=(const UnionBCData& bc);

     /// Constructor taking the problem specification
     UnionBCData(ProblemSpecP& ps);

     /// Destructor
     virtual ~UnionBCData();

     /// Clone the boundary condition -- allocates memory
     UnionBCData* clone();

     /// Get the boundary condition data
     void getBCData(BCData& bc) const;

     /// Add the boundary condition data -- no longer used.
     void addBCData(BCData& bc);

     /// Add the old boundary condition data -- no longer used.
     void addBC(BoundCondBase* bc);

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
					  vector<Point>& test_pts);

     /// Determine the face centered iterators.
     virtual void determineSFLimits(Patch::FaceType face, const Patch* patch);
         
   private:
     std::vector<BCGeomBase*> child;
     friend class BoundCondReader;
   };

} // End namespace Uintah

#endif




