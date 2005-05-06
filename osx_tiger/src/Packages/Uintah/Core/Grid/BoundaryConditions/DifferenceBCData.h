#ifndef UINTAH_GRID_DifferenceBCData_H
#define UINTAH_GRID_DifferenceBCData_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

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

     /// Determine the face centered iterators.
     virtual void determineSFLimits(Patch::FaceType face, const Patch* patch);

   private:
     BCGeomBase* left;
     BCGeomBase* right;

     friend class BCReader;
   };

} // End namespace Uintah

#endif




