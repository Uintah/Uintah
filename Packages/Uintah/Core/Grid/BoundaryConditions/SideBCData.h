#ifndef UINTAH_GRID_SideBCData_H
#define UINTAH_GRID_SideBCData_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/Variables/GridIterator.h>
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

  using namespace SCIRun;
  using std::vector;
 
  class SideBCData : public BCGeomBase {

  public:
    /// Constructor
    SideBCData();


    /// Assignment Operator
    SideBCData& operator=(const SideBCData& bc);

    /// Destructor
    virtual ~SideBCData();

    virtual bool operator==(const BCGeomBase&) const;

    /// Clone the boundary condition geometry -- allocates memory.
    SideBCData* clone();

    /// Get the boundary condition data
    void getBCData(BCData& bc) const;

    /// Add the boundary condition data
    void addBCData(BCData& bc);

    /// Add the old boundary condition data -- no longer used.
    void addBC(BoundCondBase* bc);

    /// Determines if a point is inside -- always returns true.
    bool inside(const Point& p) const;

    /// Print out the boundary condition geometry type.
    virtual void print();

    /// Determine the cell and node centered iterators
    virtual void determineIteratorLimits(Patch::FaceType face,
					 const Patch* patch, 
					 vector<Point>& test_pts);

    
    
  private:
    BCData d_bc;

  };
  
} // End namespace Uintah

#endif




