#ifndef UINTAH_GRID_SideBCData_H
#define UINTAH_GRID_SideBCData_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

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

    /// Constructor with boundary condition data.
    SideBCData(BCData& d_bc);

    /// Destructor
    virtual ~SideBCData();

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

    /// Determine the face centered iterators
    virtual void determineSFLimits(Patch::FaceType face, const Patch* patch);
    
  private:
    BCData d_bc;
  };
  
} // End namespace Uintah

#endif




