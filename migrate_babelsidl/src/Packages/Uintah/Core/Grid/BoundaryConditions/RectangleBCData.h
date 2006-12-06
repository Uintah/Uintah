#ifndef UINTAH_GRID_RectangleBCData_H
#define UINTAH_GRID_RectangleBCData_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  /*!

  \class RectangleBCData
  
  \ brief Defines a rectangular geometry for a boundary condition.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */
  
  using namespace SCIRun;

  class RectangleBCData : public BCGeomBase  {
    
  public:
    /// Constructor
    RectangleBCData();

    /// Constructor with boundary condition data.
    RectangleBCData(BCData& bc);

    /// Constructor used with a point defining the lower and upper corners 
    /// of the rectangle.
    RectangleBCData(Point& low, Point& up);

    /// Destructor
    virtual ~RectangleBCData();

    /// Clone the boundary condition geometry -- allocates memory.
    RectangleBCData* clone();

    /// Add the boundary condition data
    void addBCData(BCData& bc);

    /// Add the old boundary condition data -- no longer used.
    void addBC(BoundCondBase* bc);

    /// Get the boundary condition data
    void getBCData(BCData& bc) const;

    /// Determines if a point is inside the rectangle.
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
    Point d_min,d_max;
  };
  
} // End namespace Uintah

#endif




