#ifndef UINTAH_GRID_AnnulusBCData_H
#define UINTAH_GRID_AnnulusBCData_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  /*!
    
  \class CircleBCData
  
  \ brief Defines an annulus geometry for a boundary condition.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */
  
  using namespace SCIRun;

  class AnnulusBCData : public BCGeomBase  {

   public:
    /// Constructor
    AnnulusBCData();

    /// Constructor with boundary condition data.
    AnnulusBCData(BCData& bc);

    /// Constructor used with a point defining the origin and the radius.
    AnnulusBCData(Point& p, double inRadius, double outRadius);

    /// Destructor
    virtual ~AnnulusBCData();

    /// Clone the boundary condition geometry -- allocates memory.
    AnnulusBCData* clone();

    /// Add the boundary condition data
    void addBCData(BCData& bc);

    /// Add the old boundary condition data -- no longer used.
    void addBC(BoundCondBase* bc);

    /// Get the boundary condition data
    void getBCData(BCData& bc) const;

    /// Determines if a point is inside the circle
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
    double d_innerRadius;
    double d_outerRadius;
    Point  d_origin;
  };
  
} // End namespace Uintah

#endif




