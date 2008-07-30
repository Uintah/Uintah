#ifndef UINTAH_GRID_BCGeomBase_H
#define UINTAH_GRID_BCGeomBase_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCData.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Variables/Iterator.h>
#include <Packages/Uintah/Core/Grid/Variables/BaseIterator.h>
#include <vector>
#include <typeinfo>
#include <iterator>

namespace Uintah {

  /*!

  \class BCGeomBase

  \ brief Base class for the boundary condition geometry types.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

  */

  using SCIRun::IntVector;
  using SCIRun::Point;
  using std::vector;
   

  class BCGeomBase {
  public:

    /// Constructor
    BCGeomBase();

    /// Destructor
    virtual ~BCGeomBase();    

    /// Equality test
    virtual bool operator==(const BCGeomBase&) const = 0;

    /// Make a clone
    virtual BCGeomBase* clone() = 0;

    /// Get the boundary condition data
    virtual void getBCData(BCData& bc) const = 0;

    /// For old boundary conditions
    virtual void addBCData(BCData& bc)  = 0;

    /// For old boundary conditions
    virtual void addBC(BoundCondBase* bc)  = 0;

    void getCellFaceIterator(Iterator& b_ptr);

    void getNodeFaceIterator(Iterator& b_ptr);


    /// Determine if a point is inside the geometry where the boundary
    /// condition is applied.
    virtual bool inside(const Point& p) const = 0;

    /// Print out the type of boundary condition -- debugging
    virtual void print() = 0;

    /// Determine the cell centered boundary and node centered boundary
    /// iterators.
    virtual void determineIteratorLimits(Patch::FaceType face, 
					 const Patch* patch, 
					 vector<Point>& test_pts);
    
    /// Print out the iterators for the boundary.
    void printLimits() const;

  protected:
    Iterator d_cells;
    Iterator d_nodes;

  };

  template<class T> class cmp_type {
    public:
    bool operator()(const BCGeomBase* p) {
      return (typeid(T) == typeid(*p));
    }
  };

  template<class T> class not_type {
    public:
    bool operator()(const BCGeomBase* p) {
      return (typeid(T) != typeid(*p));
    }
  };

  template<typename T> class delete_object {
  public:
    void operator() (T* ptr) {
      delete ptr;
    }
  };

} // End namespace Uintah

#endif
