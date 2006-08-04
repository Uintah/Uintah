#ifndef UINTAH_GRID_BCGeomBase_H
#define UINTAH_GRID_BCGeomBase_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCData.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <typeinfo>
#include <sgi_stl_warnings_on.h>

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

    /// Make a clone
    virtual BCGeomBase* clone() = 0;

    /// Get the boundary condition data
    virtual void getBCData(BCData& bc) const = 0;

    /// For old boundary conditions
    virtual void addBCData(BCData& bc)  = 0;

    /// For old boundary conditions
    virtual void addBC(BoundCondBase* bc)  = 0;

    /// Set the cell centered boundary iterators
    void setBoundaryIterator(std::vector<IntVector>& b);

    /// Set the node centered boundary iterators
    void setNBoundaryIterator(std::vector<IntVector>& b);

    /// Set the face centered iterators for the x component
    void setSFCXIterator(std::vector<IntVector>& i);


    /// Set the face centered iterators for the y component
    void setSFCYIterator(std::vector<IntVector>& i);

    /// Set the face centered iterators for the z component
    void setSFCZIterator(std::vector<IntVector>& i);

    /// Return the cell centered boundary iterators
    void getBoundaryIterator(std::vector<IntVector>& b) const;

    /// Return the node centered boundary iterators
    void getNBoundaryIterator(std::vector<IntVector>& b) const;

    ///  Return the face centered iterators for the x component
    void getSFCXIterator(std::vector<IntVector>& i) const;

    ///  Return the face centered iterators for the y component
    void getSFCYIterator(std::vector<IntVector>& i) const;

    ///  Return the face centered iterators for the z component
    void getSFCZIterator(std::vector<IntVector>& i) const;

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

    ///  Determine the face centered iterators (x,y,z).
    virtual void determineSFLimits(Patch::FaceType face, 
				   const Patch* patch);
    
    /// Print out the iterators for the boundary.
    void printLimits() const;

  protected:
    std::vector<IntVector> boundary,nboundary,sfcx,sfcy,sfcz;
  };

  template<class T> class cmp_type {
    public:
    bool operator()(BCGeomBase* p) {
      return (typeid(T) == typeid(*p));
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
