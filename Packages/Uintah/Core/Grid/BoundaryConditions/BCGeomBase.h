#ifndef UINTAH_GRID_BCGeomBase_H
#define UINTAH_GRID_BCGeomBase_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCData.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <typeinfo>
#include <iterator>
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

    virtual bool operator==(const BCGeomBase&) const = 0;

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

    void setBoundaryIterator(std::vector<IntVector>::iterator b, 
                             std::vector<IntVector>::iterator e);

    /// Set the node centered boundary iterators
    void setNBoundaryIterator(std::vector<IntVector>& b);

    void setNBoundaryIterator(std::vector<IntVector>::iterator b, 
                              std::vector<IntVector>::iterator e);

#if 0
    /// Set the face centered iterators for the x component
    void setSFCXIterator(std::vector<IntVector>& i);


    /// Set the face centered iterators for the y component
    void setSFCYIterator(std::vector<IntVector>& i);

    /// Set the face centered iterators for the z component
    void setSFCZIterator(std::vector<IntVector>& i);
#endif

    /// Return the cell centered boundary iterators
    void getBoundaryIterator(std::vector<IntVector>*& b_ptr);

    void getBoundaryIterator(std::vector<IntVector>::iterator b_ptr, 
                             std::vector<IntVector>::iterator e_ptr);

    /// Return the node centered boundary iterators
    void getNBoundaryIterator(std::vector<IntVector>*& b_ptr);

    void getNBoundaryIterator(std::vector<IntVector>::iterator b_ptr,
                              std::vector<IntVector>::iterator e_ptr);

#if 0
    ///  Return the face centered iterators for the x component
    void getSFCXIterator(std::vector<IntVector>*& i_ptr);

    ///  Return the face centered iterators for the y component
    void getSFCYIterator(std::vector<IntVector>*& i_ptr);

    ///  Return the face centered iterators for the z component
    void getSFCZIterator(std::vector<IntVector>*& i_ptr);
#endif

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
    std::vector<IntVector> boundary,nboundary,sfcx,sfcy,sfcz;
    std::vector<IntVector>::iterator b_b,b_e,nb_b,nb_e;

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
