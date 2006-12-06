#ifndef __PATCH_H__
#define __PATCH_H__

#include "Box.h"
#include <map>

/* Forward declarations */
class Level;

class Patch {
  /*_____________________________________________________________________
    class Patch:
    A box of data at a certain level. A processor may own more than one
    patch. A patch can be owned by one proc only.
    _____________________________________________________________________*/

public:

  enum BoundaryType {
    Domain = 0, 
    CoarseFine, 
    Neighbor
  };
  
  enum BoundaryCondition {
    NA = -1,  // Not applicable
    Dirichlet = 0,
    Neumann
  };
  
  Patch(const int procID, 
        const Counter levelID,
        const Box& box);
  Patch(const Patch& other);
  Patch& operator = (const Patch& other);

  /* Boundary conditions get & set */
  BoundaryType getBoundaryType(const Counter d,
                               const Side s) const;
  BoundaryCondition getBC(const Counter d,
                          const Side s) const;
  void setAllBoundaries(const Vector<BoundaryType>& boundaries);
  void setBoundaryType(const Counter d,
                       const Side s,
                       const BoundaryType& bt);
  void setAllBC(const Vector<BoundaryCondition>& bc);
  void setBC(const Counter d,
             const Side s,
             const BoundaryCondition& bc);
  void setDomainBoundaries(const Level& lev);

  int         _procID;    // Owning processor's ID
  Counter     _levelID;   // Which level this Patch belongs to
  Box         _box;       // Patch box extents
  Counter     _numCells;  // Total # cells
  Counter     _patchID;   // Patch ID, unique across all levels & procs
 protected:
  Vector<BoundaryType>      _boundaries;
  Vector<BoundaryCondition> _bc;
  
 private:
};

std::ostream&
operator << (std::ostream& os, const Patch& patch);
std::ostream&
operator << (std::ostream& os, const Patch::BoundaryCondition& c);
std::ostream&
operator << (std::ostream& os, const Patch::BoundaryType& b);

#endif // __PATCH_H__
