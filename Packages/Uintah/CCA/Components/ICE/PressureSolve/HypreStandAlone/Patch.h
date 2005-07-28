#ifndef __PATCH_H__
#define __PATCH_H__

#include "mydriver.h"
#include <vector>
#include <string>
#include <map>

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
  
  int         _procID;    // Owning processor's ID
  int         _levelID;   // Which level this Patch belongs to
  vector<int> _ilower;    // Lower left corner subscript
  vector<int> _iupper;    // Upper right corner subscript
  int         _numCells;  // Total # cells

  Patch(const int procID, 
        const int levelID,
        const vector<int>& ilower,
        const vector<int>& iupper);

  BoundaryType getBoundaryType(const Counter d,
                           const Side s) const
    {
      return _boundaries[2*d + ((s+1)/2)];
    }

  BoundaryCondition getBC(const Counter d,
                          const Side s) const
    {
      return _bc[2*d + ((s+1)/2)];
    }

  void setAllBoundaries(const vector<BoundaryType>& boundaries)
    {
      _boundaries = boundaries;
    }

  void setBoundaryType(const Counter d,
                   const Side s,
                   const BoundaryType& bt) 
    {
      _boundaries[2*d+((s+1)/2)] = bt;
    }

  void setAllBC(const vector<BoundaryCondition>& bc)
    {
      _bc = bc;
    }

  void setBC(const Counter d,
             const Side s,
             const BoundaryCondition& bc) 
    {
      //      fprintf(stderr,"size(_bc) = %d, d = %d, s = %d, accessing %d\n",
      //            _bc.size(),d,s,2*d+((s+1)/2));
      _bc[2*d+((s+1)/2)] = bc;
    }

  /* Static members & functions */
  static bool initialized /* = false */;
  static std::map<BoundaryType, std::string> boundaryTypeString; 
  
  static void init(void)
    {
      boundaryTypeString[Domain    ] = "Domain";
      boundaryTypeString[CoarseFine] = "CoarseFine";
      boundaryTypeString[Neighbor  ] = "Neighbor";
    }
 protected:
  vector<BoundaryType> _boundaries;
  vector<BoundaryCondition> _bc;
  
 private:
};

#endif // __PATCH_H__
