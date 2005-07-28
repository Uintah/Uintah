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
  vector<BoundaryType> _boundaries;
  vector<BoundaryCondition> _bc;
  
  Patch(const int procID, 
        const int levelID,
        const vector<int>& ilower,
        const vector<int>& iupper);

  BoundaryType getBoundary(int d, int s) 
    {
      return _boundaries[2*d + ((s+1)/2)];
    }

  void setBoundary(const int d, 
                   const int s,
                   const BoundaryType& bt ) 
    {
      //      printf("MYID = %d: size(_boundaries) = %d   d=%d  s=%d  ind=%d\n",MYID,_boundaries.size(),d,s,2*d+((s+1)/2));
      //      fflush(stdout);
      _boundaries[2*d+((s+1)/2)] = bt;
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

private:
};

#endif // __PATCH_H__
