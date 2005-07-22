#ifndef __PATCH_H__
#define __PATCH_H__

#include "mydriver.h"
#include <string>

class Patch {
  /*_____________________________________________________________________
    class Patch:
    A box of data at a certain level. A processor may own more than one
    patch. A patch can be owned by one proc only.
    _____________________________________________________________________*/
public:
  enum BoundaryType {
    Domain, CoarseFine, Neighbor
  };
  
  int         _procID;    // Owning processor's ID
  int         _levelID;   // Which level this Patch belongs to
  vector<int> _ilower;    // Lower left corner subscript
  vector<int> _iupper;    // Upper right corner subscript
  int         _numCells;  // Total # cells
  vector<BoundaryType> _boundaries;
  
  Patch(const int procID, 
        const int levelID,
        const vector<int>& ilower,
        const vector<int>& iupper);

  BoundaryType& getBoundary(int d, int s) 
    {
      return _boundaries[2*d + ((s+1)/2)];
    }

  void setBoundary(const int d, 
                   const int s,
                   const BoundaryType& bt ) 
    {
      _boundaries[2*d+((s+1)/2)] = bt;
    }

  String boundaryTypeString(const BoundaryType& bt)
    {
      switch (bt)
        {
        case Domain:
          return "Domain";
          break;
        case CoarseFine:
          return "CoarseFine";
          break;
        case Neighbor:
          return "Neighbor";
          break;
        }
      return "???";
    }

private:
};

#endif // __PATCH_H__

