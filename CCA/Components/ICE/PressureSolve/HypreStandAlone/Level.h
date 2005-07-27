#ifndef __LEVEL_H__
#define __LEVEL_H__

#include <vector>
#include "mydriver.h"

class Patch;

class Level {
  /*_____________________________________________________________________
    class Level:
    A union of boxes that share the same meshsize and index space. Each
    proc owns several boxes of a level, not necessarily the entire level.
    _____________________________________________________________________*/
public:
  Counter              _numDims;    // # dimensions
  std::vector<double>  _meshSize;   // Meshsize in all dimensions
  std::vector<Counter> _resolution; // Size(level) if extends over the full domain
  std::vector<Patch*>  _patchList;  // owned by this proc ONLY
  std::vector<Counter> _refRat;     // Refinement ratio (h[coarser lev]./h[this lev])

  Level(const Counter numDims,
        const double& h);
private:
}; 

#endif // __LEVEL_H__

