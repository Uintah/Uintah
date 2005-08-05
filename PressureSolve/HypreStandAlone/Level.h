#ifndef __LEVEL_H__
#define __LEVEL_H__

#include "Macros.h"
#include "Vector.h"
#include <vector>
using std::vector;

class Patch;

class Level {
  /*_____________________________________________________________________
    class Level:
    A union of boxes that share the same meshsize and index space. Each
    proc owns several boxes of a level, not necessarily the entire level.
    _____________________________________________________________________*/
public:
  Counter                  _numDims;    // # dimensions
  Vector<double>           _meshSize;   // Meshsize in all dimensions
  Vector<Counter>          _resolution; // Size(level) if extends over
                                        // the full domain
  vector< vector<Patch*> > _patchList;  // element i = patches owned by proc i
  Vector<Counter>          _refRat;     // Refinement ratio (h[coarser
                                        // lev]./h[this lev])

  Level(const Counter numDims,
        const double& h);

private:
}; 

#endif // __LEVEL_H__

