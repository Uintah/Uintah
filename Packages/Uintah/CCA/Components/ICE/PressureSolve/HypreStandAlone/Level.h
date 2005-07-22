#ifndef __LEVEL_H__
#define __LEVEL_H__

#include <vector>

class Patch;

class Level {
  /*_____________________________________________________________________
    class Level:
    A union of boxes that share the same meshsize and index space. Each
    proc owns several boxes of a level, not necessarily the entire level.
    _____________________________________________________________________*/
public:
  int            _numDims;    // # dimensions
  std::vector<double> _meshSize;   // Meshsize in all dimensions
  std::vector<int>    _resolution; // Size(level) if extends over the full domain
  std::vector<Patch*> _patchList;  // owned by this proc ONLY

  Level(const int numDims,
        const double h);
private:
}; 

#endif // __LEVEL_H__

