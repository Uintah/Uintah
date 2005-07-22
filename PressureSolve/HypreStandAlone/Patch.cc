#include "Patch.h"

#include "util.h"

Patch::Patch(const int procID, 
             const int levelID,
             const vector<int>& ilower,
             const vector<int>& iupper)
{
  _procID = procID;
  _levelID = levelID; 
  _ilower = ilower; 
  _iupper = iupper;
  _boundaries.resize(2*_ilower.size());
  vector<int> sz(_ilower.size());
  for (int d = 0; d < _ilower.size(); d++)
    sz[d] = _iupper[d] - _ilower[d] + 1;
  _numCells = prod(sz);
}

