#include "Patch.h"

#include "util.h"
#include <string>
#include <map>

using namespace std;

map<Patch::BoundaryType, string> Patch::boundaryTypeString; 
bool Patch::initialized = false;

Patch::Patch(const int procID, 
             const int levelID,
             const vector<int>& ilower,
             const vector<int>& iupper)
{
  if (!initialized) {
    init();
    initialized = true;
  }
  _procID = procID;
  _levelID = levelID; 
  _ilower = ilower; 
  _iupper = iupper;
  _boundaries.resize(2*_ilower.size());
  _bc.resize(2*_ilower.size());
  vector<int> sz(_ilower.size());
  for (Counter d = 0; d < _ilower.size(); d++)
    sz[d] = _iupper[d] - _ilower[d] + 1;
  _numCells = prod(sz);
}
