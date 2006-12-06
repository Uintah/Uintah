#include "Level.h"
#include "util.h"
#include "DebugStream.h"
#include "Patch.h"

Level::Level(const Counter numDims,
             const double& h) {
  /* Domain is assumed to be of size 1.0 x ... x 1.0 and
     1/h[d] is integer for all d. */
  funcPrint("Level::Level()",FBegin);
  dbg.setLevel(10);
  dbg << "numDims              = " << numDims << "\n";
  _meshSize.resize(0,numDims);
  _resolution.resize(0,numDims);
  dbg << "_meshSize.getLen()   = " << _meshSize.getLen() << "\n";
  dbg << "_resolution.getLen() = " << _resolution.getLen() << "\n";
  for (Counter d = 0; d < numDims; d++) {
    dbg << "d = " << d << "\n";
    _meshSize[d]   = h;
    _resolution[d] = Counter(floor(1.0/_meshSize[d]));
  }
  funcPrint("Level::Level()",FEnd);
}

std::ostream&
operator << (std::ostream& os, const Level& level)
  // Write the Level to the output stream os.
{
  for (Counter owner = 0; owner < level._patchList.size(); owner++) {
    os << "==== Owned by Proc #" << owner << " ====" << "\n";
    for (Counter index = 0; index < level._patchList[owner].size(); index++) {
      os << "#### Patch #" << index << " ####" << "\n";
      os << *(level._patchList[owner][index]) << "\n";
    } // end for index
  } // end for owner
  return os;
} // end operator <<
