#include "Level.h"
#include "util.h"
#include "DebugStream.h"

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
