#include "Level.h"
#include "util.h"
#include <math.h>

Level::Level(const Counter numDims,
             const double& h) {
  /* Domain is assumed to be of size 1.0 x ... x 1.0 and
     1/h[d] is integer for all d. */
  Print("Level::Level() begin\n");
  Print("numDims              = %d\n",numDims);
  _meshSize.resize(0,numDims);
  _resolution.resize(0,numDims);
  Print("_meshSize.getLen()   = %d\n",_meshSize.getLen());
  Print("_resolution.getLen() = %d\n",_resolution.getLen());
  for (Counter d = 0; d < numDims; d++) {
    Print("d = %d\n",d);
    _meshSize[d]   = h;
    _resolution[d] = Counter(floor(1.0/_meshSize[d]));
  }
  Print("Level::Level() end\n");
}
