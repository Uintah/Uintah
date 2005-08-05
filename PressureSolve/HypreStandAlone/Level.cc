#include "Level.h"

#include <math.h>

Level::Level(const Counter numDims,
             const double& h) {
  /* Domain is assumed to be of size 1.0 x ... x 1.0 and
     1/h[d] is integer for all d. */
  _meshSize.resize(0,numDims);
  _resolution.resize(0,numDims);
  for (Counter d = 0; d < numDims; d++) {
    _meshSize[d]   = h;
    _resolution[d] = Counter(floor(1.0/_meshSize[d]));
  }
}
