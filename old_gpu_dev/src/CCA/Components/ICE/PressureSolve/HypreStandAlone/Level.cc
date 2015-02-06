/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
