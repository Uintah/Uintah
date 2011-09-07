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


#include <testprograms/TestBoxGrouper/Box.h>

using namespace Uintah;

bool Box::isInside(IntVector low, IntVector high) const
{
  IntVector lowOverlap = Max(low, low_);
  IntVector highOverlap = Min(high, high_);
  return (highOverlap.x() >= lowOverlap.x() &&
	  highOverlap.y() >= lowOverlap.y() &&
	  highOverlap.z() >= lowOverlap.z());
}

bool Box::isNeighboring(IntVector low, IntVector high) const
{
  IntVector lowOverlap = Max(low, low_);
  IntVector highOverlap = Min(high, high_);

  int neighboringSides = 0;
  int overlappingSides = 0;
  for (int i = 0; i < 3; i++) {
    if (highOverlap[i] >= lowOverlap[i])
      overlappingSides++;
    if (highOverlap[i] + 1 == lowOverlap[i])
      neighboringSides++;
  }

  // Note: no corner neighbors allowed -- must have two overlapping sides
  return neighboringSides == 1 && overlappingSides == 2;
}

