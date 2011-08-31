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


#ifndef Package_Uintah_testprograms_TestBoxGrouper_Box
#define Package_Uintah_testprograms_TestBoxGrouper_Box

#include <Core/Geometry/IntVector.h>

namespace Uintah {
  using namespace SCIRun;

class Box
{
public:
  Box(IntVector low, IntVector high, int id)
    : low_(low), high_(high), id_(id) {}
  
  const IntVector& getLow() const
  { return low_; }

  const IntVector& getHigh() const
  { return high_; }

  int getID() const
  { return id_; }

  int getVolume() const
  { return getVolume(low_, high_); }

  int getArea(int side) const
  {
    int area = 1;
    for (int i = 0; i < 3; i++)
      if (i != side)
	area *= getHigh()[i] - getLow()[i] + 1;
    return area;
  }

  bool isInside(IntVector low, IntVector high) const;
  bool isNeighboring(IntVector low, IntVector high) const;

  static int getVolume(IntVector low, IntVector high)
  { return (high.x() - low.x() + 1) * (high.y() - low.y() + 1) *
      (high.z() - low.z() + 1); }

  static IntVector Min(IntVector low, IntVector high)
  { return SCIRun::Min(low, high); }

  static IntVector Max(IntVector low, IntVector high)
  { return SCIRun::Max(low, high); }
  
private:
  IntVector low_;
  IntVector high_;
  int id_;
};

}

#endif // ndef Package_Uintah_testprograms_TestBoxGrouper_Box
