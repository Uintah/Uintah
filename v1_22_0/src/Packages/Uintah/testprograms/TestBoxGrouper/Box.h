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
