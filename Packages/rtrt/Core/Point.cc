
#include "Point.h"
#include "Vector.h"
#include <iostream>

namespace rtrt {
  ostream& operator<<(ostream& out, const Point& p)
  {
    out << '[' << p.x() << ", " << p.y() << ", " << p.z() << ']';
    return out;
  }
} // end namespace rtrt
