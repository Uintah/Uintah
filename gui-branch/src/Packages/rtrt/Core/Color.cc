
#include "Color.h"
#include <iostream>

namespace rtrt {
  ostream& operator<<(ostream& out, const Color& c)
  {
    out << '[' << c.red() << ", " << c.green() << ", " << c.blue() << ']';
    return out;
  }
} // end namespace
