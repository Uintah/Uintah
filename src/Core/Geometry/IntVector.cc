
#include <Core/Geometry/IntVector.h>
#include <iostream>
using std::ostream;

using namespace SCIRun;

ostream& operator<<(ostream& out, const IntVector& v)
{
  out << "[int " << v.x() << ", " << v.y() << ", " << v.z() << ']';
  return out;
}

namespace junk {
void foo()
{
  IntVector id;
  std::cerr << id;
}
}
