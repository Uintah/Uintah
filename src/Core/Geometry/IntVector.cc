
#include <SCICore/Geometry/IntVector.h>
#include <iostream>
using std::ostream;
using SCICore::Geometry::IntVector;

ostream& operator<<(ostream& out, const IntVector& v)
{
  out << "[int " << v.x() << ", " << v.y() << ", " << v.z() << ']';
  return out;
}
