
#include <Packages/Uintah/Core/Grid/Box.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace std;

bool Box::overlaps(const Box& otherbox, double epsilon) const
{
  if(d_lower.x()+epsilon > otherbox.d_upper.x() || d_upper.x() < otherbox.d_lower.x()+epsilon)
    return false;
  if(d_lower.y()+epsilon > otherbox.d_upper.y() || d_upper.y() < otherbox.d_lower.y()+epsilon)
    return false;
  if(d_lower.z()+epsilon > otherbox.d_upper.z() || d_upper.z() < otherbox.d_lower.z()+epsilon)
    return false;
  return true;
}

ostream& operator<<(ostream& out, const Box& b)
{
  out << b.lower() << ".." << b.upper();
  return out;
}

