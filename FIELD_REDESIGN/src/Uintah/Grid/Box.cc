
#include <Uintah/Grid/Box.h>
#include <iostream>

using namespace Uintah;
using namespace std;

Box::Box()
{
}

Box::~Box()
{
}

Box::Box(const Box& copy)
  : d_lower(copy.d_lower), d_upper(copy.d_upper)
{
}

Box::Box(const Point& lower, const Point& upper)
  : d_lower(lower), d_upper(upper)
{
}

Box& Box::operator=(const Box& copy)
{
  d_lower = copy.d_lower;
  d_upper = copy.d_upper;
  return *this;
}

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

//
// $Log$
// Revision 1.2  2000/04/26 06:48:46  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/13 06:51:01  sparker
// More implementation to get this to work
//
//
