#include "Side.h"
#include "Macros.h"

Side& operator++(Side &s)
{
  return s = Side(s+2);
}

std::ostream&
operator << (std::ostream& os, const Side& s)
     // Write side s to the stream os.
{
  if      (s == Left ) os << "Left ";
  else if (s == Right) os << "Right";
  else os << "N/A";
  return os;
}
