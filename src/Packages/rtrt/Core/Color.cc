
#include <Packages/rtrt/Core/Color.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace rtrt {
  ostream& operator<<(ostream& out, const Color& c)
  {
    out << '[' << c.red() << ", " << c.green() << ", " << c.blue() << ']';
    return out;
  }
} // end namespace rtrt

namespace SCIRun {

void Pio(Piostream& stream, rtrt::Color& p)
{
  stream.begin_cheap_delim();
  Pio(stream, p.r);
  Pio(stream, p.g);
  Pio(stream, p.b);  
  stream.end_cheap_delim();
}

} // end namespace SCIRun
