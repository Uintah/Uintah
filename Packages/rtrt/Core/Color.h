/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



#ifndef COLOR_H
#define COLOR_H 1

//class ostream;
#include <Core/Persistent/Pstreams.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace rtrt {
  class Color;
}
namespace SCIRun {
  void Pio(Piostream& stream, rtrt::Color& p);
}
namespace rtrt {

using std::ostream;

// This gets rid of compiler warnings on the SGI.
#if defined(__sgi)
#  define BREAK 
#else
#  define BREAK break;
#endif

class Color {
  float r,g,b;
public:
  inline Color() {
  }
  inline Color(float r, float g, float b) : r(r), g(g), b(b){
  }
  inline Color(double c[3]) : r(c[0]), g(c[1]), b(c[2]) {
  }
  inline ~Color() {
  }
  friend class Pixel;
  friend void SCIRun::Pio(SCIRun::Piostream&, Color&);
  
  inline Color operator*(const Color& c) const {
    return Color(r*c.r, g*c.g, b*c.b);
  }
  inline Color operator+(const Color& c) const {
    return Color(r+c.r, g+c.g, b+c.b);
  }
  inline Color operator-(const Color& c) const {
    return Color(r-c.r, g-c.g, b-c.b);
  }
  inline Color operator*(float s) const {
    return Color(r*s, g*s, b*s);
  }
  inline Color& operator+=(const Color& c) {
    r+=c.r;
    g+=c.g;
    b+=c.b;
    return *this;	
  }
  inline Color& operator-=(const Color& c) {
    r-=c.r;
    g-=c.g;
    b-=c.b;
    return *this;	
  }

  inline float operator[](int i ) const {
    switch( i ) {
    case 0: return r; BREAK;
    case 1: return g; BREAK;
    case 2: return b; BREAK;
    default: return 0.0; BREAK;
    }
  }

  inline float& operator[](int i ) {
    switch( i ) {
    case 0: return r; BREAK;
    case 1: return g; BREAK;
    case 2: return b; BREAK;
    default: return r; BREAK;
    }
  }

  inline float red() const {
    return r;
  }
  inline float green() const {
    return g;
  }
  inline float blue() const {
    return b;
  }
  inline float luminance() const {
    return 0.3*g + 0.6*r + 0.1*b;
  }

  inline float max_component() const {
    float temp = (g > r? g : r);
    return (b > temp? b : temp);
  }


  friend ostream& operator<<(ostream& out, const Color& c);
};

inline Color operator*( float lhs,  const Color& rhs)
{
  return rhs * lhs;
}

inline Color Interpolate(const Color& c1, const Color& c2, double weight)
{
  return c1*weight + c2*(1-weight);
}



} // end namespace rtrt

#endif
