
#ifndef COLOR_H
#define COLOR_H 1

//class ostream;
#include <Core/Persistent/Pstreams.h>
#include <iostream>

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
    case 0: return r; BREAK
			case 1: return g; BREAK
					    case 2: return b; BREAK
								default: return 0.0; BREAK
										       }
  }

  inline float& operator[](int i ) {
    switch( i ) {
    case 0: return r; BREAK
			case 1: return g; BREAK
					    case 2: return b; BREAK
								default: return r; BREAK
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
