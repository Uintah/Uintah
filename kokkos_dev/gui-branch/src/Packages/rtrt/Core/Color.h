
#ifndef COLOR_H
#define COLOR_H 1

//class ostream;
#include <iostream>
using namespace std;

namespace rtrt {

class Color {
    float r,g,b;
public:
    inline Color() {
    }
    inline Color(float r, float g, float b) : r(r), g(g), b(b){
    }
    inline ~Color() {
    }
    friend class Pixel;
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
