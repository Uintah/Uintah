/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  Color.h: Simple RGB color model
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 */

#ifndef SCI_project_Color_h
#define SCI_project_Color_h 1

#include <Core/Datatypes/share.h>

namespace SCIRun {
  class Piostream;

class HSVColor;

class SCISHARE Color {
protected:
    double _r, _g, _b;
public:
    Color();
    Color(double, double, double);
    Color(double[3]);
    Color(const Color&);
    Color& operator=(const Color&);
    Color(const HSVColor&);
    ~Color();

    Color operator*(const Color&) const;
    Color operator*(double) const;
    Color operator/(double) const;
    Color operator+(const Color&) const;
    Color& operator+=(const Color&);

    inline int operator==(const Color& c) const {
      return ((_r==c._r)&&(_g==c._g)&&(_b==c._b));
    }

    inline int operator!=(const Color& c) const {
      return ((_r != c._r)||(_g!=c._g)||(_b!=c._b));
    }

    void get_color(float color[4]);
    inline double r() const {return _r;}
    inline double g() const {return _g;}
    inline double b() const {return _b;}

    inline void r( const float v ) { _r = v; }
    inline void g( const float v ) { _g = v; }
    inline void b( const float v ) { _b = v; }

    inline double& operator[](int i) {   
      switch (i) {
      case 0:
	return _r;
      case 1:
	return _g;
      default:
	return _b;
      }
    }
    inline const double& operator[](int i) const
    {
      switch (i) {
      case 0:
	return _r;
      case 1:
	return _g;
      default:
	return _b;
      }
    }

    SCISHARE friend void Pio( Piostream&, Color& );

    friend class HSVColor;
};

class Colorub { // unsigned byte color
  unsigned char data[3]; // data...
public:
  Colorub() {};
  Colorub(Color& c) {
      data[0] = (unsigned char)(c.r()*255);
      data[1] = (unsigned char)(c.g()*255);
      data[2] = (unsigned char)(c.b()*255);
  }; // converts them...

  unsigned char* ptr() { return &data[0]; }; // grab pointer

  inline unsigned char r() const { return data[0]; };
  inline unsigned char g() const { return data[1]; };
  inline unsigned char b() const { return data[2]; };

  // should be enough for now - this is less bandwidth...
};

class SCISHARE HSVColor {
    double _hue;
    double _sat;
    double _val;
public:
    HSVColor();
    HSVColor(double hue, double sat, double val);
    ~HSVColor();
    HSVColor(const HSVColor&);
    HSVColor(const Color&);
    HSVColor& operator=(const HSVColor&);

    // These only affect hue.
    HSVColor operator*(double);
    HSVColor operator+(const HSVColor&);
   
    inline double& operator[](const int i) {   
      switch (i) {
      case 0:
	return _hue;
      case 1:
	return _sat;
      default:
	return _val;
      }
    }

    inline double hue() const {return _hue;}
    inline double sat() const {return _sat;}
    inline double val() const {return _val;}

    friend class Color;
};
/*********************************************************
  This structure holds a simple RGB color in char format.
*********************************************************/

class CharColor {
public:
  char red;
  char green;
  char blue;
  // char alpha;

  CharColor ();
  CharColor ( char a, char b, char c );
  CharColor ( Color& c );
  
    inline double r() const {return red;}
    inline double g() const {return green;}
    inline double b() const {return blue;}
  
  CharColor operator= ( const Color& );
  CharColor operator= ( const CharColor& );

  int operator!= ( const CharColor& ) const;

  friend void Pio( Piostream&, CharColor& );
};

} // End namespace SCIRun


#endif
