/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Color_h
#define SCI_project_Color_h 1

#include <Core/share/share.h>

namespace SCIRun {
  class Piostream;

class HSVColor;

class SCICORESHARE Color {
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

    friend SCICORESHARE void Pio( Piostream&, Color& );

    friend class HSVColor;
};

class SCICORESHARE Colorub { // unsigned byte color
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

class SCICORESHARE HSVColor {
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

class SCICORESHARE CharColor {
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

  friend SCICORESHARE void Pio( Piostream&, CharColor& );
};

} // End namespace SCIRun


#endif
