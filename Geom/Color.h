
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

class HSVColor;
class Piostream;

class Color {
    float _r, _g, _b;
public:
    Color();
    Color(float, float, float);
    Color(const Color&);
    Color& operator=(const Color&);
    Color(const HSVColor&);
    ~Color();

    Color operator*(const Color&) const;
    Color operator*(float) const;
    Color operator+(const Color&) const;
    Color& operator+=(const Color&);

    inline int operator==(const Color& c) const {
      return ((_r==c._r)&&(_g==c._g)&&(_b==c._b));
    }

    inline int operator!=(const Color& c) const {
      return ((_r != c._r)||(_g!=c._g)||(_b!=c._b));
    }

    int InInterval( Color&, double );

    void get_color(float color[4]);
    inline float r() const {return _r;}
    inline float g() const {return _g;}
    inline float b() const {return _b;}

    friend void Pio(Piostream&, Color&);
    friend class HSVColor;

  private:
    int Overlap( double, double, double );

};

class HSVColor {
    float _hue;
    float _sat;
    float _val;
public:
    HSVColor();
    HSVColor(float hue, float sat, float val);
    ~HSVColor();
    HSVColor(const HSVColor&);
    HSVColor(const Color&);
    HSVColor& operator=(const HSVColor&);

    // These only affect hue.
    HSVColor operator*(float);
    HSVColor operator+(const HSVColor&);
   
    inline float hue() const {return _hue;}
    inline float sat() const {return _sat;}
    inline float val() const {return _val;}

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
  
  CharColor operator= ( const Color& ) const;

  int operator!= ( const CharColor& ) const;
};


#endif
