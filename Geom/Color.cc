
/*
 *  Color.cc: Simple RGB color model
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Color.h>
#include <Math/MinMax.h>
#include <Classlib/Persistent.h>

Color::Color()
: _r(0), _g(0), _b(0)
{
}

Color::Color(double _r, double _g, double _b)
: _r(_r), _g(_g), _b(_b)
{
}

Color::Color(const Color& c)
: _r(c._r), _g(c._g), _b(c._b)
{
}

Color& Color::operator=(const Color& c)
{
    _r=c._r;
    _g=c._g;
    _b=c._b;
    return *this;
}

Color::~Color()
{
}

void Color::get_color(float color[4])
{
    color[0]=_r;
    color[1]=_g;
    color[2]=_b;
    color[3]=1.0;
}

Color Color::operator*(const Color& c) const
{
    return Color(_r*c._r, _g*c._g, _b*c._b);
}

Color Color::operator*(double w) const
{
    return Color(_r*w, _g*w, _b*w);
}

Color Color::operator+(const Color& c) const
{
    return Color(_r+c._r, _g+c._g, _b+c._b);
}

void Pio(Piostream& stream, Color& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p._r);
    Pio(stream, p._g);
    Pio(stream, p._b);
    stream.end_cheap_delim();
}

Color& Color::operator+=(const Color& c)
{
    _r+=c._r;
    _g+=c._g;
    _b+=c._b;
    return *this;
}

Color::Color(const HSVColor& hsv)
{
   int hh((int)(hsv._hue/360.0));
   double hue(hsv._hue-hh*360.0);
   
   double h6(hue/60.0);
   int i((int)h6);
   double f(h6-i);
   double p1(hsv._val*(1.0-hsv._sat));
   double p2(hsv._val*(1.0-(hsv._sat*f)));
   double p3(hsv._val*(1.0-(hsv._sat*(1-f))));
   switch(i){
   case 0:
      _r=hsv._val; _g=p3;       _b=p1;   break;
   case 1:
      _r=p2;       _g=hsv._val; _b=p1;   break;
   case 2:
      _r=p1;       _g=hsv._val; _b=p3;   break;
   case 3:
      _r=p1;       _g=p2;       _b=hsv._val; break;
   case 4:
      _r=p3;       _g=p1;       _b=hsv._val; break;
   case 5:
      _r=hsv._val; _g=p1;       _b=p2;   break;
   default:
      _r=_g=_b=0;
   }
}

HSVColor::HSVColor()
{
}

HSVColor::HSVColor(double _hue, double _sat, double _val)
: _hue(_hue), _sat(_sat), _val(_val)
{
}

HSVColor::~HSVColor()
{
}

HSVColor::HSVColor(const HSVColor& copy)
: _hue(copy._hue), _sat(copy._sat), _val(copy._val)
{
}

HSVColor::HSVColor(const Color& rgb)
{
   double max(Max(rgb._r,rgb._g,rgb._b));
   double min(Min(rgb._r,rgb._g,rgb._b));
   _sat = ((max == 0.0) ? 0.0 : ((max-min)/max));
   if (_sat != 0.0) {
      double rl((max-rgb._r)/(max-min));
      double gl((max-rgb._g)/(max-min));
      double bl((max-rgb._b)/(max-min));
      if (max == rgb._r) {
	 if (min == rgb._g) _hue = 60.0*(5.0+bl);
	 else _hue = 60.0*(1.0-gl);
      } else if (max == rgb._g) {
	 if (min == rgb._b) _hue = 60.0*(1.0+rl);
	 else _hue = 60.0*(3.0-bl);
      } else {
	 if (min == rgb._r)	_hue = 60.0*(3.0+gl);
	 else _hue = 60.0*(5.0-rl);
      }
   } else {
      _hue = 0.0;
   }
   _val = max;
}

HSVColor& HSVColor::operator=(const HSVColor& copy)
{
    _hue=copy._hue; _sat=copy._sat; _val=copy._val;
    return *this;
}

HSVColor HSVColor::operator*(double w)
{
   return HSVColor(_hue*w,_val*w,_sat*w);
}

HSVColor HSVColor::operator+(const HSVColor& c)
{
   return HSVColor(_hue+c._hue, _sat+c._sat, _val+c._val);
}

/***************************************************
***************************************************/

CharColor::CharColor ()
{
  red = green = blue = 0;
}

CharColor::CharColor ( char a, char b, char c )
{
  red = a;
  green = b;
  blue = c;
}

CharColor::CharColor ( Color& c )
{
  red   = (char)(c.r()*255);
  green = (char)(c.g()*255);
  blue  = (char)(c.b()*255);
}


CharColor
CharColor::operator= ( const Color& c ) const
{
  CharColor f;

  f.red = (char)(c.r()*255);
  f.green = (char)(c.g()*255);
  f.blue = (char)(c.b()*255);

  return f;
}


int
CharColor::operator!= ( const CharColor& c ) const
{
  if ( ( red == c.r() ) && ( green == c.g() ) &&
      ( blue == c.b() ) )
    return 1;
  else
    return 0;
}
