/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

#include <Core/Datatypes/Color.h>
#include <Core/Math/MinMax.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

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

Color::Color(double c[3])
: _r(c[0]), _g(c[1]), _b(c[2])
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

Color Color::operator/(double w) const
{
    return Color(_r/w, _g/w, _b/w);
}

Color Color::operator+(const Color& c) const
{
    return Color(_r+c._r, _g+c._g, _b+c._b);
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
CharColor::operator= ( const Color& c )
{
   red = (char)(c.r()*255);
   green = (char)(c.g()*255);
   blue = (char)(c.b()*255);

   return *this;
}


CharColor
CharColor::operator= ( const CharColor& c )
{
  red = c.red;
  green = c.green;
  blue = c.blue;
  return *this;
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

void Pio(Piostream& stream, Color& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p._r);
    Pio(stream, p._g);
    Pio(stream, p._b);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, CharColor& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p.red);
    Pio(stream, p.green);
    Pio(stream, p.blue);
    stream.end_cheap_delim();
}

} // End namespace SCIRun




