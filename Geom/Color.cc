
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
    double h=hsv.hue();
    double s=hsv.sat();
    double v=hsv.val();
    // Convert to HSV...
    int hh=(int)(h/360.0);
    h-=hh*360.0;
    double h6=h/60.0;
    int i=(int)h6;
    double f=h6-i;
    double p1=v*(1.0-s);
    double p2=v*(1.0-(s*f));
    double p3=v*(1.0-(s*(1-f)));
    switch(i){
    case 0:
	_r=v;  _g=p3; _b=p1; break;
    case 1:
	_r=p2; _g=v;  _b=p1; break;
    case 2:
	_r=p1; _g=v;  _b=p3; break;
    case 3:
	_r=p1; _g=p2; _b=v;  break;
    case 4:
	_r=p3; _g=p1; _b=v;  break;
    case 5:
	_r=v;  _g=p1; _b=p2; break;
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

HSVColor& HSVColor::operator=(const HSVColor& copy)
{
    _hue=copy._hue; _sat=copy._sat; _val=copy._val;
    return *this;
}
