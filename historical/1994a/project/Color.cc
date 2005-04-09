
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

#include <Color.h>

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
