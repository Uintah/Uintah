
/*
 *  Color.h: RGB color model
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

class Color {
    double _r, _g, _b;
public:
    Color();
    Color(double, double, double);
    Color(const Color&);
    Color& operator=(const Color&);
    ~Color();

    void get_color(float color[3]);
};

#endif
