
/*
 *  View.h:  The camera
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_View_h
#define SCI_Geom_View_h 1

#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class View {
    Point eyep_;
    Point lookat_;
    Vector up_;
    double fov_;

public:
    View();
    View(const Point&, const Point&, const Vector&, double);
    View(const View&);
    ~View();
    View& operator=(const View&);

    void get_viewplane(double aspect, double zdist,
		       Vector& u, Vector& v);
    Point eyespace_to_objspace(const Point& p, double aspect);
    double depth(const Point& p);

    Point eyep() const;
    void eyep(const Point&);
    Point lookat() const;
    void lookat(const Point&);
    Vector up() const;
    void up(const Vector&);
    double fov() const;	
    void fov(double);
};

#endif /* SCI_Geom_View_h */
