
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
#include <Geom/Color.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

class View {
protected:
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

    // compare 2 views; are they exactly the same?
    int operator==(const View&);
    
    void get_viewplane(double aspect, double zdist,
		       Vector& u, Vector& v);

    void get_normalized_viewplane(Vector& u, Vector& v);
    
    Point eyespace_to_objspace(const Point& p, double aspect);
    Point objspace_to_eyespace(const Point& p, double aspect);

    Point eyespace_to_objspace_ns(const Point& p, double aspect);
    Point objspace_to_eyespace_ns(const Point& p, double aspect);

    double depth(const Point& p);

    Point eyep() const;
    void eyep(const Point&);
    Point lookat() const;
    void lookat(const Point&);
    Vector up() const;
    void up(const Vector&);
    double fov() const;	
    void fov(double);

    friend void Pio (Piostream&, View&);
};

class ExtendedView : public View
{
  int xres_, yres_;
  Color bg_;

public:
  ExtendedView();
  ExtendedView( const View&, int, int, const Color& );
  ExtendedView( const Point&, const Point&, const Vector&, double,
	       int, int, const Color& );
  ExtendedView( const ExtendedView& );

  Color bg() const;
  void bg(const Color&);

  int xres() const;
  void xres(int);
  int yres() const;
  void yres(int);

  friend void Pio( Piostream&, ExtendedView& );
  
  void Print();
  
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:54  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:57  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:56:15  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:15  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:20  dav
// Import sources
//
//

#endif /* SCI_Geom_View_h */

