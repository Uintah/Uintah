
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

#include <Core/share/share.h>
  
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geom/Color.h>

namespace SCIRun {


class SCICORESHARE View {
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

    friend SCICORESHARE void Pio (Piostream&, View&);
};


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/* 
cc-1375 CC: REMARK File = ../../Core/Geom/View.h, Line = 71
  The destructor for base class "View" is not virtual.
  */
#pragma set woff 1375
#endif

class SCICORESHARE ExtendedView : public View
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

  friend SCICORESHARE void Pio( Piostream&, ExtendedView& );
  
  void Print();
  
};

} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif


#endif /* SCI_Geom_View_h */

