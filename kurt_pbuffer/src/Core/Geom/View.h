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

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/Color.h>

namespace SCIRun {


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
    int operator!=(const View&);
    
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


#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
/* 
cc-1375 CC: REMARK File = ../../Core/Geom/View.h, Line = 71
  The destructor for base class "View" is not virtual.
  */
#pragma set woff 1375
#endif

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

} // End namespace SCIRun

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif


#endif /* SCI_Geom_View_h */

