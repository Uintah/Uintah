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
 * GeomTorus.h: Torus objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   January 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_Geom_Torus_h
#define SCI_Geom_Torus_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class GeomTorus : public GeomObj {
public:
    Point cen;
    Vector axis;
    double rad1;
    double rad2;
    int nu;
    int nv;

    Vector zrotaxis;
    double zrotangle;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);

    GeomTorus(int nu=50, int nv=8);
    GeomTorus(const Point&, const Vector&, double, double,
	      int nu=50, int nv=8);
    GeomTorus(const GeomTorus&);
    virtual ~GeomTorus();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class GeomTorusArc : public GeomTorus {
public:
    Vector zero;
    double start_angle;
    double arc_angle;
    Vector yaxis;

    virtual void adjust();
    void move(const Point&, const Vector&, double, double,
	      const Vector& zero, double start_angle, double arc_angle,
	      int nu=50, int nv=8);
    GeomTorusArc(int nu=50, int nv=8);
    GeomTorusArc(const Point&, const Vector&, double, double, 
		 const Vector& zero, double start_angle, double arc_angle,
		 int nu=50, int nv=8);
    GeomTorusArc(const GeomTorusArc&);
    virtual ~GeomTorusArc();

    virtual GeomObj* clone();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Torus_h */
